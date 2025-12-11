"""
双层深度强化学习系统
==============================
上层 (Level-1): 任务图划分 - 使用GNN+PPO
下层 (Level-2): 智能体选择和设备部署 - 使用Attention+SAC

"""
import json
from collections import defaultdict, deque
from functools import lru_cache
from typing import Dict, List, Tuple, Any, Optional
import hashlib
import networkx as nx
import numpy as np
from simulation import Device
from sub_partiation import AgentTemplate


WHITE_NOISE = 1e-9  # W

# ---------- 常量 ----------
P_TX_IOT = 0.10  # W   100 mW
P_RX_IOT = 0.10
P_TX_AP = 0.025  # W    25 mW
P_RX_AP = 0.025

h_e, h_c = 3, 5
E_edge, E_core = 37e-9, 12.6e-9  # J/bit
E_cent = 20e-9  # J/bit
E_per_bit_wired = h_e * E_edge + h_c * E_core + E_cent  # ≈2.075e‑5 J/bit

def _get_transmission_time_s(data_size_mb, rate_mbps):
    if rate_mbps <= 0:
        return float("inf")
    return (8.0 * data_size_mb) / rate_mbps


def compute_transmission_delay(src, dst,
                               data_size_mb,
                               bw_req,  # 业务最小带宽需求，可用于速率下限
                               gw_matrix,
                               edge_inter_delay=1, cloud_edge_delay=20,
                               ):
    """
    返回:
    1. 总链路延迟 (ms)
    2. 实际可用带宽 (Mbps), 用于违规检查
    3. 总通信能耗 (J)
    """
    # ---------------- 0. 同设备 ----------------
    if src.id == dst.id:
        same_bw = max(getattr(src, "bandwidth", 0.0), getattr(dst, "bandwidth", 0.0))
        return 0.0, same_bw, 0.0

    # ---------- 类型布尔 ----------
    src_iot = src.type == "Device"
    dst_iot = dst.type == "Device"
    src_edge = src.type == "Edge"
    dst_edge = dst.type == "Edge"
    src_cloud = src.type == "Cloud"
    dst_cloud = dst.type == "Cloud"

    bits = data_size_mb * 8 * 1e6  # 转 bit

    # ---------- 辅助: 计算 UL / DL delay ----------
    def _get_gateway_idx(dev_iot):
        if getattr(dev_iot, "conn_type", "wired") != "wireless":
            return None
        gw_row = gw_matrix[dev_iot.id - 1]
        if not np.any(gw_row > 0):
            return None
        return int(np.where(gw_row > 0)[0][0])

    def _wired_delay_energy(dev_iot):
        rate = dev_iot.bandwidth
        prop_delay = dev_iot.delay
        transmission_time_s = _get_transmission_time_s(data_size_mb, rate)
        total_delay_ms = transmission_time_s * 1000 + prop_delay
        energy_j = E_per_bit_wired * bits
        return total_delay_ms, rate, energy_j

    def ul_delay_energy(dev_iot):
        if getattr(dev_iot, "conn_type", "wired") != "wireless":
            delay_ms, rate, energy_j = _wired_delay_energy(dev_iot)
            return delay_ms, rate, energy_j, None

        rate = dev_iot.bandwidth  # Mbps
        prop_delay = dev_iot.delay  # ms
        gw_idx = _get_gateway_idx(dev_iot)
        if gw_idx is None:
            return float("inf"), 0.0, float("inf"), None

        transmission_time_s = _get_transmission_time_s(data_size_mb, rate)
        total_delay_ms = transmission_time_s * 1000 + prop_delay
        energy_j = (P_TX_IOT + P_RX_AP) * transmission_time_s
        return total_delay_ms, rate, energy_j, gw_idx

    def dl_delay_energy(dev_iot, gw_idx):
        if getattr(dev_iot, "conn_type", "wired") != "wireless":
            return _wired_delay_energy(dev_iot)

        if gw_idx is None:
            return float("inf"), 0.0, float("inf")

        rate = dev_iot.bandwidth
        prop_delay = dev_iot.delay
        transmission_time_s = _get_transmission_time_s(data_size_mb, rate)
        total_delay_ms = transmission_time_s * 1000 + prop_delay
        energy_j = (P_TX_AP + P_RX_IOT) * transmission_time_s
        return total_delay_ms, rate, energy_j

    # ------ 场景 1: IoT → IoT ------
    if src_iot and dst_iot:
        T_ul, rate_ul, E_ul, gw_u = ul_delay_energy(src)
        dst_gw = _get_gateway_idx(dst)
        T_dl, rate_dl, E_dl, = dl_delay_energy(dst, dst_gw)
        same_gw = gw_u is not None and dst_gw is not None and gw_u == dst_gw

        total_delay = T_ul + T_dl + (0 if same_gw else edge_inter_delay)
        total_energy = E_ul + E_dl
        # <<< FIX: 返回瓶颈带宽 >>>
        bottleneck_rate = min(rate_ul, rate_dl)
        return total_delay, bottleneck_rate, total_energy

    # ------ IoT → Edge ------
    if src_iot and dst_edge:
        T_ul, rate_ul, E_ul, _ = ul_delay_energy(src)
        E_wired = E_per_bit_wired * bits
        # <<< FIX: 返回正确的总延迟、实际带宽和总能耗 >>>
        return T_ul + edge_inter_delay, rate_ul, E_ul + E_wired
    # ------ IoT → Cloud ------
    if src_iot and dst_cloud:
        T_ul, rate_ul, E_ul, _ = ul_delay_energy(src)
        E_wired = E_per_bit_wired * bits
        return T_ul + cloud_edge_delay, rate_ul, E_ul + E_wired
    # ------ Edge → IoT ------
    if src_edge and dst_iot:
        gw_idx = _get_gateway_idx(dst)
        T_dl, rate_dl, E_dl = dl_delay_energy(dst, gw_idx)
        E_wired = E_per_bit_wired * bits
        return T_dl + edge_inter_delay, rate_dl, E_dl + E_wired
    # ------ Cloud → IoT ------
    if src_cloud and dst_iot:
        gw_idx = _get_gateway_idx(dst)
        T_dl, rate_dl, E_dl = dl_delay_energy(dst, gw_idx)
        E_wired = E_per_bit_wired * bits
        return T_dl + cloud_edge_delay, rate_dl, E_dl + E_wired
    # --- 有线连接场景 ---
    # 假设有线带宽远大于无线，不会成为瓶颈
    WIRED_BANDWIDTH = 1000.0  # 假设 1 Gbps

    # ------ Edge ↔ Edge ------
    if src_edge and dst_edge:
        E_wired = E_per_bit_wired * bits
        # <<< FIX: 返回统一格式 (delay, bw, energy) >>>
        return edge_inter_delay, WIRED_BANDWIDTH, E_wired

    # ------ Cloud ↔ Edge ------
    if (src_cloud and dst_edge) or (src_edge and dst_cloud):
        E_wired = E_per_bit_wired * bits
        # <<< FIX: 返回统一格式 >>>
        return cloud_edge_delay, WIRED_BANDWIDTH, E_wired

    # 默认情况，不应该发生
    return float('inf'), 0.0, float('inf')


class TaskGraph:
    def __init__(self):
        self.G = nx.DiGraph()
        self.id = 0

    def load_from_json(self, json_data, task_id):
        self.id = task_id
        # 加载节点
        for node in json_data["nodes"]:
            node_id = node["id"]
            node_type = node.get("type", "proc")
            idx = node.get("idx", -1)

            self.G.add_node(node_id, type=node_type, idx=idx)

        # 加载边
        for link in json_data["links"]:
            source = link["source"]
            target = link["target"]

            self.G.add_edge(
                source, target,
                data_size=link["data_size"],  # MB
                bandwidth_req=link["bandwidth_req"],  # Mbps
                latency_req=link["latency_req"]  # ms
            )

    def get_proc_nodes(self):
        return [n for n, attr in self.G.nodes(data=True) if attr['type'] == 'proc']

    def get_sense_nodes(self):
        return [n for n, attr in self.G.nodes(data=True) if attr['type'] == 'sense']

    def get_act_nodes(self):
        return [n for n, attr in self.G.nodes(data=True) if attr['type'] == 'act']

    def get_all_capability_nodes(self):
        return list(self.G.nodes)

    def get_dependencies(self):
        return list(self.G.edges(data=True))


def norm_latency(actual_ms, req_ms):
    if req_ms is None or req_ms <= 0:  # 无约束
        return 0.0
    return max((actual_ms - req_ms) / max(req_ms, 1e-6), 0.0)


def norm_bandwidth(actual_mbps, req_mbps):
    if req_mbps is None or req_mbps <= 0:  # 无约束
        return 0.0
    # 需要带宽，实际不足才违反
    return max((req_mbps - actual_mbps) / max(req_mbps, 1e-6), 0.0)


# <<< NEW: 传感器协作惩罚 >>>
# 当一个模块缺失的传感器能力需要由另一个模块提供时，施加的惩罚值
COLLABORATION_PENALTY_PER_CAP = 1.0
# 为了区分，我们给无法协作的能力一个更高的基础惩罚
UNSOLVABLE_MISMATCH_PENALTY = 10.0
# <<< NEW: 将奖励权重定义为类常量，方便调优 >>>
# 惩罚项权重 (数值越大，代表越不希望发生)
W_CAP_VIOLATION = 2.0  # 能力违反
W_RES_VIOLATION = 2.0  # 资源违反
W_LAT_VIOLATION = 2.5  # 延迟违反 (给予更高权重，因为它影响makespan)
W_BW_VIOLATION = 1.5  # 带宽违反
W_LAT_SHAPING = 0.15  # 建议 0.1 ~ 0.2
W_BW_SHAPING = 0.10  # 建议 0.05 ~ 0.15
# 成本项权重 (作为负奖励)
W_COMM_ENERGY = 0.5  # 通信能耗
W_EXEC_ENERGY = 0.3  # 计算能耗
# —— 终局奖励建议：保持与每个 episode 的 step 累积同量级（~1-3）——
FINISH_BONUS = 40  # 成功小幅正奖励
FAIL_PENALTY = 40  # 失败小幅负奖励
K_LEFTOVER = 20  # 未完成比例的扣分权重
K_SOFT_OVER = 50  # 软约束（lat/bw）超限的扣分权重
K_HARD_VIOL = 0.2  # 硬违规计数的扣分权重
# 进度奖励
R_PROGRESS = 1  # 成功部署一个模块的奖励
# 硬违规的 step 负奖励系数（PPO 需要这个）
P_ILLEGAL = 0.25  # 每个非法子位（compute/sense/act）惩罚
P_CAP = 0.50  # 能力不匹配（按缺失数计）
P_RESOURCE = 0.75  # 资源越界（一次就给这档）
# 截断惩罚
TRUNC_PENALTY = 10.0

# Level-1 环境是否吸收 Level-2 的奖励
L1_TAKES_L2_REWARD_COEF = 0.0  # 建议设为0，或很小的值如0.1

# 通信能耗估计用的有线能耗常数（J/bit），用于归一化分母估计
COMM_ENERGY_PER_BIT_WIRED = 2.075e-5

W_FINAL_MAKESPAN = 12
W_FINAL_ENERGY = 2.5
W_FINAL_PENALTY = 1.0

# <<< NEW: 定义全局路径变量 >>>
LOG_DIR = "logs"
PLOT_DIR = "plots"
MODEL_DIR = "models"


def _dag_fingerprint(G: nx.DiGraph) -> str:
    """
    为 DAG 生成一个稳定的指纹，用于核对计划是否与当前任务图一致。
    包含：节点(含type/idx) + 有向边集合。
    """
    nodes_sig = []
    for n, d in G.nodes(data=True):
        nodes_sig.append((str(n), d.get("type", ""), int(d.get("idx", -1))))
    nodes_sig.sort()

    edges_sig = []
    for u, v, a in G.edges(data=True):
        # 只要结构，不带权（如需带上带宽/数据量，可拼进来）
        edges_sig.append((str(u), str(v)))
    edges_sig.sort()

    m = hashlib.sha256()
    m.update(json.dumps(nodes_sig).encode("utf-8"))
    m.update(json.dumps(edges_sig).encode("utf-8"))
    return m.hexdigest()




# 你提供的用于计算 makespan 的函数，我们把它放在这里以确保依赖完整
def compute_task_finish_time(task_id: int,
                             agent_dag_edges: List[Tuple[Any, Any, Dict]],
                             exec_time_map: Dict[Tuple[int, int], float],
                             edge_delay_map: Dict[Tuple[int, int, int], float]) -> float:
    """
    计算并返回任务的 makespan (T_m_total)。
    """
    succ = defaultdict(list)
    indeg = defaultdict(int)
    modules_in_task = set()

    # 1. 构建邻接表 & 入度
    for u_mod, v_mod, attr in agent_dag_edges:
        succ[u_mod].append(v_mod)
        indeg[v_mod] += 1
        modules_in_task.update([u_mod, v_mod])

    # 2. 拓扑排序计算最早完成时间
    q = deque()
    earliest_finish = {}

    # 找到所有源节点
    for mod in modules_in_task:
        if indeg[mod] == 0:
            earliest_finish[mod] = 0.0
            q.append(mod)

    while q:
        u = q.popleft()
        # 节点u的完成时间 = u的开始时间 + u的执行时间
        base_finish_time = earliest_finish.get(u, 0.0) + exec_time_map.get((task_id, u), 0.0)

        for v in succ[u]:
            # v的开始时间是所有前驱完成时间的最大值
            edge_delay = edge_delay_map.get((task_id, u, v), 0.0)
            candidate_start_time = base_finish_time + edge_delay
            earliest_finish[v] = max(earliest_finish.get(v, 0.0), candidate_start_time)

            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    # 3. 找到所有汇节点（没有后继的节点）的最大完成时间
    max_finish = 0.0
    for mod in modules_in_task:
        if not succ[mod]:
            # 汇节点的完成时间 = 它的开始时间 + 它的执行时间
            finish_time = earliest_finish.get(mod, 0.0) + exec_time_map.get((task_id, mod), 0.0)
            max_finish = max(max_finish, finish_time)

    return max_finish


def evaluation_func_rl(
        placement: Dict[Tuple[int, int], Dict],
        task_graph: 'TaskGraph',  # 使用引号避免循环导入
        agent_lookup: Dict[int, 'AgentTemplate'],
        device_map: Dict[int, 'Device'],
        gw_matrix: np.ndarray,
        exec_time_cache: Optional[Dict[Tuple[int, int], float]] = None,
) -> tuple[float, float, dict[str, float]]:
    """
    构造三相位模块图 (pre/core/post) 计算 makespan，
    同时返回能耗和各类惩罚指标：
      - res_penalty: 资源违反（按比例累计）
      - cap_penalty: 能力违反（计数）
      - bw_penalty : 带宽违反（归一化累计）
      - lat_penalty: 时延违反（归一化累计）
      - load_penalty: 负载均衡惩罚（基于设备利用率 max + 方差）
      - cloud_frac  : 软实例在云上的比例
      - penalty_base: 上述按权重加权后的总惩罚
    """
    @lru_cache(maxsize=100000)
    def cached_transmission_delay(src_id: int, dst_id: int, data_size_mb: float, bw_req: float):
        src = device_map[src_id]
        dst = device_map[dst_id]
        return compute_transmission_delay(
            src=src,
            dst=dst,
            data_size_mb=data_size_mb,
            bw_req=bw_req,
            gw_matrix=gw_matrix,
        )

    # --- 初始化 & 节点→模块映射 ---
    node_to_module_map: Dict[str, int] = {}
    task_id = -1
    for (tid, mod_id), mod_info in placement.items():
        task_id = tid
        for node in mod_info["nodes"]:
            node_to_module_map[node] = mod_id
    if task_id == -1:
        return -1.0, 0.0, {
            "exec_energy": 0.0,
            "comm_energy": 0.0,
            "total_energy": 0.0,
            "res_penalty": 0.0,
            "cap_penalty": 0.0,
            "bw_penalty": 0.0,
            "lat_penalty": 0.0,
            "load_penalty": 0.0,
            "cloud_frac": 0.0,
            "penalty_base": 0.0,
        }

    # 设备查询
    def get_device_for_node(node_id: str) -> Optional[int]:
        module_id = node_to_module_map.get(node_id)
        if module_id is None:
            return None
        mod_place = placement.get((task_id, module_id))
        if mod_place is None:
            return None

        ninfo = task_graph.G.nodes[node_id]
        ntype = ninfo.get("type", "proc")
        if ntype == "proc":
            return mod_place["soft_device"]
        idx = ninfo.get("idx")
        if ntype == "sense":
            return mod_place["sense_map"].get(idx)
        if ntype == "act":
            return mod_place["act_map"].get(idx)
        return None

    # ----------------------------------------------------------------------
    # 1) 模块执行时间 (core phase)
    # ----------------------------------------------------------------------
    exec_time_map: Dict[Tuple[int, Any], float] = {}  # (task_id, phase_node) -> time(s)
    for (tid, module_id), info in placement.items():
        core_time = None
        if exec_time_cache is not None:
            core_time = exec_time_cache.get((info["agent_id"], info["soft_device"]))
        if core_time is None:
            agent = agent_lookup[info["agent_id"]]
            res = device_map[info["soft_device"]].resource
            r_cpu, _, r_gpu, _, _, _ = agent.r
            cpu_cap = float(getattr(res, "cpu", 0.0))
            gpu_cap = float(getattr(res, "gpu", 0.0))
            cpu_time = (r_cpu / cpu_cap) if cpu_cap > 0 else float('inf')
            gpu_time = (r_gpu / gpu_cap) if gpu_cap > 0 else 0.0
            core_time = max(cpu_time, gpu_time)

        n_pre = (module_id, "pre")
        n_core = (module_id, "core")
        n_post = (module_id, "post")

        exec_time_map[(task_id, n_pre)] = 0.0
        exec_time_map[(task_id, n_core)] = float(core_time)
        exec_time_map[(task_id, n_post)] = 0.0

    # ----------------------------------------------------------------------
    # 2) 模块内 S→P / P→A 最大链路时延 (只记录跨设备)
    # ----------------------------------------------------------------------
    intra_pre_ms = defaultdict(float)
    intra_post_ms = defaultdict(float)

    for u, v, attr in task_graph.get_dependencies():
        u_mod = node_to_module_map.get(u)
        v_mod = node_to_module_map.get(v)
        if u_mod is None or v_mod is None:
            continue
        if u_mod != v_mod:
            continue

        du = get_device_for_node(u)
        dv = get_device_for_node(v)
        if du is None or dv is None or du == dv:
            continue

        u_type = task_graph.G.nodes[u].get("type", "proc")
        v_type = task_graph.G.nodes[v].get("type", "proc")

        delay_ms, _, _ = cached_transmission_delay(
            du,
            dv,
            attr.get("data_size", 0.0),
            attr.get("bandwidth_req", 0.0),
        )

        if u_type == "sense" and v_type == "proc":
            intra_pre_ms[u_mod] = max(intra_pre_ms[u_mod], float(delay_ms))
        elif u_type == "proc" and v_type == "act":
            intra_post_ms[u_mod] = max(intra_post_ms[u_mod], float(delay_ms))

    # ----------------------------------------------------------------------
    # 3) 三相位 DAG 的边与延迟
    # ----------------------------------------------------------------------
    agent_dag_edges: List[Tuple[Any, Any, Dict]] = []
    edge_delay_map: Dict[Tuple[int, Any, Any], float] = {}

    # 3a) pre→core、core→post
    for (tid, module_id), _ in placement.items():
        n_pre = (module_id, "pre")
        n_core = (module_id, "core")
        n_post = (module_id, "post")

        agent_dag_edges.append((n_pre, n_core, {}))
        edge_delay_map[(task_id, n_pre, n_core)] = intra_pre_ms[module_id] / 1000.0

        agent_dag_edges.append((n_core, n_post, {}))
        edge_delay_map[(task_id, n_core, n_post)] = intra_post_ms[module_id] / 1000.0

    # 3b) 跨模块 core→core 的最大通信延迟
    inter_core_delay_sec = defaultdict(float)
    for u, v, attr in task_graph.get_dependencies():
        u_mod = node_to_module_map.get(u)
        v_mod = node_to_module_map.get(v)
        if u_mod is None or v_mod is None or u_mod == v_mod:
            continue

        du = get_device_for_node(u)
        dv = get_device_for_node(v)
        if du is None or dv is None:
            continue

        if du == dv:
            delay_sec = 0.0
        else:
            d_ms, _, _ = cached_transmission_delay(
                du,
                dv,
                attr.get("data_size", 0.0),
                attr.get("bandwidth_req", 0.0),
            )
            delay_sec = float(d_ms) / 1000.0

        u_core = (u_mod, "core")
        v_core = (v_mod, "core")
        key = (u_core, v_core)
        inter_core_delay_sec[key] = max(inter_core_delay_sec[key], delay_sec)

    for (u_core, v_core), dsec in inter_core_delay_sec.items():
        agent_dag_edges.append((u_core, v_core, {}))
        edge_delay_map[(task_id, u_core, v_core)] = dsec

    # ----------------------------------------------------------------------
    # 4) 计算 makespan
    # ----------------------------------------------------------------------
    makespan = compute_task_finish_time(
        task_id=task_id,
        agent_dag_edges=agent_dag_edges,
        exec_time_map=exec_time_map,
        edge_delay_map=edge_delay_map
    )

    # ----------------------------------------------------------------------
    # 5) 计算执行能耗
    # ----------------------------------------------------------------------
    exec_energy_J = 0.0
    for (tid, module_id), info in placement.items():
        agent = agent_lookup[info["agent_id"]]
        dev = device_map[info["soft_device"]]
        r_cpu, _, r_gpu, _, _, _ = agent.r

        cpu_cap = float(getattr(dev.resource, "cpu", 0.0))
        gpu_cap = float(getattr(dev.resource, "gpu", 0.0))
        cpu_pow = float(getattr(dev.resource, "cpu_power", 0.0))  # W
        gpu_pow = float(getattr(dev.resource, "gpu_power", 0.0))  # W

        # 你原先就有的时间近似，这里沿用
        cpu_time = (r_cpu / cpu_cap) / 3600.0 if cpu_cap > 0 else 0.0  # s
        gpu_time = (r_gpu / gpu_cap) / 3600.0 if gpu_cap > 0 else 0.0
        exec_energy_J += cpu_time * cpu_pow + gpu_time * gpu_pow

    # ----------------------------------------------------------------------
    # 6) 通信能耗 + 带宽/时延违约：对任务图所有跨设备边累加（不取 max）
    # ----------------------------------------------------------------------
    SENSE_DATA_MB_DEFAULT = 0.5
    CTRL_DATA_MB_DEFAULT = 0.1
    PROC_DATA_MB_DEFAULT = 1.0
    LAT_REQ_DEFAULT = 50.0
    BW_REQ_DEFAULT = 0.0

    comm_energy_J = 0.0
    bw_violation_sum = 0.0
    lat_violation_sum = 0.0

    G = task_graph.G
    for u, v, attr in task_graph.get_dependencies():
        du = get_device_for_node(u)
        dv = get_device_for_node(v)
        # 同设备认为无通信违约，直接跳过
        if du is None or dv is None or du == dv:
            continue

        u_type = G.nodes[u].get("type", "proc")
        v_type = G.nodes[v].get("type", "proc")
        data_mb = attr.get("data_size")
        bw_req = attr.get("bandwidth_req")
        lat_req = attr.get("latency_req")

        if data_mb is None:
            if u_type == "sense" and v_type == "proc":
                data_mb = SENSE_DATA_MB_DEFAULT
            elif u_type == "proc" and v_type == "act":
                data_mb = CTRL_DATA_MB_DEFAULT
            else:
                data_mb = PROC_DATA_MB_DEFAULT
        if bw_req is None:
            bw_req = BW_REQ_DEFAULT
        if lat_req is None:
            # 没有配置时，我们给一个类型相关的宽松默认值
            lat_req = 20.0 if (u_type == "sense" and v_type == "proc") or (
                    u_type == "proc" and v_type == "act") else LAT_REQ_DEFAULT

        delay_ms, actual_bw, ej = cached_transmission_delay(
            du,
            dv,
            data_mb,
            bw_req,
        )
        comm_energy_J += float(ej)

        # 归一化违约：>0 时表示违反程度
        lat_violation_sum += norm_latency(delay_ms, lat_req)
        bw_violation_sum += norm_bandwidth(actual_bw, bw_req)

    # ----------------------------------------------------------------------
    # 7) 资源/能力违约：按模块统计 + 设备级资源使用累计（用于负载评估）
    # ----------------------------------------------------------------------
    res_violation_sum = 0.0
    cap_violation_sum = 0.0

    # 初始化设备使用量字典：device_id -> {cpu, num, gpu, gpu_mem, ram, disk}
    device_usage = {dev_id: {"cpu": 0.0, "num": 0.0, "gpu": 0.0, "gpu_mem": 0.0, "ram": 0.0, "disk": 0.0}
                    for dev_id in device_map.keys()}

    soft_on_cloud_count = 0
    soft_total_count = 0

    for (tid, module_id), info in placement.items():
        agent = agent_lookup[info["agent_id"]]
        dev_soft = device_map.get(info["soft_device"])
        soft_total_count += 1
        if dev_soft is None:
            # 找不到设备，直接记一笔比较重的资源+能力违约
            res_violation_sum += 1.0
            cap_violation_sum += 1.0
            continue

        # 资源向量：顺序要和模板 r 一致
        need_cpu, need_num, need_gpu, need_gpu_mem, need_ram, need_disk = agent.r
        res = dev_soft.resource
        avail_cpu = float(getattr(res, "cpu", 0.0))
        avail_num = float(getattr(res, "num", 0.0))
        avail_gpu = float(getattr(res, "gpu", 0.0))
        avail_gpu_mem = float(getattr(res, "gpu_mem", 0.0))
        avail_ram = float(getattr(res, "ram", 0.0))
        avail_disk = float(getattr(res, "disk", 0.0))

        # 累加到设备使用量（用于后续的利用率计算）
        du = device_usage[dev_soft.id]
        du["cpu"] += float(need_cpu)
        du["num"] += float(need_num)
        du["gpu"] += float(need_gpu)
        du["gpu_mem"] += float(need_gpu_mem)
        du["ram"] += float(need_ram)
        du["disk"] += float(need_disk)

        if getattr(dev_soft, "type", "").lower() == "cloud":
            soft_on_cloud_count += 1

        # 对每个维度，超出的比例作为一部分违约（max((need-avail)/avail, 0)）
        def over_ratio(need, avail):
            if avail <= 0:
                return 1.0 if need > 0 else 0.0
            return max((need - avail) / avail, 0.0)

        res_violation_sum += (
                over_ratio(need_cpu, avail_cpu) +
                over_ratio(need_num, avail_num) +
                over_ratio(need_gpu, avail_gpu) +
                over_ratio(need_gpu_mem, avail_gpu_mem) +
                over_ratio(need_ram, avail_ram) +
                over_ratio(need_disk, avail_disk)
        )

        # 能力匹配检查：soft/sense/act
        dev_soft_caps = getattr(dev_soft, "soft_cap", set())
        for cap in agent.C_soft:
            if cap not in dev_soft_caps:
                cap_violation_sum += 1.0

        for cap in agent.C_sense:
            dev_id = info["sense_map"].get(cap)
            dev = device_map.get(dev_id)
            if dev is None:
                cap_violation_sum += 1.0
                continue
            dev_sense_caps = getattr(dev, "sense_cap", set())
            if cap not in dev_sense_caps:
                cap_violation_sum += 1.0

        for cap in agent.C_act:
            dev_id = info["act_map"].get(cap)
            dev = device_map.get(dev_id)
            if dev is None:
                cap_violation_sum += 1.0
                continue
            dev_act_caps = getattr(dev, "act_cap", set())
            if cap not in dev_act_caps:
                cap_violation_sum += 1.0

    # ----------------------------------------------------------------------
    # 8) 设备利用率与负载惩罚（load_penalty）
    #    - 每个设备的利用率：对可用维度计算 used/avail 的均值（不可用维度忽略）
    #    - load_penalty = max_util + var(util)
    # ----------------------------------------------------------------------
    util_list = []
    for dev_id, usage in device_usage.items():
        dev = device_map.get(dev_id)
        if dev is None:
            continue
        res = dev.resource
        # 维度集合：cpu(num cores), gpu_mem, ram, disk
        dims = []
        # cpu cores
        cap_cpu = float(getattr(res, "cpu", 0.0))
        if cap_cpu > 0:
            dims.append(usage["cpu"] / cap_cpu)
        # gpu memory
        cap_gpu_mem = float(getattr(res, "gpu_mem", 0.0))
        if cap_gpu_mem > 0:
            dims.append(usage["gpu_mem"] / cap_gpu_mem)
        # ram
        cap_ram = float(getattr(res, "ram", 0.0))
        if cap_ram > 0:
            dims.append(usage["ram"] / cap_ram)
        # disk
        cap_disk = float(getattr(res, "disk", 0.0))
        if cap_disk > 0:
            dims.append(usage["disk"] / cap_disk)

        if not dims:
            continue
        # 利用率下限为0，上限可大于1（代表超载）
        util = float(np.mean(dims))
        util_list.append(util)

    if util_list:
        util_arr = np.array(util_list, dtype=float)
        max_util = float(np.max(util_arr))
        mean_util = float(np.mean(util_arr))
        var_util = float(np.mean((util_arr - mean_util) ** 2))
        load_penalty = max_util + var_util
    else:
        max_util = 0.0
        mean_util = 0.0
        var_util = 0.0
        load_penalty = 0.0

    # 云上 soft 实例比例
    cloud_frac = (soft_on_cloud_count / soft_total_count) if soft_total_count > 0 else 0.0

    # ----------------------------------------------------------------------
    # 9) 合成惩罚（可调权重）
    # ----------------------------------------------------------------------
    # 权重（可调）
    LOAD_PENALTY_WEIGHT = 200.0    # 负载均衡惩罚权重（越大越希望分散）
    CLOUD_PENALTY_WEIGHT = 1e4    # 抑制把太多放云上，按需调小/调大
    # 资源/能力/带宽/延迟 四类原始惩罚使用已有常量（如果你在外部模块定义了 W_*，也可引用）
    # 这里给一套默认权重（可以和你整体 alpha/beta 一起调）
    W_RES = 1.0
    W_CAP = 1.0
    W_BW = 1.0
    W_LAT = 1.0

    res_penalty = float(res_violation_sum)
    cap_penalty = float(cap_violation_sum)
    bw_penalty = float(bw_violation_sum)
    lat_penalty = float(lat_violation_sum)

    penalty_base = (
        W_RES * res_penalty +
        W_CAP * cap_penalty +
        W_BW * bw_penalty +
        W_LAT * lat_penalty +
        LOAD_PENALTY_WEIGHT * float(load_penalty) +
        CLOUD_PENALTY_WEIGHT * float(cloud_frac)
    )

    # ----------------------------------------------------------------------
    # 10) 组装 detail 并返回
    # ----------------------------------------------------------------------
    energy_bd = {
        "exec": float(exec_energy_J),
        "comm": float(comm_energy_J),
        "total": float(exec_energy_J + comm_energy_J),
        "res_violation": res_penalty,
        "cap_violation": cap_penalty,
        "bw_violation": bw_penalty,
        "lat_violation": lat_penalty,
        "max_util": max_util,
        "mean_util": mean_util,
        "var_util": var_util,
        "load_penalty": float(load_penalty),
        "cloud_frac": float(cloud_frac),
        "penalty_base": float(penalty_base),
    }

    return float(makespan), float(energy_bd["total"]), energy_bd



