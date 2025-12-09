"""
双层深度强化学习系统
==============================
上层 (Level-1): 任务图划分 - 使用GNN+PPO
下层 (Level-2): 智能体选择和设备部署 - 使用Attention+SAC

"""
import json
import os
import random
from collections import defaultdict, deque
import time
from typing import Dict, List, Tuple, Any, Optional, cast
import hashlib
import gymnasium as gym
import networkx as nx
import numpy as np
import pandas as pd
import torch

from gymnasium import spaces
from gymnasium.utils import seeding

from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor  # 在文件顶部添加导入

from cSAC import CSACConfig, ConstrainedDiscreteSAC, _to_torch_obs
from simulation import Device, device_number, Resource
from sub_partiation import AgentTemplate
from partiation import parse_capability_field

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


def load_infrastructure(config_path="infrastructure_config.json", gw_path="gw_matrix.npy"):
    # 读取 JSON 文件
    with open(config_path, "r") as f:
        data = json.load(f)

    # 解析云服务器
    cloud_data = data["cloud_server"]
    cloud = Device(
        type=cloud_data["type"],
        id=cloud_data["id"],
        resource=Resource(**cloud_data["resource"]),
        act_cap=set(cloud_data["act_cap"]),
        sense_cap=set(cloud_data["sense_cap"]),
        soft_cap=set(cloud_data["soft_cap"]),
        bandwidth=cloud_data["bandwidth"],
        delay=cloud_data["delay"],
    )

    # 解析 IoT 设备列表
    device_list = []
    for dev in data["device_list"]:
        device_list.append(Device(
            type=dev["type"],
            id=dev["id"],
            resource=Resource(**dev["resource"]),
            act_cap=set(dev["act_cap"]),
            sense_cap=set(dev["sense_cap"]),
            soft_cap=set(dev["soft_cap"]),
            bandwidth=dev["bandwidth"],
            delay=dev["delay"],
        ))

    # 解析边缘服务器列表
    edge_list = []
    for edge in data["edge_server_list"]:
        edge_list.append(Device(
            type=edge["type"],
            id=edge["id"],
            resource=Resource(**edge["resource"]),
            act_cap=set(edge["act_cap"]),
            sense_cap=set(edge["sense_cap"]),
            soft_cap=set(edge["soft_cap"]),
            bandwidth=edge["bandwidth"],
            delay=edge["delay"],
        ))

    # 读取网关矩阵
    gw_matrix = np.load(gw_path)

    return cloud, device_list, edge_list, gw_matrix


def build_agent_lookup(df) -> Dict[int, AgentTemplate]:
    lookup = {}
    for _, row in df.iterrows():
        agent_id = row["Agent ID"]
        C_sense = parse_capability_field(row["Sense Capabilities"])
        C_act = parse_capability_field(row["Act Capabilities"])
        C_soft = parse_capability_field(row["Soft Capabilities"])
        r = (
            row["CPU (FLOPs)"] * 1e9,
            row["CPU Count"],
            row["GPU (FLOPs)"] * 1e12,
            row["GPU Memory (GB)"],
            row["RAM (GB)"],
            row["Disk (GB)"]
        )
        lookup[agent_id] = AgentTemplate(agent_id, C_sense, C_act, C_soft, r)
    return lookup


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
        return 0.0, True, 0.0

    # ---------- 类型布尔 ----------
    src_iot = src.type == "Device"
    dst_iot = dst.type == "Device"
    src_edge = src.type == "Edge"
    dst_edge = dst.type == "Edge"
    src_cloud = src.type == "Cloud"
    dst_cloud = dst.type == "Cloud"

    bits = data_size_mb * 8 * 1e6  # 转 bit

    # ---------- 辅助: 计算 UL / DL delay ----------
    # ------- 子函数：UL / DL -------
    # ------ UL / DL helper ------
    # ------ UL / DL helper (已修正) ------

    def ul_delay_energy(dev_iot):
        # <<< FIX: 使用 dev_iot 的属性，而不是全局的 src >>>
        rate = dev_iot.bandwidth  # Mbps
        prop_delay = dev_iot.delay  # ms
        gw_row = gw_matrix[dev_iot.id - 1]
        gw_idx = int(np.where(gw_row > 0)[0][0]) if np.any(gw_row > 0) else -1

        # <<< FIX: 正确计算传输时间、总延迟和能耗 >>>
        transmission_time_s = _get_transmission_time_s(data_size_mb, rate)
        total_delay_ms = transmission_time_s * 1000 + prop_delay
        energy_j = (P_TX_IOT + P_RX_AP) * transmission_time_s

        # <<< FIX: 返回实际速率和以焦耳为单位的能耗 >>>
        return total_delay_ms, rate, energy_j, gw_idx

    def dl_delay_energy(dev_iot, gw_idx):
        # <<< FIX: 使用 dev_iot 的属性 >>>
        # 注意：下行速率可能与上行不同，这里为简化假设相同，实际可从dev_iot或gw获取
        rate = dev_iot.bandwidth
        prop_delay = dev_iot.delay

        # <<< FIX: 正确计算 >>>
        transmission_time_s = _get_transmission_time_s(data_size_mb, rate)
        total_delay_ms = transmission_time_s * 1000 + prop_delay
        energy_j = (P_TX_AP + P_RX_IOT) * transmission_time_s

        # <<< FIX: 返回实际速率和以焦耳为单位的能耗 >>>
        return total_delay_ms, rate, energy_j

    # ------ 场景 1: IoT → IoT ------
    if src_iot and dst_iot:
        T_ul, rate_ul, E_ul, gw_u = ul_delay_energy(src)
        T_dl, rate_dl, E_dl, = dl_delay_energy(dst, gw_u)
        same_gw = gw_u == np.where(gw_matrix[dst.id - 1] > 0)[0][0]

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
        gw_idx = np.where(gw_matrix[dst.id - 1] > 0)[0][0]
        T_dl, rate_dl, E_dl = dl_delay_energy(dst, gw_idx)
        E_wired = E_per_bit_wired * bits
        return T_dl + edge_inter_delay, rate_dl, E_dl + E_wired
    # ------ Cloud → IoT ------
    if src_cloud and dst_iot:
        gw_idx = np.where(gw_matrix[dst.id - 1] > 0)[0][0]
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


def serialize_partition_plan(G: nx.DiGraph, plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    新式版本：serialize_partition_plan(G, plan)

    plan 需要包含:
      - "task_id": int
      - "objective": float (可选)
      - "modules": [
            {
              "module_id": int,
              "agent_id": int,
              "nodes": Iterable[str or int]
            }, ...
        ]

    返回统一 JSON friendly 结构:
    {
      "fingerprint": <str>,
      "task_id": <int>,
      "objective": <float or None>,
      "modules": [
        {"module_id": int, "agent_id": int, "nodes": [str, ...]}, ...
      ]
    }
    """

    def _norm_nodes(seq) -> List[str]:
        # 节点ID统一转字符串 & 排序
        return sorted([str(x) for x in seq])

    out_modules: List[Dict[str, Any]] = []
    for m in plan.get("modules", []):
        out_modules.append({
            "module_id": int(m["module_id"]),
            "agent_id": int(m.get("agent_id", -1)),
            "nodes": _norm_nodes(m["nodes"]),
        })
    out_modules.sort(key=lambda x: x["module_id"])

    return {
        "fingerprint": _dag_fingerprint(G),
        "task_id": int(plan.get("task_id", -1)),
        "objective": (
            None if plan.get("objective") is None
            else float(plan["objective"])
        ),
        "modules": out_modules,
    }


def save_partition_plans(plans_by_task: Dict[int, Dict[str, Any]], path: str):
    """
    plans_by_task: { task_id -> serialized_plan_dict }
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plans_by_task, f, ensure_ascii=False, indent=2)


def load_partition_plans(path: str) -> Dict[int, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # 兼容两种格式：字典或列表
    if isinstance(raw, list):
        return {int(p["task_id"]): p for p in raw}
    elif isinstance(raw, dict):
        # 可能是 {task_id: plan} 或 {"plans":[...]}
        if "plans" in raw and isinstance(raw["plans"], list):
            return {int(p["task_id"]): p for p in raw["plans"]}
        return {int(k): v for k, v in raw.items()}
    else:
        raise ValueError("Unknown JSON format for precomputed plans")


def realize_plan_to_modules_queue(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    将序列化的 plan 转回 env 的 modules_queue 结构。
    """
    out = []
    for m in sorted(plan["modules"], key=lambda x: x["module_id"]):
        out.append({
            "module_id": int(m["module_id"]),
            "agent_id": int(m["agent_id"]),
            "nodes": set(m["nodes"]),  # env 内部用 set
        })
    return out


# --- 全局辅助函数 ---
def get_module_capabilities(G: nx.DiGraph, nodes: set) -> Tuple[set, set, set]:
    C_sense, C_act, C_soft = set(), set(), set()
    for node in nodes:
        if node not in G.nodes:
            continue
        node_type = G.nodes[node].get("type")
        idx = G.nodes[node].get("idx")
        if node_type == "sense":
            C_sense.add(idx)
        elif node_type == "act":
            C_act.add(idx)
        elif node_type == "proc":
            C_soft.add(idx)
    return C_sense, C_act, C_soft


def get_agent_capabilities(agent: AgentTemplate) -> Tuple[set, set, set]:
    return agent.C_sense, agent.C_act, agent.C_soft


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
        gw_matrix: np.ndarray
) -> tuple[float, float, float] | tuple[float, float, dict[str, float]]:
    """
    方案B：构造"三相位"模块图（pre/core/post），
    - pre→core 承载模块内 S→P 的最大延迟
    - core 承载执行时间
    - core→post 承载模块内 P→A 的最大延迟
    - 跨模块边统一用 U(core)→V(core) 承载通信延迟
    仅返回 makespan，能耗与惩罚置 0.0
    """
    # --- 初始化 ---
    # --- 映射：节点 -> 模块 ---
    node_to_module_map: Dict[str, int] = {}
    task_id = -1
    for (tid, mod_id), mod_info in placement.items():
        task_id = tid
        for node in mod_info["nodes"]:
            node_to_module_map[node] = mod_id
    if task_id == -1:
        return -1.0, 0.0, 0.0  # 空方案

    # 2) 辅助：查节点落在哪个设备
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

    # --- 1) 每个模块的执行时间（core phase） ---
    exec_time_map: Dict[Tuple[int, Any], float] = {}  # (task_id, phase_node) -> time(s)
    for (tid, module_id), info in placement.items():
        agent = agent_lookup[info["agent_id"]]
        res = device_map[info["soft_device"]].resource
        r_cpu, _, r_gpu, _, _, _ = agent.r
        cpu_cap = float(getattr(res, "cpu", 0.0))
        gpu_cap = float(getattr(res, "gpu", 0.0))
        cpu_time = (r_cpu / cpu_cap) if cpu_cap > 0 else 0.0
        gpu_time = (r_gpu / gpu_cap) if gpu_cap > 0 else 0.0
        core_time = max(cpu_time, gpu_time)

        n_pre, n_core, n_post = (module_id, "pre"), (module_id, "core"), (module_id, "post")
        exec_time_map[(task_id, n_pre)] = 0.0
        exec_time_map[(task_id, n_core)] = float(core_time)
        exec_time_map[(task_id, n_post)] = 0.0
    # --- 2) 统计模块内 S→P / P→A 的最大链路时延（仅跨设备才有时延） ---
    intra_pre_ms = defaultdict(float)  # 模块内 S→P
    intra_post_ms = defaultdict(float)  # 模块内 P→A
    for u, v, attr in task_graph.get_dependencies():
        u_mod, v_mod = node_to_module_map.get(u), node_to_module_map.get(v)
        if u_mod != v_mod or u_mod is None:
            continue
        du, dv = get_device_for_node(u), get_device_for_node(v)
        if du is None or dv is None or du == dv:
            continue
        delay_ms, _, _ = compute_transmission_delay(
            src=device_map[du], dst=device_map[dv],
            data_size_mb=attr.get("data_size", 0.0),
            bw_req=attr.get("bandwidth_req", 0.0),
            gw_matrix=gw_matrix
        )
        u_type, v_type = task_graph.G.nodes[u].get("type"), task_graph.G.nodes[v].get("type")
        if u_type == "sense" and v_type == "proc":
            intra_pre_ms[u_mod] = max(intra_pre_ms[u_mod], delay_ms)
        elif u_type == "proc" and v_type == "act":
            intra_post_ms[u_mod] = max(intra_post_ms[u_mod], delay_ms)

    # --- 3) 组装三相位图的边与延迟 ---
    agent_dag_edges: List[Tuple[Any, Any, Dict]] = []  # (phase_u, phase_v, {})
    edge_delay_map: Dict[Tuple[int, Any, Any], float] = {}  # (task_id, phase_u, phase_v) -> delay(s)
    # 3a) 模块内 pre→core、core→post（ms -> s）
    for (tid, module_id), _ in placement.items():
        n_pre = (module_id, "pre")
        n_core = (module_id, "core")
        n_post = (module_id, "post")

        agent_dag_edges.append((n_pre, n_core, {}))
        edge_delay_map[(task_id, n_pre, n_core)] = intra_pre_ms[module_id] / 1000.0

        agent_dag_edges.append((n_core, n_post, {}))
        edge_delay_map[(task_id, n_core, n_post)] = intra_post_ms[module_id] / 1000.0
    # 3b) 跨模块边：统一用 U(core) → V(core)，延迟取该边跨设备通信（ms -> s）
    # 若同一模块对之间多条边，取最大延迟（关键路径）
    # --- 3b) 跨模块边（含阻塞延迟） ---
    inter_core_delay_sec = defaultdict(float)
    incoming_trans_delay = defaultdict(list)
    for u, v, attr in task_graph.get_dependencies():
        u_mod, v_mod = node_to_module_map.get(u), node_to_module_map.get(v)
        if u_mod is None or v_mod is None or u_mod == v_mod:
            continue

        du, dv = get_device_for_node(u), get_device_for_node(v)
        if du is None or dv is None:
            continue

        d_ms, _, _ = compute_transmission_delay(
            src=device_map[du], dst=device_map[dv],
            data_size_mb=attr.get("data_size", 1.0),
            bw_req=attr.get("bandwidth_req", 0.0),
            gw_matrix=gw_matrix
        )
        delay_sec = float(d_ms) / 1000.0

        # ✅ 模块阻塞延迟
        src_exec = exec_time_map.get((task_id, (u_mod, "core")), 0.0)
        # 若模块内存在多个proc节点，则认为阻塞完整执行时间
        num_proc = len([n for n in task_graph.G.nodes()
                        if node_to_module_map.get(n) == u_mod and
                        task_graph.G.nodes[n].get("type") == "proc"])
        block_wait = src_exec if num_proc > 1 else 0.0
        total_delay = delay_sec + block_wait

        u_core, v_core = (u_mod, "core"), (v_mod, "core")
        inter_core_delay_sec[(u_core, v_core)] = max(inter_core_delay_sec[(u_core, v_core)], total_delay)
        incoming_trans_delay[v_core].append(total_delay)
    for (u_core, v_core), dsec in inter_core_delay_sec.items():
        agent_dag_edges.append((u_core, v_core, {}))
        edge_delay_map[(task_id, u_core, v_core)] = dsec
    # --- 3c) 模块同步启动等待（输入最大延迟） ---
    sync_wait_map = {m: max(v) if v else 0.0 for m, v in incoming_trans_delay.items()}
    for (tid, module_id), _ in placement.items():
        n_core = (module_id, "core")
        if n_core in sync_wait_map:
            exec_time_map[(task_id, n_core)] += sync_wait_map[n_core]
    # --- 4) 计算 makespan ---
    makespan = compute_task_finish_time(
        task_id=task_id,
        agent_dag_edges=agent_dag_edges,
        exec_time_map=exec_time_map,
        edge_delay_map=edge_delay_map
    )
    # --- 5) 能耗计算 ---
    exec_energy_J, comm_energy_J = 0.0, 0.0
    # (a) 执行能耗
    for (tid, module_id), info in placement.items():
        agent = agent_lookup[info["agent_id"]]
        dev = device_map[info["soft_device"]]
        r_cpu, _, r_gpu, _, _, _ = agent.r
        cpu_cap = float(getattr(dev.resource, "cpu", 0.0))
        gpu_cap = float(getattr(dev.resource, "gpu", 0.0))
        cpu_pow = float(getattr(dev.resource, "cpu_power", 0.0))
        gpu_pow = float(getattr(dev.resource, "gpu_power", 0.0))
        cpu_time = (r_cpu / cpu_cap) if cpu_cap > 0 else 0.0
        gpu_time = (r_gpu / gpu_cap) if gpu_cap > 0 else 0.0
        exec_energy_J += cpu_time * cpu_pow + gpu_time * gpu_pow

    # (b) 通信能耗
    for u, v, attr in task_graph.get_dependencies():
        du, dv = get_device_for_node(u), get_device_for_node(v)
        if du is None or dv is None or du == dv:
            continue
        src, dst = device_map[du], device_map[dv]
        _, _, ej = compute_transmission_delay(
            src=src, dst=dst,
            data_size_mb=attr.get("data_size", 1.0),
            bw_req=attr.get("bandwidth_req", 0.0),
            gw_matrix=gw_matrix
        )
        # 有线能耗权重降低
        if getattr(src, "conn_type", "wired") == "wired" and getattr(dst, "conn_type", "wired") == "wired":
            ej *= 0.1
        comm_energy_J += ej

    energy_bd = {
        "exec": float(exec_energy_J),
        "comm": float(comm_energy_J),
        "total": float(exec_energy_J + comm_energy_J),
    }

    return float(makespan), float(energy_bd["total"]), energy_bd


def _calculate_module_fitness_collab(
        nodes: set,
        G: nx.DiGraph,
        agent_lookup: Dict[int, AgentTemplate],
        cut_penalty_coef: float = 0.02,
) -> float:
    """
    计算“候选模块”的适应度（越小越好）。
    - 软能力(C_soft)与执行器(C_act)缺失：不可协作 => 大惩罚（UNSOLVABLE_MISMATCH_PENALTY）
    - 传感能力(C_sense)缺失：可跨模块协作 => 小惩罚（COLLABORATION_PENALTY_PER_CAP）
    - cut 惩罚：轻微惩罚跨模块边，鼓励把强耦合子图放一起
    - size 惩罚：模块越小越惩罚，促使模块更“饱满”
    """
    if not nodes:
        return float('inf')

    # 1) 聚合模块所需能力
    C_sense, C_act, C_soft = get_module_capabilities(G, nodes)
    if not (C_sense or C_act or C_soft):
        # 没能力需求的“空壳”，避免被选中
        return float('inf')

    # 2) 在所有智能体模板里找“惩罚最小”的一个（乐观假设最佳匹配）
    best_agent_penalty = float('inf')
    for agent in agent_lookup.values():
        miss_soft = len(C_soft - agent.C_soft)  # 不可协作
        miss_act = len(C_act - agent.C_act)  # 不可协作
        miss_sense_collab = len(C_sense - agent.C_sense)  # 可协作

        penalty = (
                (miss_soft + miss_act) * UNSOLVABLE_MISMATCH_PENALTY +
                miss_sense_collab * COLLABORATION_PENALTY_PER_CAP
        )
        if penalty < best_agent_penalty:
            best_agent_penalty = penalty

    # 3) 轻量 cut 惩罚：模块外连接越多，惩罚越大（非常小的系数，避免喧宾夺主）
    #   你也可以把它关掉：把 cut_penalty_coef 设为 0 即可
    cut_edges = 0
    for u in nodes:
        # 出边到模块外
        for v in G.successors(u):
            if v not in nodes:
                cut_edges += 1
        # 入边来自模块外
        for v in G.predecessors(u):
            if v not in nodes:
                cut_edges += 1
    # 归一化到 [0,1] 范围的一个粗略指标
    possible_edges = max(1, len(nodes) * max(1, len(G.nodes) - len(nodes)))
    cut_penalty = cut_penalty_coef * (cut_edges / possible_edges)

    # 4) 模块大小的反比惩罚（鼓励更饱满）
    size_penalty = (1.0 / len(nodes)) * 0.1

    return best_agent_penalty + cut_penalty + size_penalty


def partition_with_greedy_algorithm(
        task_graph: TaskGraph,
        agent_lookup: Dict[int, AgentTemplate],
        max_module_size: int = 8
) -> List[set]:
    """
    使用“协作友好”的贪心增长算法对任务图进行划分。

    差异点：
      - 适应度函数改为 _calculate_module_fitness_collab（允许传感协作）
      - 生长时只接受能显著降低适应度的邻居，否则停止
    """
    G = task_graph.G
    unassigned = set(G.nodes())
    final_modules: List[set] = []

    while unassigned:
        # 1) 选种子：优先 'proc'
        proc_nodes = [n for n in unassigned if G.nodes[n].get("type") == "proc"]
        if proc_nodes:
            seed = random.choice(proc_nodes)
        else:
            # 无 proc 时随便挑一个
            seed = next(iter(unassigned))

        current = {seed}
        unassigned.remove(seed)

        # 2) 贪心生长
        while len(current) < max_module_size:
            # 候选 = 当前模块所有邻居 ∩ 未分配
            neighbors = set()
            for n in current:
                neighbors.update(G.successors(n))
                neighbors.update(G.predecessors(n))
            candidates = neighbors & unassigned
            if not candidates:
                break

            # 当前适应度
            base_fit = _calculate_module_fitness_collab(current, G, agent_lookup)
            best_fit = base_fit
            best_cand = None

            for c in candidates:
                tmp = current | {c}
                fit = _calculate_module_fitness_collab(tmp, G, agent_lookup)
                if fit < best_fit:
                    best_fit = fit
                    best_cand = c

            # 若没有候选能改进，停止生长
            if best_cand is None:
                break

            # 接纳最优候选
            current.add(best_cand)
            unassigned.remove(best_cand)

        final_modules.append(current)

    return final_modules


def match_agent_for_module(
        G: nx.DiGraph,
        nodes: set,
        agent_lookup: Dict[int, AgentTemplate],
        strict: bool = True
) -> Tuple[int, Dict[str, Any]]:
    C_sense, C_act, C_soft = get_module_capabilities(G, nodes)
    best_agent_id = None
    best_score = float("inf")
    best_info = {"perfect": False, "miss_soft": 0, "miss_act": 0, "miss_sense": 0}

    BIG = 1e6
    for aid, agent in agent_lookup.items():
        miss_soft = len(C_soft - agent.C_soft)
        miss_act = len(C_act - agent.C_act)
        miss_sense = len(C_sense - agent.C_sense)
        if strict and (miss_soft == 0 and miss_act == 0 and miss_sense == 0):
            return aid, {"perfect": True, "miss_soft": 0, "miss_act": 0, "miss_sense": 0}
        score = miss_soft * BIG + miss_act * BIG + miss_sense
        if score < best_score:
            best_score = score
            best_agent_id = aid
            best_info = {"perfect": (miss_soft == 0 and miss_act == 0 and miss_sense == 0),
                         "miss_soft": miss_soft, "miss_act": miss_act, "miss_sense": miss_sense}
    return best_agent_id, best_info



def train_tripart_csac(total_steps=200_000, seed=42, K_s_max=6, K_a_max=6,
                       precomputed_plans_path: Optional[str] = "plans.json"):
    env = create_tripart_env(
        seed=seed, K_s_max=K_s_max, K_a_max=K_a_max,
        max_module_size=5, min_module_size=1,
        lns_rounds=300, grasp_runs=40, grasp_rcl_k=3,
        precomputed_plans_path=precomputed_plans_path
    )
    base_env = env.envs[0]  # Monitor 包了一层
    obs, _ = base_env.reset(seed=seed)
    # 通过动作空间获取每头可选设备数（Monitor 包装后同样可用）
    nvec = base_env.action_space.nvec
    num_devices = int(nvec[1])  # 第 1 维才是计算设备头

    cfg = CSACConfig(
        buffer_size=200_000,
        batch_size=256,
        gamma=0.99, tau=0.005, lr=3e-4,
        alpha_init=0.2, target_entropy_per_head=-np.log(max(1, num_devices)),
        lambda_lr=5e-3,
        constraint_keys=["latency", "bandwidth", "resource_violation", "illegal_caps", "cap_violation"],
        cost_limits={"latency": 5e-2, "bandwidth": 5e-2, "resource_violation": 0, "illegal_caps": 0,
                     "cap_violation": 0}, warmup_steps=10_000, updates_per_step=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    algo = ConstrainedDiscreteSAC(base_env, cfg)

    rng = np.random.default_rng(seed)
    ep_ret, ep_len = 0.0, 0
    episode_idx = 1
    # === NEW: 日志器 ===
    run_logger = TrainRunLogger(log_dir=LOG_DIR, run_name="cSAC")
    try:
        for t in range(1, total_steps + 1):
            # 动作
            obs_t = _to_torch_obs(obs, algo.device)
            action, _ = algo.policy.act(obs_t, deterministic=False)

            next_obs, reward, done, trunc, info = base_env.step(action[0])  # action shape (1,H) -> 取[0]
            costs = info.get("costs", {})
            # === NEW: 逐步日志（含 step 与 episode）===
            run_logger.log_step(step=t, episode=episode_idx, reward=float(reward), costs=costs, info=info)

            costs = info.get("costs", {})  # env 已在 step 中放进去
            # 在训练 loop 里：
            # if costs.get("illegal_caps", 0) > 0:
            # # print("[DEBUG][illegal]", costs.get("illegal_detail"))
            for rec in info.get("violation_log", []):
                if rec and float(rec.get("penalty", 0)) > 0:
                    print(
                        f"[step={rec.get('step')}] task={rec.get('task_id')} type={rec.get('type')} penalty={rec.get('penalty')}")

            algo.replay.add(obs, action[0], float(reward), next_obs, bool(done or trunc), costs)

            obs = next_obs
            ep_ret += float(reward)
            ep_len += 1

            if done or trunc:
                # 终局信息
                print(f"[cSAC] Ep {episode_idx} done: ret={ep_ret:.2f}, len={ep_len}, status={info.get('status')}, "
                      f"obj={0.5 * info.get('makespan', -1) + 0.5 * info.get('energy', -1):.2f},mk={info.get('makespan', -1):.2f}, en={info.get('energy', -1):.2f}, pen={info.get('penalty', -1):.2f}")
                # === NEW: 回合日志 ===
                run_logger.log_episode(
                    episode=episode_idx,
                    ep_return=ep_ret,
                    ep_len=ep_len,
                    final_info=info
                )
                episode_idx += 1
                obs, _ = base_env.reset(seed=int(rng.integers(0, 10_000)))
                ep_ret, ep_len = 0.0, 0

            # 更新
            if len(algo.replay) > cfg.warmup_steps:
                for _ in range(cfg.updates_per_step):
                    batch = algo.replay.sample(cfg.batch_size)
                    log_info = algo.update(batch)
                    if t % 2000 == 0:
                        print(f"[cSAC] t={t} | "
                              f"Q={log_info['critic_loss']:.3f} | Pi={log_info['policy_loss']:.3f} | "
                              f"alpha={log_info['alpha']:.3f} | "
                              f"λ_lat={log_info.get('lambda_latency', 0):.3f}({log_info.get('avg_latency', 0):.3f}) | "
                              f"λ_bw={log_info.get('lambda_bandwidth', 0):.3f}({log_info.get('avg_bandwidth', 0):.3f})")
    finally:
        # === NEW: 训练结束确保落盘 ===
        run_logger.close()
        paths = run_logger.paths
        print(f"[cSAC] Training logs saved:\n  - steps:    {paths['steps']}\n  - episodes: {paths['episodes']}")
    return algo


def evaluate_tripart_csac(algo,
                          episodes: int = 10,
                          seed: int = 123,
                          deterministic: bool = True):
    """
    使用 cSAC 的策略进行纯评估（不学习）。
    - algo: ConstrainedDiscreteSAC 实例（内含 env 与 policy）
    - episodes: 评估回合数
    - deterministic: 是否用贪心动作（True=argmax）
    """
    # 兼容 DummyVecEnv(Monitor(...)) 或原生 env
    env = getattr(algo, "env", None)
    if env is None:
        raise ValueError("algo.env is None. Please pass the cSAC algo created by train_tripart_csac.")
    base_env = env.envs[0] if hasattr(env, "envs") else env

    rng = np.random.default_rng(seed)

    ep_summaries = []
    cost_keys = ["latency", "bandwidth", "comm_energy", "exec_energy"]

    for ep in range(episodes):
        obs, _ = base_env.reset(seed=int(seed + ep))
        done, trunc = False, False
        ep_ret, ep_len = 0.0, 0

        # 累计约束成本（按回合求均值）
        cost_acc = {k: 0.0 for k in cost_keys}
        cost_cnt = 0

        final_info = {}

        while not (done or trunc):
            # 用 cSAC 策略出动作
            a, _ = algo.policy.act(_to_torch_obs(obs, algo.device), deterministic=deterministic)
            # act 返回 (1,H)，需要取第 0 个样本
            obs, r, done, trunc, info = base_env.step(a[0])

            ep_ret += float(r)
            ep_len += 1
            final_info = info

            # 记录 step 的 costs
            step_costs = info.get("costs", {})
            for k in cost_keys:
                cost_acc[k] += float(step_costs.get(k, 0.0))
            cost_cnt += 1

        # 回合级统计
        avg_costs = {k: (cost_acc[k] / max(1, cost_cnt)) for k in cost_keys}
        summary = {
            "episode": ep + 1,
            "reward": ep_ret,
            "length": ep_len,
            "status": final_info.get("status", "unknown"),
            "makespan": float(final_info.get("makespan", -1.0)),
            "energy": float(final_info.get("energy", -1.0)),
            "penalty": float(final_info.get("penalty", -1.0)),
            **{f"avg_{k}": v for k, v in avg_costs.items()}
        }
        ep_summaries.append(summary)

        print(
            f"[cSAC Eval] Ep {ep + 1:02d} | "
            f"Ret={summary['reward']:.2f} Len={summary['length']:3d} | "
            f"Status={summary['status']} | "
            f"mk={summary['makespan']:.2f} en={summary['energy']:.2f} pen={summary['penalty']:.2f} | "
            f"lat={summary['avg_latency']:.3f} bw={summary['avg_bandwidth']:.3f} "
            f"ce={summary['avg_comm_energy']:.3f} ee={summary['avg_exec_energy']:.3f}"
        )

    # 整体汇总
    import numpy as _np

    def _safe_avg(arr):
        return float(_np.mean(arr)) if len(arr) > 0 else float("nan")

    rets = [x["reward"] for x in ep_summaries]
    lens = [x["length"] for x in ep_summaries]
    succ = [x for x in ep_summaries if x["status"] == "success"]
    succ_rate = len(succ) / max(1, episodes)

    agg = {
        "episodes": episodes,
        "avg_reward": _safe_avg(rets),
        "avg_length": _safe_avg(lens),
        "success_rate": succ_rate,
        "success_avg_makespan": _safe_avg([x["makespan"] for x in succ]),
        "success_avg_energy": _safe_avg([x["energy"] for x in succ]),
        "success_avg_penalty": _safe_avg([x["penalty"] for x in succ]),
        "all_avg_latency": _safe_avg([x["avg_latency"] for x in ep_summaries]),
        "all_avg_bandwidth": _safe_avg([x["avg_bandwidth"] for x in ep_summaries]),
        "all_avg_comm_energy": _safe_avg([x["avg_comm_energy"] for x in ep_summaries]),
        "all_avg_exec_energy": _safe_avg([x["avg_exec_energy"] for x in ep_summaries]),
    }

    print("\n[cSAC Eval] ===== Summary =====")
    print(
        f"Episodes={agg['episodes']} | "
        f"AvgRet={agg['avg_reward']:.2f} | AvgLen={agg['avg_length']:.1f} | "
        f"SuccRate={agg['success_rate'] * 100:.1f}%"
    )
    print(
        f"Success Only -> "
        f"AvgMakespan={agg['success_avg_makespan']:.3f} | "
        f"AvgEnergy={agg['success_avg_energy']:.3f} | "
        f"AvgPenalty={agg['success_avg_penalty']:.3f}"
    )
    print(
        f"All Episodes Avg Costs -> "
        f"Latency={agg['all_avg_latency']:.3f} | "
        f"Bandwidth={agg['all_avg_bandwidth']:.3f} | "
        f"CommE={agg['all_avg_comm_energy']:.3f} | "
        f"ExecE={agg['all_avg_exec_energy']:.3f}"
    )
    return {"episodes": ep_summaries, "summary": agg}


# <<< NEW: Custom Callback for logging training data >>>
class TrainingLoggerCallback(BaseCallback):
    def __init__(self, log_path: str, level_name: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self.level_name = level_name
        self.episode_data = []
        self.start_time = time.time()

    def _on_step(self) -> bool:
        # 检查是否有环境完成了 Episode
        # 我们从日志中已经确认，这个条件是会满足的
        if self.locals["dones"][0]:
            info = self.locals["infos"][0]

            # <<< 核心修复：直接从 info 字典顶层获取数据 >>>
            # 我们不再查找 'episode' 或 'final_info'，因为日志显示它们不存在。

            # 尝试获取 SB3 的 'episode' 键，如果不存在则使用默认值
            # 注意：在你的日志中，这个键不存在，所以会使用默认值
            ep_info = info.get("episode")
            ep_reward = ep_info["r"] if ep_info else -999.0  # 使用一个明显的值表示数据缺失
            ep_length = ep_info["l"] if ep_info else -1

            # 直接从 info 字典获取我们的自定义性能指标
            makespan = info.get("makespan", -1.0)
            energy = info.get("energy", -1.0)
            # 注意：penalty 是 np.float64 类型，需要转换为 float 以便格式化
            penalty = float(info.get("penalty", -1.0))
            status = info.get("status", "unknown")

            # 记录数据
            self.episode_data.append({
                "timesteps": self.num_timesteps,
                "episode_reward": ep_reward,
                "episode_length": ep_length,
                "makespan": makespan,
                "energy": energy,
                "penalty": penalty,
                "status": status
            })

            # 计算并打印日志
            now = time.time()
            duration = now - self.start_time
            sps = self.num_timesteps / duration if duration > 0 else 0

            log_str = (
                f"[{self.level_name}] "
                f"TS: {self.num_timesteps:<8} | "
                f"Ep: {len(self.episode_data):<5} | "
                f"Reward: {ep_reward:<8.2f} | "  # Reward 会显示 -999.00
                f"Len: {ep_length:<4} | "
                f"Status: {status:<22} | "
                f"Makespan: {makespan:<7.2f} | "
                f"Energy: {energy:<8.2f} | "
                f"Penalty: {penalty:<6.2f} | "
                f"SPS: {int(sps):<5}"
            )
            print(log_str)

        return True

    def _on_training_end(self) -> None:
        if self.episode_data:
            df = pd.DataFrame(self.episode_data)
            df.to_csv(self.log_path, index=False)
            if self.verbose > 0:
                print(f"[{self.level_name}] Training log with {len(df)} episodes saved to {self.log_path}")


# === NEW: Simple run logger for manual cSAC training ===
import os, csv
from datetime import datetime


class TrainRunLogger:
    """
    记录两类日志：
    - steps.csv：每个环境步的信息（step, episode, reward, costs...）
    - episodes.csv：每个回合的聚合指标（episode, ret, len, makespan, energy, penalty, status...）
    """

    def __init__(self, log_dir: str = "logs", run_name: str | None = None):
        os.makedirs(log_dir, exist_ok=True)
        stamp = run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.step_path = os.path.join(log_dir, f"steps_{stamp}.csv")
        self.ep_path = os.path.join(log_dir, f"episodes_{stamp}.csv")

        # 写入 CSV 头
        self._step_writer = None
        self._ep_writer = None
        self._step_f = open(self.step_path, "w", newline="", encoding="utf-8")
        self._ep_f = open(self.ep_path, "w", newline="", encoding="utf-8")

    def log_step(self, *, step: int, episode: int, reward: float, costs: dict, info: dict):
        # 统一一些常见键，缺失则给默认值
        row = {
            "step": step,
            "episode": episode,
            "reward": float(reward),
            "latency": float(costs.get("latency", 0.0)),
            "bandwidth": float(costs.get("bandwidth", 0.0)),
            "comm_energy": float(costs.get("comm_energy", 0.0)),
            "exec_energy": float(costs.get("exec_energy", 0.0)),
            "illegal_caps": float(costs.get("illegal_caps", 0.0)),
            "cap_violation": float(costs.get("cap_violation", 0.0)),
            "resource_violation": float(costs.get("resource_violation", 0.0)),
            "rejected": bool(info.get("rejected", False)),
        }
        # 懒初始化 writer（根据首次行的列名创建）
        if self._step_writer is None:
            self._step_writer = csv.DictWriter(self._step_f, fieldnames=list(row.keys()))
            self._step_writer.writeheader()
        self._step_writer.writerow(row)

    def log_episode(self, *, episode: int, ep_return: float, ep_len: int, final_info: dict):
        row = {
            "episode": episode,
            "episode_return": float(ep_return),
            "episode_length": int(ep_len),
            "status": str(final_info.get("status", "unknown")),
            "makespan": float(final_info.get("makespan", -1.0)),
            "energy": float(final_info.get("energy", -1.0)),
            "penalty": float(final_info.get("penalty", -1.0)),
            "max_latency_cost": float(final_info.get("max_latency_cost", 0.0)),
            "max_bandwidth_cost": float(final_info.get("max_bandwidth_cost", 0.0)),
            "feasible": bool(final_info.get("feasible", False)),
        }
        if self._ep_writer is None:
            self._ep_writer = csv.DictWriter(self._ep_f, fieldnames=list(row.keys()))
            self._ep_writer.writeheader()
        self._ep_writer.writerow(row)

    def close(self):
        try:
            self._step_f.flush();
            self._ep_f.flush()
        finally:
            self._step_f.close();
            self._ep_f.close()

    @property
    def paths(self):
        return {"steps": self.step_path, "episodes": self.ep_path}


# <<< NEW: 函数用于生成详细的部署方案报告 >>>
def generate_deployment_report(
        placement: Dict[Tuple[int, int], Dict],
        task_graph: TaskGraph,
        agent_lookup: Dict[int, AgentTemplate],
        device_map: Dict[int, Device],
        final_info: Dict
) -> Dict:
    """
    将最终的部署方案（placement）转换成一个详细、易读的字典/JSON报告。

    Args:
        placement: 最终的部署方案字典。
        task_graph: 对应的任务图。
        agent_lookup: 智能体模板查找表。
        device_map: 设备实例查找表。
        final_info: 包含最终性能指标的字典。

    Returns:
        一个包含完整部署细节的字典。
    """
    task_id = task_graph.task_id if hasattr(task_graph, 'task_id') else list(placement.keys())[0][0]

    report = {
        "task_id": task_id,
        "overall_performance": final_info,
        "deployment_plan": []
    }

    # --- 预处理：建立一个快速查找表，用于查找哪个智能体提供哪个传感器能力 ---
    sensor_provider_map = {}
    for (tid, mod_id), mod_info in placement.items():
        agent = agent_lookup[mod_info["agent_id"]]
        for sensor_cap in agent.C_sense:
            sensor_provider_map[sensor_cap] = {
                "agent_id": agent.id,
                "module_id": mod_id
            }

    # --- 遍历每个模块，生成其详细报告 ---
    sorted_modules = sorted(placement.items(), key=lambda item: item[0][1])  # 按 module_id 排序

    for (tid, module_id), info in sorted_modules:
        agent = agent_lookup[info["agent_id"]]

        # 1. 获取模块所需的能力
        required_sense, _, _ = get_module_capabilities(task_graph.G, info["nodes"])

        # 2. 识别缺失和需要协作的传感器
        missing_sensors = required_sense - agent.C_sense
        collaboration_details = []
        for missing_cap in missing_sensors:
            if missing_cap in sensor_provider_map:
                provider = sensor_provider_map[missing_cap]
                collaboration_details.append({
                    "capability_id": missing_cap,
                    "status": "Collaborative",
                    "provided_by_agent_id": provider["agent_id"],
                    "in_module_id": provider["module_id"]
                })
            else:
                collaboration_details.append({
                    "capability_id": missing_cap,
                    "status": "Missing (Unsolved)",
                    "provided_by_agent_id": None,
                    "in_module_id": None
                })

        # 3. 整理设备部署信息
        soft_dev_id = info["soft_device"]
        device_deployments = {
            "software_deployment": {
                "device_id": soft_dev_id,
                "device_type": device_map[soft_dev_id].type
            },
            "sensor_deployments": [
                {
                    "capability_id": cap_id,
                    "device_id": dev_id,
                    "device_type": device_map[dev_id].type
                } for cap_id, dev_id in info["sense_map"].items()
            ],
            "actuator_deployments": [
                {
                    "capability_id": cap_id,
                    "device_id": dev_id,
                    "device_type": device_map[dev_id].type
                } for cap_id, dev_id in info["act_map"].items()
            ]
        }

        # 4. 组装该模块的报告
        module_report = {
            "module_id": module_id,
            "nodes": sorted(list(info["nodes"])),  # 排序以保证输出一致性
            "assigned_agent_template_id": agent.id,
            "deployment_devices": device_deployments,
            "collaboration_details": {
                "required_sensors": sorted(list(required_sense)),
                "provided_by_this_agent": sorted(list(agent.C_sense)),
                "collaborative_sensors_needed": collaboration_details
            }
        }
        report["deployment_plan"].append(module_report)

    return report
