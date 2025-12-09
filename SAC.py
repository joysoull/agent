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
from sub_partiation import AgentTemplate, parse_capability_field

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

# --- GRASP + LNS (joint partition & match) ---
from grasp_lns import (
    solve_stage1_partition_and_match,
    GRASP_DEFAULT_WEIGHTS, Weights, GRASPParams,
)


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


class TriPartPlacementEnv(gym.Env):
    """
    Tri-Part 放置环境：
    - 动作：MultiDiscrete([num_devices] * (1 + K_s_max + K_a_max))
        [0]         : 计算部分 -> 设备
        [1..Ks]     : 每个“必需传感能力” -> 设备
        [Ks+1..end] : 每个“必需驱动能力” -> 设备
    - 观测：DAG特征、设备剩余资源、一些约束统计 + 三类 mask
      (compute_mask, sense_mask[K_s_max, D], act_mask[K_a_max, D])
    - 奖励（step）：
        进度奖励 R_PROGRESS
        - 延迟/带宽违约（模块间、传感→算力、算力→驱动）
        - 通信/计算能耗
      （终评时用 makespan/总能耗/违约进行一次性加扣）
    """
    metadata = {"render_modes": []}

    def __init__(self, task_graphs, agent_lookup: Dict[int, AgentTemplate], device_map, gw_matrix,
                 K_s_max=3, K_a_max=3, seed=42,
                 task_num = 10,
                 # NEW ↓↓↓
                 max_module_size: int = 5,
                 min_module_size: int = 1,
                 lns_rounds: int = 300,
                 grasp_runs: int = 40,
                 grasp_rcl_k: int = 3,
                 grasp_weights: Dict[str, float] = GRASP_DEFAULT_WEIGHTS,
                 # NEW: 预计算计划
                 precomputed_plans: Optional[Dict[int, Dict[str, Any]]] = None,
                 precomputed_plans_path: Optional[str] = None,
                 strict_fingerprint_check: bool = True,
                 # >>> 新增：用于可行性判定的阈值（与 cSAC 的 cost_limits 对齐）
                 constraint_limits: Optional[Dict[str, float]] = None,
                 sense_cap_quota_per_device: int = 2,  # 每台设备同一传感能力最多协作次数L（默认为2）
                 act_cap_quota_per_device: int = 1,  # 执行器默认独占：同一执行能力每台设备只能绑定1次
                 # 在 __init__ 参数列表里加：
                 prune_by_latency: bool = False,
                 terminal_penalty: bool = True,
                 # >>> NEW: reward config
                 reward_mode: str = "penalty",  # "dense" | "paper"
                 paper_mid_alpha: float = 0.5,  # r_step = -(alpha*T + (1-alpha)*E)
                 paper_final_bonus_per_app: float = 50.0,  # +/- 50 * N (N = T = #apps)
                 early_terminate_on_violation: bool = False,  # 发生硬违规是否立即 -50N 终止
                 ):
        super().__init__()
        self.reward_mode = str(reward_mode).lower()
        self.paper_mid_alpha = float(paper_mid_alpha)
        self.paper_final_bonus_per_app = float(paper_final_bonus_per_app)
        self.early_terminate_on_violation = bool(early_terminate_on_violation)

        # __init__ 体内：标准化权重
        def _to_weights(wlike):
            if isinstance(wlike, Weights):
                return wlike
            if isinstance(wlike, dict):
                return Weights(
                    w_module=wlike.get("w_module", 0.20),
                    w_interface=wlike.get("w_interface", 1.00),
                    w_redund=wlike.get("w_redund", 0.30),
                    w_collab=wlike.get("w_collab", 0.60),
                    w_balance=wlike.get("w_balance", 0.15),
                )
            return Weights()

        self.grasp_weights = _to_weights(grasp_weights)

        # 保存新开关
        self.prune_by_latency = bool(prune_by_latency)
        self.terminal_penalty = bool(terminal_penalty)
        self.ep_cost_max = None
        self.ep_flags = None
        self.rng = np.random.default_rng(seed)


        loaded_tgs: list[TaskGraph] = []
        if task_graphs:  # 模式 A
            # 如果 task_count 指定了，就优先取 id 在 [1..task_count] 的 tg；否则全用
            if isinstance(task_num, int) and task_num > 0:
                # 先按 id 排个序，确保 1..task_count 顺序
                sorted_tg = sorted(task_graphs, key=lambda tg: int(getattr(tg, "id", 10 ** 9)))
                keep_ids = set(range(1, task_num + 1))
                loaded_tgs = [tg for tg in sorted_tg if int(getattr(tg, "id", -1)) in keep_ids]
                # 若传入的 tg 没有 id 信息或数量不足，就退化为简单切片
                if len(loaded_tgs) == 0:
                    loaded_tgs = sorted_tg[:task_num]
            else:
                loaded_tgs = list(task_graphs)
        self.task_graphs = loaded_tgs
        self.T = len(self.task_graphs)  # 任务数
        self.agent_lookup = agent_lookup
        self.device_map = device_map
        self.gw_matrix = gw_matrix

        # NEW: 保存 GRASP/LNS 超参
        self.max_module_size = max_module_size
        self.min_module_size = min_module_size
        self.lns_rounds = lns_rounds
        self.grasp_runs = grasp_runs
        self.grasp_rcl_k = grasp_rcl_k

        self.device_ids = sorted(device_map.keys())
        self.num_devices = len(self.device_ids)
        self.dev_id2idx = {d: i for i, d in enumerate(self.device_ids)}
        self.idx2dev = {i: d for d, i in self.dev_id2idx.items()}

        self.max_nodes = max(len(tg.G.nodes) for tg in self.task_graphs)
        self.K_s_max = K_s_max
        self.K_a_max = K_a_max
        self.num_heads = 1 + self.K_s_max + self.K_a_max
        self._soft_ema = 1.0  # 任何正数起步都行，避免除零；会很快自适应
        self.precomputed_plans = precomputed_plans
        self.precomputed_plans_path = precomputed_plans_path
        self.strict_fingerprint_check = strict_fingerprint_check
        if self.precomputed_plans is None and self.precomputed_plans_path is not None:
            # 延迟加载
            self.precomputed_plans = load_partition_plans(self.precomputed_plans_path)

        self.modules_queue_by_task: Dict[int, List[Dict[str, Any]]] = {}
        self.current_modules_by_task: Dict[int, List[Dict[str, Any]]] = {}
        self.assigned_nodes_by_task: Dict[int, set] = {}

        self.curr_req_sense_by_task: Dict[int, List[int]] = {}
        self.curr_req_act_by_task: Dict[int, List[int]] = {}
        self.compute_mask_by_task: Dict[int, np.ndarray] = {}
        self.sense_mask_by_task: Dict[int, np.ndarray] = {}
        self.act_mask_by_task: Dict[int, np.ndarray] = {}
        self.LAT_TOL = 0.2  # 允许 5% 规范化超额仅作为 shaping，不当硬违规
        self.BW_TOL = 0.2
        self.constraint_limits = constraint_limits or {"latency": self.LAT_TOL, "bandwidth": self.BW_TOL}
        # 观测
        self.observation_space = spaces.Dict({
            "device_resources": spaces.Box(0, 1, shape=(self.num_devices, 6), dtype=np.float32),
            "constraints": spaces.Box(0, 1, shape=(8,), dtype=np.float32),

            # ✅ 多任务形状，与 _obs() 完全一致
            "task_mask": spaces.Box(0, 1, shape=(self.T,), dtype=np.float32),
            "compute_mask": spaces.Box(0, 1, shape=(self.T, self.num_devices), dtype=np.float32),
            "sense_mask": spaces.Box(0, 1, shape=(self.T, self.K_s_max, self.num_devices), dtype=np.float32),
            "act_mask": spaces.Box(0, 1, shape=(self.T, self.K_a_max, self.num_devices), dtype=np.float32),
        })

        # 运行期
        self.current_task = None
        self.modules_queue: List[Dict[str, Any]] = []
        self.current_modules: List[Dict[str, Any]] = []
        self.assigned_nodes = set()
        self.step_count = 0
        self.resource_used = defaultdict(lambda: np.zeros(6, dtype=np.float32))
        self.curr_req_sense: List[int] = []
        self.curr_req_act: List[int] = []

        self.iot_indices = np.array(
            [i for i, did in enumerate(self.device_ids)
             if str(getattr(self.device_map[did], "type", "")).lower() == "device"],
            dtype=int
        )
        self.num_iot = int(len(self.iot_indices))
        self._iot_index_set = set(self.iot_indices.tolist())

        self._compute_mask = np.ones(self.num_devices, dtype=np.float32)
        self._sense_mask = np.ones((self.K_s_max, self.num_devices), dtype=np.float32)
        self._act_mask = np.ones((self.K_a_max, self.num_devices), dtype=np.float32)

        self._init_energy_denominators()

        self.sense_cap_quota_per_device = int(sense_cap_quota_per_device)
        self.act_cap_quota_per_device = int(act_cap_quota_per_device)

        # 运行期占用计数：device_id -> {cap_id -> used_count}
        self.sense_cap_usage = defaultdict(lambda: defaultdict(int))
        self.act_cap_usage = defaultdict(lambda: defaultdict(int))

        # 动作：MultiDiscrete（更干净）
        # ==== 动作空间：第 0 维为任务选择头 ====
        # __init__ 里
        self.action_space = spaces.MultiDiscrete(
            [self.num_devices] + [self.num_iot] * self.K_s_max + [self.num_iot] * self.K_a_max
        )

        self.violation_log = []  # 🚀 逐回合违规记录缓冲
        # Episode 级能耗累计（焦耳）
        self._episode_comm_J = 0.0
        self._episode_exec_J = 0.0
        self._progress_total_modules = sum(len(self.modules_queue_by_task.get(int(getattr(tg, "id", -1)), []))
                                           for tg in self.task_graphs)
        self._progress_done = 0
        self.STEP_TIME_COST = 0.01  # 微小逐步成本，抑制拖步
        self._progress_total_modules = 0  # 总模块数（每次 reset 重算）
        self._progress_done = 0  # 已成功接纳的模块数
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.action_space.seed(seed)
        # === 调试开关 ===
        self.debug_level = 1  # 0=静默, 1=关键信息, 2=详细
        self._last_obs = None
        # === 调度（环境选择当前要部署的任务）===
        self.schedule_policy = "round_robin"  # 可选: "round_robin" | "random"
        self.task_ptr = -1
        self.active_tid = None  # 当前激活的任务id
        self.ep_cost_max = {"latency": 0.0, "bandwidth": 0.0}
        self.ep_cost_sum = {"latency": 0.0, "bandwidth": 0.0}  # 可选：累计
        self.ep_cost_cnt = 0  # 可选：步数

    def _pick_next_task(self):
        """按策略选下一个有就绪模块的任务，更新 self.active_tid；若都空则置 None。"""
        tids = [int(getattr(tg, "id", -1)) for tg in self.task_graphs]
        if len(tids) == 0:
            self.active_tid = None
            return None
        if self.schedule_policy == "random":
            self.rng.shuffle(tids)
            for cand in tids:
                if len(self.modules_queue_by_task.get(cand, [])) > 0:
                    self.active_tid = cand
                    return cand
            self.active_tid = None
            return None
        # 默认：round-robin
        for _ in range(len(tids)):
            self.task_ptr = (self.task_ptr + 1) % len(tids)
            cand = tids[self.task_ptr]
            if len(self.modules_queue_by_task.get(cand, [])) > 0:
                self.active_tid = cand
                return cand
        self.active_tid = None
        return None

    # === 动作与掩码一致性检查（只做调试，不阻塞） ===
    def _check_action_vs_mask(self, action: np.ndarray):
        try:
            nvec = self.action_space.nvec
            mask = self._flat_action_mask()
            offsets = np.cumsum([0] + list(nvec[:-1]))
            bad = []
            for h, val in enumerate(action):
                idx = int(offsets[h] + val)
                if idx >= mask.size or not mask[idx]:
                    bad.append((h, int(val)))
            if bad and self.debug_level >= 1:
                print(f"[WARN] Action not allowed by mask at heads: {bad}")
            return bad
        except Exception as e:
            if self.debug_level >= 1:
                print(f"[WARN] _check_action_vs_mask failed: {e}")
            return [("exception", -1)]

    # === 掩码统计（覆盖率 & 异常行） ===
    def _mask_stats(self, obs: dict | None = None) -> dict:
        m = self._flat_action_mask()
        nvec = self.action_space.nvec
        offsets = np.cumsum([0] + list(nvec[:-1]))

        def seg(h): return m[offsets[h]: offsets[h] + nvec[h]]

        stats = {}
        # head 0: compute
        comp_seg = seg(0)
        stats["compute_true"] = int(comp_seg.sum())
        stats["compute_total"] = int(comp_seg.size)

        # heads 1..Ks: sense
        base = 1
        stats["sense_true_per_row"] = [int(seg(base + k).sum()) for k in range(self.K_s_max)]
        stats["sense_total_per_row"] = [int(nvec[base + k]) for k in range(self.K_s_max)]

        # heads Ks+1..end: act
        base2 = 1 + self.K_s_max
        stats["act_true_per_row"] = [int(seg(base2 + k).sum()) for k in range(self.K_a_max)]
        stats["act_total_per_row"] = [int(nvec[base2 + k]) for k in range(self.K_a_max)]

        stats["flat_true"] = int(m.sum())
        stats["flat_total"] = int(m.size)
        stats["flat_ratio"] = float(m.mean()) if m.size > 0 else 0.0
        return stats

    # 在类里新增
    def action_masks(self) -> np.ndarray:
        # 用当前缓存的观测，确保与策略输入一致
        if self._last_obs is None:
            # 训练刚开始时，ActionMasker 会在 env.reset() 后立刻调用
            self._last_obs, _ = self.reset()
        return self._flat_action_mask(self._last_obs)

    # === 掩码形状一致性检查（在 reset()/step() 里偶尔调用） ===
    def _check_masks_shapes(self, obs: dict):
        assert obs["task_mask"].shape == (self.T,), f"task_mask shape {obs['task_mask'].shape} != (T,)"
        assert obs["compute_mask"].shape == (self.T, self.num_devices), "compute_mask shape mismatch"
        assert obs["sense_mask"].shape == (self.T, self.K_s_max, self.num_devices), "sense_mask shape mismatch"
        assert obs["act_mask"].shape == (self.T, self.K_a_max, self.num_devices), "act_mask shape mismatch"
        # 额外：IoT 索引有效
        if self.num_iot > 0:
            assert np.all((self.iot_indices >= 0) & (self.iot_indices < self.num_devices)), "iot_indices out of range"

    # === 统一构造扁平掩码（与 MultiDiscrete.nvec 完全一致）===
    def _flat_action_mask(self, obs: dict | None = None) -> np.ndarray:
        # 忽略 obs 参数，直接用当前 active 任务的掩码
        tid = self.active_tid
        # 若所有任务都完成/无可部署，返回全 True 以免采样崩溃（下一步就会终局）
        if tid is None:
            return np.ones(int(self.action_space.nvec.sum()), dtype=bool)

        cm = self.compute_mask_by_task.get(tid, np.ones(self.num_devices, dtype=np.float32))
        sm = self.sense_mask_by_task.get(tid, np.ones((self.K_s_max, self.num_devices), dtype=np.float32))
        am = self.act_mask_by_task.get(tid, np.ones((self.K_a_max, self.num_devices), dtype=np.float32))

        parts = [cm > 0.5]
        if self.num_iot > 0:
            iot = self.iot_indices
            parts += [(sm[k, iot] > 0.5) for k in range(self.K_s_max)]
            parts += [(am[k, iot] > 0.5) for k in range(self.K_a_max)]
        mask = np.concatenate(parts, axis=0).astype(bool)

        # —— 保险：每个 head 至少一个 True（MaskablePPO 需要）
        seg_lens = [self.num_devices] + [self.num_iot] * self.K_s_max + [self.num_iot] * self.K_a_max
        off = 0
        for L in seg_lens:
            seg = mask[off:off + L]
            if L > 0 and seg.sum() == 0:
                seg[np.argmax(np.arange(L))] = True
            off += L
        expected = int(self.action_space.nvec.sum())
        assert mask.size == expected, f"mask.size={mask.size} != sum(nvec)={expected}"
        return mask

    def _load_or_build_modules_queue_for_task(self, tg: TaskGraph):
        """给单个任务构建/加载 modules_queue。"""
        task_id = getattr(tg, "id", None)
        if task_id is None:
            # 回退 GRASP/LNS
            plan = solve_stage1_partition_and_match(
                task_graph=tg, agent_lookup=self.agent_lookup, max_module_size=self.max_module_size,
                weights=(self.grasp_weights if isinstance(self.grasp_weights, Weights) else GRASP_DEFAULT_WEIGHTS)
            )
            q = [{"module_id": int(m["module_id"]), "nodes": set(m["nodes"]), "agent_id": int(m["agent_id"])} for m in
                 plan["modules"]]
            self.modules_queue_by_task[task_id] = q
            return

        # 优先读取预计算
        if self.precomputed_plans is None and self.precomputed_plans_path is not None:
            self.precomputed_plans = load_partition_plans(self.precomputed_plans_path)

        if self.precomputed_plans:
            plan = self.precomputed_plans.get(int(task_id))
            self.modules_queue_by_task[task_id] = realize_plan_to_modules_queue(plan)
            return
            # if plan is not None:
            #     if self.strict_fingerprint_check:
            #         fp_now = _dag_fingerprint(tg.G)
            #         if plan.get("fingerprint") and plan["fingerprint"] != fp_now:
            #             # 重新构造
            #             pass
            #         else:
            #             self.modules_queue_by_task[task_id] = realize_plan_to_modules_queue(plan)
            #             return

        # 构造
        plan = solve_stage1_partition_and_match(
            task_graph=tg, agent_lookup=self.agent_lookup, max_module_size=self.max_module_size,
            weights=(self.grasp_weights if isinstance(self.grasp_weights, Weights) else GRASP_DEFAULT_WEIGHTS)
        )
        self.modules_queue_by_task[task_id] = [
            {"module_id": int(m["module_id"]), "nodes": set(m["nodes"]), "agent_id": int(m["agent_id"])}
            for m in plan["modules"]
        ]

    def _init_energy_denominators(self):
        # 与你现有 ACED 环境相同
        try:
            max_data_mb = max(
                (attr.get("data_size", 0.0) for tg in self.task_graphs for _, _, attr in tg.get_dependencies()),
                default=1.0
            )
            max_data_mb = max(1.0, max_data_mb)
        except Exception:
            max_data_mb = 1.0
        self.comm_energy_norm = max(1e-6, COMM_ENERGY_PER_BIT_WIRED * max_data_mb * 8e6)

        try:
            max_cpu_need = max(float(tpl.r[0]) for tpl in self.agent_lookup.values())
            max_gpu_need = max(float(tpl.r[2]) for tpl in self.agent_lookup.values())
            min_cpu_cap, min_gpu_cap = float('inf'), float('inf')
            max_cpu_power, max_gpu_power = 0.0, 0.0
            for dev in self.device_map.values():
                res = dev.resource
                if getattr(res, "cpu", 0) > 0:
                    min_cpu_cap = min(min_cpu_cap, float(getattr(res, "cpu", 0)))
                    max_cpu_power = max(max_cpu_power, float(getattr(res, "cpu_power", 1.0)))
                if getattr(res, "gpu", 0) > 0:
                    min_gpu_cap = min(min_gpu_cap, float(getattr(res, "gpu", 0)))
                    max_gpu_power = max(max_gpu_power, float(getattr(res, "gpu_power", 1.0)))
            cpu_term = (max_cpu_need / max(1.0, min_cpu_cap)) * (max_cpu_power / 3600.0)
            gpu_term = (max_gpu_need / max(1.0, min_gpu_cap)) * (max_gpu_power / 3600.0)
            self.exec_energy_norm = max(1e-6, cpu_term + gpu_term)
        except Exception:
            self.exec_energy_norm = 1.0

    def _prune_compute_mask_by_latency(self, tid: int, tg: TaskGraph, cm: np.ndarray):
        """
        用已部署节点的设备，预估当前模块若选择某个 compute_dev，
        是否会导致 与已部署相邻节点 的跨模块边出现明显延迟超额。
        若某个 compute_dev 对任意相关边的规范化延迟 > (LAT_TOL)，则 cm[i]=0.
        """
        if cm is None or not np.any(cm > 0.5):
            return cm

        # 收集当前任务里“已部署”的节点 -> 设备
        placed_dev_of = {}
        for mod in self.current_modules_by_task.get(tid, []):
            for n in mod["nodes"]:
                # 查这个节点在该已部署模块上对应的设备
                G = tg.G
                ntype = G.nodes[n].get("type", "proc")
                idx = G.nodes[n].get("idx")
                if ntype == "proc":
                    placed_dev_of[n] = mod["soft_device"]
                elif ntype == "sense":
                    placed_dev_of[n] = mod["sense_map"].get(idx)
                elif ntype == "act":
                    placed_dev_of[n] = mod["act_map"].get(idx)

        if not placed_dev_of:
            return cm  # 没有相邻已部署节点，不修剪

        # 找出“当前模块”的节点集合（来自队首模块）
        q = self.modules_queue_by_task.get(tid, [])
        if not q:
            return cm
        cur_nodes = q[0]["nodes"]
        G = tg.G

        # 逐个候选 compute_dev 试算它与已部署邻居的链路延迟是否明显超额
        for i, did in enumerate(self.device_ids):
            if cm[i] < 0.5:
                continue
            # 只评估当前模块中“proc 节点”作为 compute_dev 的承载者
            # 与已部署邻居的边：cur_nodes ↔ placed_nodes
            risky = False
            for u, v, attr in tg.get_dependencies():
                u_in = u in cur_nodes
                v_in = v in cur_nodes
                if u_in == v_in:
                    continue  # 只看跨模块
                other = v if u_in else u
                if other not in placed_dev_of:
                    continue

                # 设备对 (compute_dev, placed_dev)
                if u_in and G.nodes[u].get("type", "proc") == "proc":
                    du, dv = did, placed_dev_of[other]
                elif v_in and G.nodes[v].get("type", "proc") == "proc":
                    du, dv = placed_dev_of[other], did
                else:
                    # 与非 proc（比如当前模块的 sense/act）在这个阶段很难准确预估，跳过
                    continue

                if du is None or dv is None or du == dv:
                    continue

                # 设定合理默认
                u_type = G.nodes[u].get("type", "proc")
                v_type = G.nodes[v].get("type", "proc")
                data_mb = attr.get("data_size")
                bw_req = attr.get("bandwidth_req")
                lat_req = attr.get("latency_req")
                if data_mb is None:
                    data_mb = 0.5 if (u_type == "sense" and v_type == "proc") else (
                        0.1 if (u_type == "proc" and v_type == "act") else 1.0)
                if bw_req is None:
                    bw_req = 0.0
                if lat_req is None:
                    lat_req = 20.0 if ((u_type == "sense" and v_type == "proc") or (
                            u_type == "proc" and v_type == "act")) else 50.0

                dms, bw_act, _ = compute_transmission_delay(
                    src=self.device_map[du], dst=self.device_map[dv],
                    data_size_mb=data_mb, bw_req=bw_req, gw_matrix=self.gw_matrix
                )
                lat_norm = norm_latency(dms, lat_req)
                if lat_norm > self.LAT_TOL:
                    risky = True
                    break

            if risky:
                cm[i] = 0.0

        # 若全被剔除，保留一个最不坏的候选兜底，避免动作空间塌陷
        if np.sum(cm > 0.5) == 0:
            feas = []
            for i, did in enumerate(self.device_ids):
                # 计算一个“最不坏”的 lat_norm 作为排序依据
                worst = 0.0
                for u, v, attr in tg.get_dependencies():
                    u_in = u in cur_nodes
                    v_in = v in cur_nodes
                    if u_in == v_in:
                        continue
                    other = v if u_in else u
                    if other not in placed_dev_of:
                        continue
                    # 只试 proc 侧
                    if (u_in and G.nodes[u].get("type", "proc") != "proc") or \
                            (v_in and G.nodes[v].get("type", "proc") != "proc"):
                        continue
                    du = did
                    dv = placed_dev_of[other]
                    if du is None or dv is None or du == dv:
                        continue
                    dms, _, _ = compute_transmission_delay(
                        src=self.device_map[du], dst=self.device_map[dv],
                        data_size_mb=attr.get("data_size", 1.0),
                        bw_req=attr.get("bandwidth_req", 0.0),
                        gw_matrix=self.gw_matrix
                    )
                    lat_req = attr.get("latency_req", 50.0)
                    worst = max(worst, norm_latency(dms, lat_req))
                feas.append((worst, i))
            if feas:
                feas.sort(key=lambda x: x[0])
                cm[feas[0][1]] = 1.0  # 选一个最不坏的兜底
        return cm

    # 🚀 新增：统一的违规记录器
    def _log_violation(self, tid: int, kind: str, payload: Dict[str, Any]):
        rec = {
            "step": int(self.step_count),
            "task_id": int(tid),
            "type": kind,
            **payload
        }
        self.violation_log.append(rec)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):

        self.illegal_steps = 0  # 本 EP 硬违规步计数（非法动作/能力/资源任一触发）
        self.rejected_steps = 0  # 本 EP 被拒绝计数（可选）

        self._soft_ema = 1.0  # 任何正数起步都行，避免除零；会很快自适应
        self.violation_log = []
        self.ep_flags = {"illegal": 0, "cap": 0, "resource": 0}
        self.ep_cost_max = {"latency": 0.0, "bandwidth": 0.0}
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.resource_used = defaultdict(lambda: np.zeros(6, dtype=np.float32))
        self.sense_cap_usage.clear()
        self.act_cap_usage.clear()

        # 为每个 task 构建/加载队列、清空状态并准备掩码
        self.modules_queue_by_task = {}
        self.current_modules_by_task = {}
        self.assigned_nodes_by_task = {}
        self.curr_req_sense_by_task = {}
        self.curr_req_act_by_task = {}
        self.compute_mask_by_task = {}
        self.sense_mask_by_task = {}
        self.act_mask_by_task = {}

        for tg in self.task_graphs:
            tid = int(getattr(tg, "id", -1))
            self._load_or_build_modules_queue_for_task(tg)
            self.current_modules_by_task[tid] = []
            self.assigned_nodes_by_task[tid] = set()
            self._prepare_masks_for_task(tg)
        # 统计本回合总模块数（只算队列里的计划模块）
        self._progress_total_modules = sum(len(self.modules_queue_by_task.get(int(getattr(tg, "id", -1)), []))
                                           for tg in self.task_graphs)
        self._progress_done = 0
        # 选择首个可部署任务
        self.task_ptr = -1
        self._pick_next_task()
        obs = self._obs()
        self._last_obs = obs  # 缓存
        try:
            self._check_masks_shapes(obs)
            if self.debug_level >= 2:
                print("[DBG reset] mask stats:", self._mask_stats(obs))
        except AssertionError as e:
            print("[ERR reset]", e)
        self.ep_cost_max["latency"] = 0.0
        self.ep_cost_max["bandwidth"] = 0.0
        self.ep_cost_sum["latency"] = 0.0
        self.ep_cost_sum["bandwidth"] = 0.0
        return obs, {}

    def _prepare_masks_for_task(self, tg: TaskGraph):
        tid = int(getattr(tg, "id", -1))
        q = self.modules_queue_by_task.get(tid, [])
        if not q:
            self.curr_req_sense_by_task[tid] = []
            self.curr_req_act_by_task[tid] = []
            self.compute_mask_by_task[tid] = np.ones(self.num_devices, dtype=np.float32)
            self.sense_mask_by_task[tid] = np.ones((self.K_s_max, self.num_devices), dtype=np.float32)
            self.act_mask_by_task[tid] = np.ones((self.K_a_max, self.num_devices), dtype=np.float32)
            return

        mod = q[0]
        agent = self.agent_lookup[mod["agent_id"]]
        req_s, req_a, req_soft = get_agent_capabilities(agent)
        req_s_list = sorted(list(req_s))[:self.K_s_max]
        req_a_list = sorted(list(req_a))[:self.K_a_max]

        self.curr_req_sense_by_task[tid] = req_s_list
        self.curr_req_act_by_task[tid] = req_a_list

        # ✅ 修正：第二个参数应为 agent_id
        self.compute_mask_by_task[tid] = self._build_compute_mask(req_soft, agent_id=mod["agent_id"])
        if self.prune_by_latency:
            self.compute_mask_by_task[tid] = self._prune_compute_mask_by_latency(tid, tg,
                                                                                 self.compute_mask_by_task[tid])

        self.sense_mask_by_task[tid] = self._build_cap_mask(req_s_list, "sense")
        self.act_mask_by_task[tid] = self._build_cap_mask(req_a_list, "act")

    def _build_compute_mask(self, req_soft: set, agent_id: int) -> np.ndarray:
        mask = np.ones(self.num_devices, dtype=np.float32)
        needed_r6 = np.array(self.agent_lookup[agent_id].r, dtype=np.float32)
        need4 = needed_r6[[1, 3, 4, 5]]
        for i, did in enumerate(self.device_ids):
            dev = self.device_map[did]
            # AFTER:
            if len(req_soft - dev.soft_cap) > 0:
                mask[i] = 0.0
                continue
            # 资源
            # 四维资源检查（与 step 保持一致）
            used4 = self.resource_used[did][[1, 3, 4, 5]]
            cap4 = self._cap4(dev)
            if np.any(used4 + need4 > cap4 + 1e-6):
                mask[i] = 0.0

        return mask

    def _build_cap_mask(self, cap_list: List[int], cap_type: str) -> np.ndarray:
        K = self.K_s_max if cap_type == "sense" else self.K_a_max
        mask = np.ones((K, self.num_devices), dtype=np.float32)  # 默认全 1（未用位）
        for k in range(K):
            if k >= len(cap_list):
                continue  # 未使用位保持全 1
            cap = cap_list[k]
            # 先置 0，再填可行设备为 1
            mask[k, :] = 0.0

            for i, did in enumerate(self.device_ids):
                # ✅ sense/act 只能去 IoT 行
                if cap_type in ("sense", "act") and (i not in self._iot_index_set):
                    continue

                dev = self.device_map[did]
                has = cap in (dev.sense_cap if cap_type == "sense" else dev.act_cap)
                if not has:
                    continue

                usage = (self.sense_cap_usage if cap_type == "sense" else self.act_cap_usage)[did][cap]
                quota = (self.sense_cap_quota_per_device if cap_type == "sense" else self.act_cap_quota_per_device)
                if usage < quota:
                    mask[k, i] = 1.0
                    # 调试：这一行完全 0，说明这个能力在 IoT 上没人能接 or 配额耗尽
            if mask[k].max() < 0.5:
                # 若 IoT 上“有此cap的设备”存在，只是配额耗尽，则选一个资源最松的 IoT 作为兜底
                cand = []
                for i, did in enumerate(self.device_ids):
                    if i not in self._iot_index_set:
                        continue
                    dev = self.device_map[did]
                    has = cap in (dev.sense_cap if cap_type == "sense" else dev.act_cap)
                    if has:
                        cand.append(i)

                if len(cand) > 0:
                    # 选一个“最松”的设备兜底，避免该head全0
                    def slack(i):
                        did = self.device_ids[i]
                        used4 = self.resource_used[did][[1, 3, 4, 5]]
                        cap4 = self._cap4(self.device_map[did])
                        return float(np.sum((cap4 - used4) / (cap4 + 1e-6)))

                    best_i = max(cand, key=slack)
                    mask[k, best_i] = 1.0
                else:
                    # 真不可行：IoT上不存在该cap，留全0，并在 task 层做屏蔽（见下）
                    if self.debug_level >= 1:
                        print(f"[DEBUG] truly infeasible {cap_type} cap={cap}: no IoT has this capability.")

        return mask

    def _obs(self):
        # 设备资源
        dev_feats = np.zeros((self.num_devices, 6), dtype=np.float32)
        for i, did in enumerate(self.device_ids):
            res = self.device_map[did].resource
            initial = np.array([
                getattr(res, "cpu", 0.0), getattr(res, "num", 0.0),
                getattr(res, "gpu", 0.0), getattr(res, "gpu_mem", 0.0),
                getattr(res, "ram", 0.0), getattr(res, "disk", 0.0)
            ], dtype=np.float32)
            remain = initial - self.resource_used[did]
            denom = np.maximum(1.0, initial)
            dev_feats[i, :] = remain / denom

        # 约束/进度（简单全局指标）
        total_nodes = sum(len(tg.G.nodes) for tg in self.task_graphs)
        placed_nodes = 0
        for tg in self.task_graphs:
            tid = int(getattr(tg, "id", -1))
            placed_nodes += len(self.assigned_nodes_by_task.get(tid, set()))
        constraints = np.array([
            0.0,
            placed_nodes / max(1, total_nodes),
            0.0,
            (sum(len(self.current_modules_by_task.get(int(getattr(tg, "id", -1)), [])) for tg in self.task_graphs)) /
            max(1, sum(len(self.modules_queue_by_task.get(int(getattr(tg, "id", -1)), [])) +
                       len(self.current_modules_by_task.get(int(getattr(tg, "id", -1)), []))
                       for tg in self.task_graphs)),
            # 下面两个占位：可以做成平均的感知/执行比例
            0.0, 0.0,
            self.step_count / max(1, total_nodes * 5),
            0.0
        ], dtype=np.float32)

        # 多任务掩码拼接
        task_ids = [int(getattr(tg, "id", -1)) for tg in self.task_graphs]
        T = len(task_ids)
        task_mask = np.zeros((T,), dtype=np.float32)
        compute_mask = np.zeros((T, self.num_devices), dtype=np.float32)
        sense_mask = np.ones((T, self.K_s_max, self.num_devices), dtype=np.float32)
        act_mask = np.ones((T, self.K_a_max, self.num_devices), dtype=np.float32)

        for t_idx, tid in enumerate(task_ids):
            q = self.modules_queue_by_task.get(tid, [])
            task_mask[t_idx] = 1.0 if len(q) > 0 else 0.0
            cm = self.compute_mask_by_task.get(tid)
            sm = self.sense_mask_by_task.get(tid)
            am = self.act_mask_by_task.get(tid)
            if cm is not None:
                compute_mask[t_idx, :] = cm
            if sm is not None:
                sense_mask[t_idx, :, :] = sm
            if am is not None:
                act_mask[t_idx, :, :] = am

        return {
            "device_resources": dev_feats,
            "constraints": constraints,
            "task_mask": task_mask,
            "compute_mask": compute_mask,
            "sense_mask": sense_mask,
            "act_mask": act_mask,
        }

    def _all_tasks_done(self) -> bool:
        return all(len(q) == 0 for q in self.modules_queue_by_task.values())

    def _explain_cap_row(self, cap: int, cap_type: str):
        rows = []
        usage = self.sense_cap_usage if cap_type == "sense" else self.act_cap_usage
        quota = self.sense_cap_quota_per_device if cap_type == "sense" else self.act_cap_quota_per_device
        for did in self.device_ids:
            dev = self.device_map[did]
            has_cap = cap in (dev.sense_cap if cap_type == "sense" else dev.act_cap)
            used = usage[did][cap]
            reason = (
                "device_lacks_cap" if not has_cap
                else ("quota_exhausted" if used >= quota else "OK")
            )
            rows.append({
                "device_id": int(did),
                "has_cap": bool(has_cap),
                "used": int(used),
                "quota": int(quota),
                "reason": reason
            })
        return rows

    def _scan_infra_availability(self):
        missing = {"sense": [], "act": []}
        iot_devs = [self.device_map[self.device_ids[i]] for i in self.iot_indices]

        for cap in self.curr_req_sense:
            if not any(cap in dev.sense_cap for dev in iot_devs):
                missing["sense"].append(int(cap))
        for cap in self.curr_req_act:
            if not any(cap in dev.act_cap for dev in iot_devs):
                missing["act"].append(int(cap))
        return missing

    def _finalize_all_tasks(self):
        info = {}
        try:
            total_mk, total_energy = 0.0, 0.0
            energy_bd_all = {"exec": 0.0, "comm": 0.0, "total": 0.0}

            # ==== 汇总每个任务的 makespan/能耗 ====
            for tg in self.task_graphs:
                tid = int(getattr(tg, "id", -1))
                placement = {(tid, m['module_id']): m for m in self.current_modules_by_task.get(tid, [])}
                if len(placement) == 0:
                    continue
                mk, en, ebd = evaluation_func_rl(placement, tg, self.agent_lookup, self.device_map, self.gw_matrix)
                total_mk += float(mk)
                total_energy += float(en)
                energy_bd_all["exec"] += float(ebd.get("exec", 0.0))
                energy_bd_all["comm"] += float(ebd.get("comm", 0.0))
                energy_bd_all["total"] += float(ebd.get("total", 0.0))

            # ==== 软/硬约束统计 ====
            lim_lat = float(self.constraint_limits.get("latency", self.LAT_TOL))
            lim_bw = float(self.constraint_limits.get("bandwidth", self.BW_TOL))
            hard_ok = (self.ep_flags["illegal"] == 0 and self.ep_flags["cap"] == 0 and self.ep_flags["resource"] == 0)
            soft_ok = (self.ep_cost_max["latency"] <= lim_lat) and (self.ep_cost_max["bandwidth"] <= lim_bw)
            feasible = bool(hard_ok)

            over_lat = max(0.0, self.ep_cost_max["latency"] - lim_lat)
            over_bw = max(0.0, self.ep_cost_max["bandwidth"] - lim_bw)
            soft_over = over_lat + over_bw
            hard_cnt = float(self.ep_flags["illegal"] + self.ep_flags["cap"] + self.ep_flags["resource"])
            all_done, dep_detail = self.check_all_modules_deployed()
            completion = float(self._progress_done) / max(1, self._progress_total_modules)  # [0,1]

            self._soft_ema = 0.95 * getattr(self, "_soft_ema", 1.0) + 0.05 * max(soft_over, 1e-6)

            # ==== 记录 info ====
            info.update({
                "status": "success_feasible" if feasible else "infeasible",
                "makespan": float(total_mk),
                "energy": float(total_energy),
                "energy_breakdown": {k: float(v) for k, v in energy_bd_all.items()},
                "feasible": feasible,
                "max_latency_cost": self.ep_cost_max["latency"],
                "max_bandwidth_cost": self.ep_cost_max["bandwidth"],
                "limits": {"latency": lim_lat, "bandwidth": lim_bw},
                "hard_violations": dict(self.ep_flags),
                "violation_log": list(self.violation_log),
                "completion": completion,
                "soft_over": {"lat": over_lat, "bw": over_bw, "sum": soft_over},
                "all_modules_deployed": bool(all_done),
                "deployment_detail": dep_detail,
            })

            # ==== 归一化基准（EMA）====
            self._mk_ema = 0.95 * getattr(self, "_mk_ema", max(1e-6, total_mk)) + 0.05 * max(1e-6, total_mk)
            self._en_ema = 0.95 * getattr(self, "_en_ema", max(1e-6, total_energy)) + 0.05 * max(1e-6, total_energy)
            mk_norm = float(total_mk) / max(1e-6, self._mk_ema)
            en_norm = float(total_energy) / max(1e-6, self._en_ema)

            # ==== 软约束（统一口径：都用“归一 + 同一常数”）====
            # 你前面已经有：over_lat, over_bw, soft_over = over_lat + over_bw
            # 用 EMA 做稳定归一（与 step 的统计分布一致）
            self._soft_ema = 0.95 * getattr(self, "_soft_ema", max(1e-3, soft_over)) + 0.05 * max(1e-3, soft_over)
            soft_norm = float(soft_over / max(1e-6, self._soft_ema))
            # 为避免尖峰，做截断（Huber 风格）
            soft_norm = float(min(soft_norm, 3.0))

            # ==== 终局权重（建议放到 __init__ 可配置）====
            W_MK = getattr(self, "W_FINAL_MAKESPAN", 12.0)
            W_EN = getattr(self, "W_FINAL_ENERGY", 2.5)
            K_LEFT = getattr(self, "K_LEFTOVER", 20.0)
            K_HARD = getattr(self, "K_HARD_VIOL", 0.2)

            # 是否在 episode 再对 soft_over 计罚：
            # - 如果 step 已经对 lat/bw 做了 shaping（你的情况），这里建议用一个 **较小** 的终局软惩罚（或仅通过 bonus gating 体现）
            USE_EP_SOFT_PENALTY = getattr(self, "USE_EP_SOFT_PENALTY", True)
            K_SOFT = getattr(self, "K_SOFT_OVER_EP", 1)  # 比原来的 K_SOFT_OVER 小一些

            # ✅ 不要覆盖 feasible！沿用上面根据限值算出的 feasible
            # feasible = True  # ← 删掉这行

            # 终局奖励：bonus 只在“完成且可行”时发放
            FINISH_BONUS = 0
            base_bonus = (FINISH_BONUS if (completion >= 0.999 and feasible) else -FAIL_PENALTY)

            soft_term = (K_SOFT * soft_norm) if USE_EP_SOFT_PENALTY else 0.0

            final_reward = (
                    - W_MK * mk_norm
                    - W_EN * en_norm
                    - K_LEFT * (1.0 - completion)
                    - 10 * hard_cnt
                    - 50*self.ep_cost_sum["latency"]
                    - self.ep_cost_sum["bandwidth"]
                    + base_bonus
            )

            # 统一 info：保持“显示用 penalty”与 reward 同口径（方便对齐）
            info["penalty"] = float(
                W_MK * mk_norm
                + W_EN * en_norm
                + K_LEFT * (1.0 - completion)
                + K_HARD * hard_cnt
                + (K_SOFT * soft_norm if USE_EP_SOFT_PENALTY else 0.0)
            )

            # ==== 写入 episode 详细分解，便于可视化 ====
            info["epinfo"] = {
                "r_final": float(final_reward),  # 你设计的“终局一步”奖励
                "mk": float(total_mk),
                "E": float(total_energy),
                "soft_over": float(soft_over),
                "comp": float(completion),
                "hard": int(hard_cnt),
                "lat_sum": float(self.ep_cost_sum["latency"]),
            }

            # 返回下一观测（Gym 接口需要），但已终止
            return self._obs(), float(final_reward), True, False, info

        except Exception as e:
            info.update({
                "status": f"fail_eval_error: {e}",
                "makespan": -1.0, "energy": -1.0,
                "penalty": float("inf"),
            })
            final_reward = -getattr(self, "FAIL_PENALTY", 2.0)
            info["episode"] = {"r": float(final_reward), "l": int(self.step_count)}
            return self._obs(), float(final_reward), True, False, info

    def _refresh_all_masks(self):
        for tg in self.task_graphs:
            self._prepare_masks_for_task(tg)


    def step(self, action: np.ndarray):

        """
        解码多头动作 -> 修正任何“mask 不可行”的选择 ->
        构造 module_assignment ->
        计算 step 奖励（含三类链路）-> 接纳并更新资源 ->
        若完成/截断：做一次“全局终评”（makespan/能耗/违约）
        """
        R_PROGRESS = 1.0
        W_COMM_ENERGY = 1.0
        W_EXEC_ENERGY = 1.0

        # 违法动作/能力/资源的单步惩罚与提前终止策略
        P_ILLEGAL = 5.0  # 单步非法动作（比如 head 选择到 mask=0）的惩罚基数
        P_CAP = 10.0  # 能力不匹配（soft/sense/act 缺失）惩罚基数
        P_RESOURCE = 10.0  # 资源越界（num/gpu_mem/ram/disk）惩罚基数
        MAX_ILLEGAL_STEPS = 25  # 单回合最多允许多少次“硬违规步”
        terminate_on_illegal = True  # 达到阈值是否提前终止

        if self.debug_level >= 2:
            self._check_action_vs_mask(action)
        self.step_count += 1
        # 如果全任务都完成，直接终局
        if self._all_tasks_done():
            return self._finalize_all_tasks()

        # 若当前 active 任务为空或其队列为空，则切到下一个
        if (self.active_tid is None) or (len(self.modules_queue_by_task.get(self.active_tid, [])) == 0):
            self._pick_next_task()
            if (self.active_tid is None):
                return self._finalize_all_tasks()

        tid = self.active_tid
        tg = next(t for t in self.task_graphs if int(getattr(t, "id", -1)) == tid)
        queue = self.modules_queue_by_task[tid]
        if not queue:
            self._pick_next_task()
            if (self.active_tid is None):
                return self._finalize_all_tasks()
            tid = self.active_tid
            tg = next(t for t in self.task_graphs if int(getattr(t, "id", -1)) == tid)
            queue = self.modules_queue_by_task[tid]
            if not queue:
                return self._finalize_all_tasks()

        action = np.asarray(action, dtype=int)

        # ✅ 从“该任务”的队列里弹出
        mod = queue.pop(0)
        aid = mod["agent_id"]
        agent = self.agent_lookup[aid]

        # 掩码取该任务的
        cm = self.compute_mask_by_task[tid]
        sm = self.sense_mask_by_task[tid]
        am = self.act_mask_by_task[tid]
        # 解码多头动作
        # 解码并修正非法动作
        # —— 解码（无任务头）——
        illegal_caps, illegal_details = 0, []
        compute_dev = -1

        # head 0: compute
        comp_idx = int(np.clip(action[0], 0, self.num_devices - 1))
        if (cm > 0.5).sum() == 0 or (cm[comp_idx] < 0.5):
            illegal_caps += 1
            illegal_details.append({"type": "compute"})
        else:
            compute_dev = self.idx2dev[comp_idx]

        # 传感头
        sense_map, act_map = {}, {}

        base = 1  # 动作索引：0=task, 1=compute_head

        req_s = self.curr_req_sense_by_task.get(tid, [])
        for k in range(self.K_s_max):
            if k >= len(req_s):  # 未使用位
                continue
            cap = req_s[k]
            local = int(np.clip(action[base + k], 0, self.num_iot - 1))
            idx = int(self.iot_indices[local]) if self.num_iot > 0 else -1
            if (idx < 0) or (sm[k, idx] < 0.5):
                illegal_caps += 1
                illegal_details.append({"type": "sense", "cap": int(cap)})
            else:
                sense_map[cap] = self.idx2dev[idx]

        # 驱动头
        base2 = base + self.K_s_max
        req_a = self.curr_req_act_by_task.get(tid, [])
        for k in range(self.K_a_max):
            if k >= len(req_a):  # 未使用位
                continue
            cap = req_a[k]
            local = int(np.clip(action[base2 + k], 0, self.num_iot - 1))
            idx = int(self.iot_indices[local]) if self.num_iot > 0 else -1
            if (idx < 0) or (am[k, idx] < 0.5):
                illegal_caps += 1
                illegal_details.append({"type": "act", "cap": int(cap)})
            else:
                act_map[cap] = self.idx2dev[idx]
        # 组装
        m = {
            "nodes": set(mod["nodes"]),
            "agent_id": aid,
            "soft_device": compute_dev,
            "sense_map": sense_map,
            "act_map": act_map,
            "module_id": len(self.current_modules_by_task[tid]),
        }
        _, step_costs, step_details = self._tri_part_step_reward_for_task(tid, tg, m, illegal_caps)

        # 在 step() 中，计算 reward、step_costs、step_details 之后：
        hard_violation = (
                step_costs.get("illegal_caps", 0.0) > 0.0 or
                step_costs.get("cap_violation", 0.0) > 0.0 or
                step_costs.get("resource_violation", 0.0) > 0.0
        )

        if hard_violation:
            # ❌ 不接纳：把模块放回队列开头
            # self.modules_queue_by_task[tid].insert(0, mod)
            self.modules_queue_by_task[tid].append(mod)
            # 刷新掩码并切换下一个任务，生成下一状态观测
            self._refresh_all_masks()
            obs = self._obs()
            self._last_obs = obs
            # === PPO用：按违规强度给负奖励 ===
            g_illegal = float(step_costs.get("illegal_caps", 0.0))  # 多少个子位非法
            g_cap = float(step_costs.get("cap_violation", 0.0))  # 缺失能力计数
            g_res = float(step_costs.get("resource_violation", 0.0))  # 资源越界(0/1)

            self.ep_flags["illegal"] += g_illegal
            self.ep_flags["cap"] += g_cap
            self.ep_flags["resource"] += g_res
            penalty = (
                    P_ILLEGAL * g_illegal +
                    P_CAP * g_cap +
                    P_RESOURCE * g_res
            )
            # 少量时间成本，避免 agent 在违规状态空转
            reward = -(penalty + 0.5 * self.STEP_TIME_COST)
            # 覆盖奖励：拒绝不许拿进度奖励
            info = {
                "costs": step_costs, "violations_this_step": step_details,
                "rejected": True, "accepted": False,
                "rew_terms": {"penalty": float(penalty)},
                "mask_stats": self._mask_stats()
            }
            if self.debug_level >= 1:
                info["mask_stats"] = self._mask_stats(self._last_obs) if hasattr(self, "_last_obs") else {}

                # ✅ 返回缓存的观测，保持与 ActionMasker 同步
            return self._last_obs if hasattr(self, "_last_obs") and self._last_obs is not None else self._obs(), \
                float(reward), False, False, info

        # 接纳 & 更新
        self.current_modules_by_task[tid].append(m)
        self.assigned_nodes_by_task[tid].update(mod["nodes"])
        self.resource_used[compute_dev] += np.array(agent.r, dtype=np.float32)
        self._progress_done += 1
        progress_delta = 1.0 / max(1, self._progress_total_modules)
        # —— 新增：更新能力占用计数（用于后续mask） ——
        for cap, dev_id in sense_map.items():
            if dev_id is not None:
                self.sense_cap_usage[dev_id][cap] += 1
        for cap, dev_id in act_map.items():
            if dev_id is not None:
                self.act_cap_usage[dev_id][cap] += 1

        # 🔁 关键：更新后统一刷新所有任务的 mask
        self._refresh_all_masks()

        obs = self._obs()
        self.last_obs = obs
        terminated = self._all_tasks_done()
        # 给予更多步数余量
        max_steps = int(1 * sum(len(tg.G.nodes) for tg in self.task_graphs))
        truncated = (self.step_count >= max_steps)
        # 计算 step 奖励（接受时）
        progress_delta = 1.0 / max(1, self._progress_total_modules)

        # ==== step(): 硬违规 ====
        # 这些是你 already 计算好的“归一化” costs（建议弱裁剪到 [0, 3~5] 范围）
        lat = float(np.clip(step_costs.get("latency", 0.0), 0.0, 1.0))
        bw = float(np.clip(step_costs.get("bandwidth", 0.0), 0.0, 1.0))
        commE = float(np.clip(step_costs.get("comm_energy", 0.0), 0.0, 1.0))
        execE = float(np.clip(step_costs.get("exec_energy", 0.0), 0.0, 1.0))

        self.ep_cost_max["latency"] = max(self.ep_cost_max["latency"], lat)
        self.ep_cost_max["bandwidth"] = max(self.ep_cost_max["bandwidth"], bw)
        self.ep_cost_sum["latency"] += lat  # 可选
        self.ep_cost_sum["bandwidth"] += bw  # 可选
        self.ep_cost_cnt += 1  # 可选

        energy = commE + execE
        C_PROGRESS = 1.0
        W_E = 0.5  # 能耗较“贵”
        W_LAT = 5
        W_BW = 1
        TOL = 0.05  # 给 5% 的容忍区；要更严格就设 0.0

        if getattr(self, "reward_mode", "") == "penalty":
            lat_over = max(0.0, lat - TOL)
            bw_over = max(0.0, bw - TOL)
            reward = (
                    progress_delta
                    - W_LAT * lat_over
                    - W_BW * bw_over
                    - W_E * energy
                    - self.STEP_TIME_COST)

            info = {
                "costs": step_costs, "accepted": True, "rejected": False,
                "rew_terms": {"lat_over": float(lat_over),
                              "bw_over": float(bw_over),
                              "energy": float(W_COMM_ENERGY * commE + W_EXEC_ENERGY * execE),
                              "net": float(reward)},
                "mask_stats": self._mask_stats()
            }
        else:
            # 关键：整块乘 progress_delta，使得一条完整链路累计奖励 ~ O(1)
            reward = progress_delta * (
                    C_PROGRESS  # 正向推进基线（如 1.0）
                    - W_E * energy  # 减能耗
                    - W_LAT * lat  # 延迟 shaping
                    - W_BW * bw  # 带宽 shaping
            ) - self.STEP_TIME_COST

            info = {
                "costs": step_costs, "accepted": True, "rejected": False,
                "rew_terms": {
                    "progress": float(R_PROGRESS * progress_delta),
                    "energy": float(W_COMM_ENERGY * commE + W_EXEC_ENERGY * execE),
                    "net": float(reward),
                },
                "mask_stats": self._mask_stats()
            }

        if terminated or truncated:
            return self._finalize_all_tasks()

        # 为下一步挑选 active 任务（本步已放置完一个模块）
        self._pick_next_task()
        return obs, float(reward), False, False, info

    def check_all_modules_deployed(self):
        """
        判断当前 episode 是否已把所有任务的所有模块部署完成。
        返回:
          all_done: bool
          detail: {
            tid: {
              "total_planned": int,   # 计划中的模块总数
              "deployed": int,        # 已部署数
              "remaining": int,       # 未部署数
              "percent": float,       # 完成比例 0~1
              "missing_module_ids": [int, ...],  # 未部署的 module_id 列表
              "missing_agents": [int, ...],      # 未部署模块对应的 agent_id 列表（如可获取）
            }, ...
          }
        """
        detail = {}
        all_done = True

        # 便利所有任务
        task_ids = [int(getattr(tg, "id", -1)) for tg in self.task_graphs]
        for tid in task_ids:
            cur_list = self.current_modules_by_task.get(tid, [])
            q_list = self.modules_queue_by_task.get(tid, [])

            deployed_ids = {int(m.get("module_id", -1)) for m in cur_list}
            planned_ids = deployed_ids | {int(m.get("module_id", -1)) for m in q_list}
            remaining_ids = sorted(list(planned_ids - deployed_ids))

            # 尝试给出未部署模块对应的 agent_id（从当前队列/已部署里都搜一遍做映射）
            mod2agent = {}
            for m in cur_list:
                mod2agent[int(m.get("module_id", -1))] = int(m.get("agent_id", -1))
            for m in q_list:
                mod2agent[int(m.get("module_id", -1))] = int(m.get("agent_id", -1))
            missing_agents = [mod2agent.get(mid, -1) for mid in remaining_ids]

            total_planned = len(planned_ids)
            deployed_cnt = len(deployed_ids)
            remaining_cnt = total_planned - deployed_cnt
            done_this_task = (remaining_cnt == 0)

            detail[tid] = {
                "total_planned": total_planned,
                "deployed": deployed_cnt,
                "remaining": remaining_cnt,
                "percent": (deployed_cnt / total_planned) if total_planned > 0 else 1.0,
                "missing_module_ids": remaining_ids,
                "missing_agents": missing_agents,
            }
            all_done = all_done and done_this_task

        return all_done, detail

    def _device_for_node_in(self, tid: int, node_id: str, current_module: Dict[str, Any]) -> Optional[int]:
        tg_mods = self.current_modules_by_task.get(tid, [])
        G = next(t for t in self.task_graphs if int(getattr(t, "id", -1)) == tid).G
        node_info = G.nodes[node_id]
        ntype = node_info.get("type", "proc")
        cap_idx = node_info.get("idx")

        if node_id in current_module["nodes"]:
            if ntype == "proc":
                return current_module["soft_device"]
            elif ntype == "sense":
                return current_module["sense_map"].get(cap_idx)
            elif ntype == "act":
                return current_module["act_map"].get(cap_idx)

        for mod in tg_mods:
            if node_id in mod["nodes"]:
                if ntype == "proc":
                    return mod["soft_device"]
                elif ntype == "sense":
                    return mod["sense_map"].get(cap_idx)
                elif ntype == "act":
                    return mod["act_map"].get(cap_idx)
        return None

    def _tri_part_step_reward_for_task(
            self, tid: int, tg: TaskGraph, m: Dict[str, Any], illegal_caps: int
    ) -> tuple[float, dict[str, float], dict[str, Any]]:
        """
        针对“当前任务 tid 的当前模块 m”的一步奖励与约束度量。
        - 仅统计：当前模块 ↔ 已放置(同一任务) 的跨模块链路（避免跨任务串扰）
        - 能力/资源可行性检查失败直接返回负奖励 + 对应 costs
        - 新增：延迟/带宽的轻量 shaping（避免训练早期完全无信号）
        """
        # ===== 轻量 shaping 权重（不与 cSAC 的 λ 冲突） =====

        ILLEGAL_ACTION_PENALTY = 2.0  # 单步的轻负奖励（保持和你原来一致的量级）

        # ===== 统一代价容器 =====
        costs: Dict[str, float] = {
            "illegal_caps": 0.0, "cap_violation": 0.0, "resource_violation": 0.0,
            "latency": 0.0, "bandwidth": 0.0, "comm_energy": 0.0, "exec_energy": 0.0,
            "cap_penalty": 0.0, "resource_penalty": 0.0,
        }
        details: Dict[str, Any] = {}

        # ===== 0) 非法能力（掩码无可行设备导致的“占位动作”） =====
        if illegal_caps > 0:
            costs["illegal_caps"] = float(illegal_caps)
            details["illegal"] = {
                "module_id": int(m.get("module_id", -1)),
                "soft_device": int(m.get("soft_device", -1)),
                "count": int(illegal_caps),
            }
            self._log_violation(tid, "illegal", {**details["illegal"]})
            return 0, costs, details

        # ===== 1) 能力可行性（soft / sense / act） =====
        dev = self.device_map[m["soft_device"]]
        agent = self.agent_lookup[m["agent_id"]]
        req_soft = agent.C_soft

        miss_soft = len(req_soft - dev.soft_cap)
        miss_sense = sum(1 for cap, dv in m["sense_map"].items()
                         if cap not in self.device_map[dv].sense_cap)
        miss_act = sum(1 for cap, dv in m["act_map"].items()
                       if cap not in self.device_map[dv].act_cap)

        if (miss_soft + miss_sense + miss_act) > 0:
            total_miss = miss_soft + miss_sense + miss_act
            penalty_val = float(UNSOLVABLE_MISMATCH_PENALTY * total_miss)
            costs["cap_violation"] = float(total_miss)
            costs["cap_penalty"] = penalty_val
            details["cap"] = {
                "module_id": int(m.get("module_id", -1)),
                "soft_device": int(m.get("soft_device", -1)),
                "missing_soft": int(miss_soft),
                "missing_act": int(miss_act),
                "missing_sense": int(miss_sense),
            }
            self._log_violation(tid, "cap", {**details["cap"]})
            return 0, costs, details

        # ===== 2) 资源可行性（四维 num/gpu_mem/ram/disk） =====
        need6 = np.array(self.agent_lookup[m["agent_id"]].r, dtype=np.float32)
        need4 = need6[[1, 3, 4, 5]]
        used4 = self.resource_used[m["soft_device"]][[1, 3, 4, 5]]
        cap4 = self._cap4(self.device_map[m["soft_device"]])
        if np.any(used4 + need4 > cap4 + 1e-6):
            costs["resource_violation"] = 1.0
            penalty_val = float(W_RES_VIOLATION * 2.0)
            costs["resource_penalty"] = penalty_val

            names = ["num", "gpu_mem", "ram", "disk"]
            over = (used4 + need4) - cap4
            detail_list = []
            for i, n in enumerate(names):
                if over[i] > 1e-6:
                    detail_list.append({
                        "resource": n,
                        "used": float(used4[i]),
                        "need": float(need4[i]),
                        "cap": float(cap4[i]),
                        "excess": float(over[i]),
                    })
            self._log_violation(tid, "resource", {
                "module_id": int(m.get("module_id", -1)),
                "soft_device": int(m.get("soft_device", -1)),
                "details": detail_list,
            })
            return 0.0, costs, {"resource": {"details": detail_list}}

        # ===== 3) 通信违约与通信能耗（仅统计：当前模块 ↔ 已放置(同任务)） =====
        G = tg.G
        lat_violations: List[float] = []
        bw_violations: List[float] = []
        comm_energy = 0.0

        # 缺省参数
        SENSE_DATA_MB_DEFAULT, CTRL_DATA_MB_DEFAULT, PROC_DATA_MB_DEFAULT = 0.5, 0.1, 1.0
        LAT_REQ_DEFAULT, BW_REQ_DEFAULT = 50.0, 0.0

        # 仅看“当前任务”的已放置节点集合
        assigned = self.assigned_nodes_by_task.get(tid, set())

        def _edge_defaults(u_type, v_type, attr):
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
                lat_req = 20.0 if ((u_type == "sense" and v_type == "proc") or
                                   (u_type == "proc" and v_type == "act")) else LAT_REQ_DEFAULT
            return data_mb, bw_req, lat_req

        for u, v, attr in tg.get_dependencies():
            u_in_cur = (u in m["nodes"])
            v_in_cur = (v in m["nodes"])

            u_type = G.nodes[u].get("type", "proc")
            v_type = G.nodes[v].get("type", "proc")

            # A) 模块内链路：u,v 都在当前模块
            if u_in_cur and v_in_cur:
                du = self._device_for_node_in(tid, u, m)
                dv = self._device_for_node_in(tid, v, m)
                if du is None or dv is None or du == dv:
                    continue
                data_mb, bw_req, lat_req = _edge_defaults(u_type, v_type, attr)
                dms, bw_act, ej = compute_transmission_delay(
                    src=self.device_map[du], dst=self.device_map[dv],
                    data_size_mb=data_mb, bw_req=bw_req, gw_matrix=self.gw_matrix
                )
                lat_violations.append(norm_latency(dms, lat_req))
                bw_violations.append(norm_bandwidth(bw_act, bw_req))
                comm_energy += float(ej)
                continue
            # B) 跨模块链路：u,v 其中之一在当前模块；另一端必须已放置（同任务）
            if u_in_cur != v_in_cur:
                other = v if u_in_cur else u
                if other not in assigned:
                    continue  # 另一端还没放，不在本步统计
                du = self._device_for_node_in(tid, u, m)  # 当前端点用 m；已放置端点从历史模组里查
                dv = self._device_for_node_in(tid, v, m)
                if du is None or dv is None or du == dv:
                    continue
                data_mb, bw_req, lat_req = _edge_defaults(u_type, v_type, attr)
                dms, bw_act, ej = compute_transmission_delay(
                    src=self.device_map[du], dst=self.device_map[dv],
                    data_size_mb=data_mb, bw_req=bw_req, gw_matrix=self.gw_matrix
                )
                lat_violations.append(norm_latency(dms, lat_req))
                bw_violations.append(norm_bandwidth(bw_act, bw_req))
                comm_energy += float(ej)
                continue

        # ✅ 只更新“当前任务”的已放置集合
        self.assigned_nodes_by_task.setdefault(tid, set()).update(m["nodes"])

        costs["latency"] = float(max(lat_violations) if lat_violations else 0.0)
        costs["bandwidth"] = float(max(bw_violations) if bw_violations else 0.0)

        # ===== 4) 计算执行能耗 + 即时奖励（含 shaping） =====
        need = np.array(self.agent_lookup[m["agent_id"]].r, dtype=np.float32)
        cpu_cap = float(getattr(dev.resource, "cpu", 0.0))
        gpu_cap = float(getattr(dev.resource, "gpu", 0.0))
        cpu_pow = float(getattr(dev.resource, "cpu_power", 0.0))
        gpu_pow = float(getattr(dev.resource, "gpu_power", 0.0))

        cpu_time = need[0] / cpu_cap if cpu_cap > 0 else 0.0
        gpu_time = need[2] / gpu_cap if gpu_cap > 0 else 0.0
        E_cpu = (cpu_time * cpu_pow) / 3600.0 if cpu_pow > 0 else 0.0
        E_gpu = (gpu_time * gpu_pow) / 3600.0 if gpu_pow > 0 else 0.0
        exec_energy = float(E_cpu + E_gpu)

        # 归一化能耗（与环境初始化时的分母一致）
        comm_e = float(np.clip(comm_energy / self.comm_energy_norm, 0.0, 1.0))
        exec_e = float(np.clip(exec_energy / self.exec_energy_norm, 0.0, 1.0))
        costs["comm_energy"] = comm_e
        costs["exec_energy"] = exec_e

        # 即时奖励：进度 - 能耗 - 轻量(延迟/带宽) shaping
        # reward = (
        #         R_PROGRESS
        #         - (W_COMM_ENERGY * comm_e + W_EXEC_ENERGY * exec_e)
        #         - (W_LAT_SHAPING * costs["latency"])
        #         - (W_BW_SHAPING * costs["bandwidth"])
        # )
        return float(0), costs, details

    def _cap4(self, dev):
        vals = []
        dev_type = str(getattr(dev, "type", "")).lower()
        unlimited = (dev_type in ("edge", "cloud"))
        for name in ("num", "gpu_mem", "ram", "disk"):
            v = getattr(dev.resource, name, None)
            if v is None:
                v = 0.0
            if (v == 0 or v is None) and unlimited:
                v = 1e9  # 视为几乎不受限
            vals.append(float(v))
        return np.array(vals, dtype=np.float32)


def create_tripart_env(seed=42, K_s_max=6, K_a_max=6,
                       max_module_size=8, min_module_size=1,
                       lns_rounds=300, grasp_runs=40, grasp_rcl_k=3,
                       precomputed_plans_path: Optional[str] = "plans.json"
                       ):
    cloud, device_list, edge_list, gw_matrix = load_infrastructure()

    device_num = len(device_list)
    edge_num = len(edge_list)
    # 给 Edge 服务器分配连续 id
    for idx, edge in enumerate(edge_list, start=device_num + 1):
        edge.id = idx  # 例如 IoT 有 25 台，则第一台 Edge 从 26 开始

    device_map = {d.id: d for d in device_list}
    device_map.update({e.id: e for e in edge_list})
    # Cloud 服务器 id
    cloud.id = device_num + edge_num + 1  # 唯一编号

    device_map[cloud.id] = cloud

    df = pd.read_csv("redundant_agent_templates.csv")
    agent_lookup = build_agent_lookup(df)

    task_graphs = []
    for i in range(1, 11):
        tg = TaskGraph()
        with open(f"task/dag{i}_typed_constraint.json", "r") as f:
            data = json.load(f)
            tg.load_from_json(data, i)
            task_graphs.append(tg)

    env = TriPartPlacementEnv(
        task_graphs, agent_lookup, device_map, gw_matrix,
        K_s_max=K_s_max, K_a_max=K_a_max, seed=seed,
        # NEW ↓↓↓
        max_module_size=max_module_size,
        min_module_size=min_module_size,
        lns_rounds=lns_rounds,
        grasp_runs=grasp_runs,
        grasp_rcl_k=grasp_rcl_k,
        grasp_weights=GRASP_DEFAULT_WEIGHTS,
        # 关键：读取预计算方案
        precomputed_plans=None,
        precomputed_plans_path=precomputed_plans_path,
        strict_fingerprint_check=True,
    )
    return DummyVecEnv([lambda: Monitor(env)])


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


def build_and_save_stage1_plans(
        save_path: str = "plans.json",
        max_module_size: int = 8,
        grasp_weights: Optional[Weights] = None,
        params: Optional[GRASPParams] = None,
        seed: int = 42,
):
    """
    一次性对所有 task_graphs 生成 Stage-1 方案并保存。
    使用与你训练相同的 load_infrastructure / build_agent_lookup / TaskGraph 加载流程。
    """
    cloud, device_list, edge_list, gw_matrix = load_infrastructure()
    device_map = {d.id: d for d in device_list}
    device_map.update({e.id: e for e in edge_list})
    cloud.id = max(device_map.keys()) + 1
    device_map[cloud.id] = cloud

    df = pd.read_csv("redundant_agent_templates.csv")
    agent_lookup = build_agent_lookup(df)

    # 加载任务
    task_graphs = []
    table = []  # NEW: 汇总表
    for i in range(1, 11):
        tg = TaskGraph()
        with open(f"task/dag{i}_typed_constraint.json", "r") as f:
            data = json.load(f)
            tg.load_from_json(data, i)
            task_graphs.append(tg)

    rng = np.random.default_rng(seed)
    plans_by_task: Dict[int, Dict[str, Any]] = {}

    # --- 批量生成 plan ---
    all_plans = []
    for tg in task_graphs:
        plan = solve_stage1_partition_and_match(
            tg, agent_lookup,
            max_module_size=max_module_size,
            weights=grasp_weights,
            params=params,
            seed=seed,
            verbose=True,
        )
        ser = serialize_partition_plan(tg.G, plan)
        all_plans.append(ser)
        print(f"[Stage-1] Task {tg.id}: modules={len(plan['modules'])}, "
              f"objective={plan['objective']:.2f}")

    if table:
        print("\n[Stage-1] Summary")
        print(" Task | #Modules | MaxSize | TotalNodes | Objective")
        print("------+----------+---------+------------+----------")
        for tid, k, smax, totaln, obj in table:
            print(f"{tid:5d} | {k:8d} | {smax:7d} | {totaln:10d} | {obj:9.3f}")

    # --- 保存 ---
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_plans, f, ensure_ascii=False, indent=2)
    print(f"[Stage-1] Saved {len(all_plans)} plans to {save_path}")


if __name__ == "__main__":
    # 例：权重（与旧字典等价；可按需调整或置为 None）
    weights = Weights(
        w_module=0.20,  # 模块数权重
        w_interface=1.00,  # 跨模块接口权重
        w_redund=0.30,  # 能力冗余权重
        w_collab=0.60,  # 协作次数权重
        w_balance=0.15  # 模块负载均衡权重
    )

    # 例：参数（含 LNS 轮次/破坏比例/LS 迭代等；可按需调整或置为 None）
    params = GRASPParams(
        ls_iters=200,
        lns_rounds=300,
        lns_destroy_frac=0.25
    )

    # 1) 预生成 plans（统一走 Stage-1: GRASP → LS → LNS）
    # build_and_save_stage1_plans(
    #     save_path="plans.json",
    #     max_module_size=3,
    #     grasp_weights=weights,  # 可传 None 使用默认
    #     params=params,  # 可传 None 使用默认
    #     seed=42
    # )

    # 2) 训练（带约束的离散 SAC）
    print("--- Training TriPart (Constrained Discrete SAC) ---")
    algo = train_tripart_csac(
        total_steps=200_000,
        seed=42,
        K_s_max=3,
        K_a_max=3,
        precomputed_plans_path="plans.json"
    )

    # 3) 评估（沿用你原有 evaluate_tripart 即可，传入一个“带 act 的代理”）
    print("--- Evaluating ---")


    # 简单适配：用算法的 policy.act 来“预测”动作
    class _ActorWrap:
        def __init__(self, algo): self.algo = algo

        def predict(self, obs, deterministic=True):
            a, _ = algo.policy.act(_to_torch_obs(obs, algo.device), deterministic=deterministic)
            return a[0], None


    print("--- Evaluating (cSAC) ---")
    eval_result = evaluate_tripart_csac(algo, episodes=10, seed=999, deterministic=True)
    print("--- Done ---")
