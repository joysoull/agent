"""
ABGSK 支持模块（从原始 RL 代码中抽取，仅保留可直接用于 ABGSK 求解的部分）

包含：
- 基础常量
- 基础结构：Device/Resource 对应的 load_infrastructure
- TaskGraph 加载与基本操作
- 能力聚合工具
- 通信时延与能耗计算
- makespan + 能耗评价函数 evaluation_func_rl
- Stage-1 GRASP / 模块划分相关的辅助工具（可选）
- 部署结果报告生成 generate_deployment_report
"""

import json
import random
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Any, Optional

import hashlib

import networkx as nx
import numpy as np
import pandas as pd

from simulation import Device, Resource
from sub_partiation import AgentTemplate, parse_capability_field
from grasp_lns import (
    solve_stage1_partition_and_match,
    Weights,
    GRASPParams,
)

WHITE_NOISE = 1e-9  # W

# ---------- 常量 ----------

P_TX_IOT = 0.10  # W   100 mW
P_RX_IOT = 0.10
P_TX_AP = 0.025  # W    25 mW
P_RX_AP = 0.025

h_e, h_c = 3, 5
E_edge, E_core = 37e-9, 12.6e-9  # J/bit
E_cent = 20e-9  # J/bit
E_per_bit_wired = h_e * E_edge + h_c * E_core + E_cent  # ≈2.075e-5 J/bit

# 传感协作 & 违规惩罚
COLLABORATION_PENALTY_PER_CAP = 1.0
UNSOLVABLE_MISMATCH_PENALTY = 10.0

# 通信能耗估计用的有线能耗常数（J/bit）
COMM_ENERGY_PER_BIT_WIRED = 2.075e-5

# 终局目标权重（可在 ABGSK 中直接复用）
W_FINAL_MAKESPAN = 12.0
W_FINAL_ENERGY = 2.5
W_FINAL_PENALTY = 1.0
K_LEFTOVER = 20.0
K_SOFT_OVER = 50.0
K_HARD_VIOL = 0.2

# ---------------- 基础加载函数 ----------------


def load_infrastructure(
    config_path: str = "infrastructure_config.json",
    gw_path: str = "gw_matrix.npy",
):
    """
    从 JSON + npy 中加载基础设施：
    - cloud_server
    - device_list (IoT 设备)
    - edge_server_list (边缘服务器)
    - gw_matrix (IoT -> GW 邻接矩阵)
    """
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 云服务器
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

    # IoT 设备
    device_list = []
    for dev in data["device_list"]:
        device_list.append(
            Device(
                type=dev["type"],
                id=dev["id"],
                resource=Resource(**dev["resource"]),
                act_cap=set(dev["act_cap"]),
                sense_cap=set(dev["sense_cap"]),
                soft_cap=set(dev["soft_cap"]),
                bandwidth=dev["bandwidth"],
                delay=dev["delay"],
            )
        )

    # 边缘服务器
    edge_list = []
    for edge in data["edge_server_list"]:
        edge_list.append(
            Device(
                type=edge["type"],
                id=edge["id"],
                resource=Resource(**edge["resource"]),
                act_cap=set(edge["act_cap"]),
                sense_cap=set(edge["sense_cap"]),
                soft_cap=set(edge["soft_cap"]),
                bandwidth=edge["bandwidth"],
                delay=edge["delay"],
            )
        )

    gw_matrix = np.load(gw_path)

    return cloud, device_list, edge_list, gw_matrix


def build_agent_lookup(df: pd.DataFrame) -> Dict[int, AgentTemplate]:
    """
    将模板表（redundant_agent_templates.csv）构建为 {agent_id -> AgentTemplate}
    """
    lookup: Dict[int, AgentTemplate] = {}
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
            row["Disk (GB)"],
        )
        lookup[agent_id] = AgentTemplate(agent_id, C_sense, C_act, C_soft, r)
    return lookup


# ---------------- TaskGraph 定义 ----------------


class TaskGraph:
    """
    任务 DAG 封装：从 JSON 中读取 DAG（带 type, idx, 数据量/带宽/时延约束）
    """

    def __init__(self):
        self.G = nx.DiGraph()
        self.id: int = 0

    def load_from_json(self, json_data: dict, task_id: int):
        self.id = task_id
        # 节点
        for node in json_data["nodes"]:
            node_id = node["id"]
            node_type = node.get("type", "proc")
            idx = node.get("idx", -1)
            self.G.add_node(node_id, type=node_type, idx=idx)
        # 边
        for link in json_data["links"]:
            source = link["source"]
            target = link["target"]
            self.G.add_edge(
                source,
                target,
                data_size=link["data_size"],  # MB
                bandwidth_req=link["bandwidth_req"],  # Mbps
                latency_req=link["latency_req"],  # ms
            )

    def get_proc_nodes(self):
        return [n for n, attr in self.G.nodes(data=True) if attr["type"] == "proc"]

    def get_sense_nodes(self):
        return [n for n, attr in self.G.nodes(data=True) if attr["type"] == "sense"]

    def get_act_nodes(self):
        return [n for n, attr in self.G.nodes(data=True) if attr["type"] == "act"]

    def get_all_capability_nodes(self):
        return list(self.G.nodes)

    def get_dependencies(self):
        return list(self.G.edges(data=True))


# ---------------- 能力 / 约束归一化 ----------------


def norm_latency(actual_ms: float, req_ms: Optional[float]) -> float:
    if req_ms is None or req_ms <= 0:
        return 0.0
    return max((actual_ms - req_ms) / max(req_ms, 1e-6), 0.0)


def norm_bandwidth(actual_mbps: float, req_mbps: Optional[float]) -> float:
    if req_mbps is None or req_mbps <= 0:
        return 0.0
    # 需要带宽，实际不足才违反
    return max((req_mbps - actual_mbps) / max(req_mbps, 1e-6), 0.0)


# ---------------- 通信时延与能耗计算 ----------------


def _get_transmission_time_s(data_size_mb: float, rate_mbps: float) -> float:
    if rate_mbps <= 0:
        return float("inf")
    return (8.0 * data_size_mb) / rate_mbps


def compute_transmission_delay(
    src: Device,
    dst: Device,
    data_size_mb: float,
    bw_req: float,
    gw_matrix: np.ndarray,
    edge_inter_delay: float = 1.0,
    cloud_edge_delay: float = 20.0,
) -> Tuple[float, float, float]:
    """
    计算 src -> dst 的传输延迟 / 实际带宽 / 通信能耗。

    返回:
        total_delay_ms: float
        bottleneck_rate_mbps: float
        total_energy_J: float
    """
    if src.id == dst.id:
        return 0.0, float("inf"), 0.0

    src_type = src.type
    dst_type = dst.type

    src_iot = src_type == "Device"
    dst_iot = dst_type == "Device"
    src_edge = src_type == "Edge"
    dst_edge = dst_type == "Edge"
    src_cloud = src_type == "Cloud"
    dst_cloud = dst_type == "Cloud"

    bits = data_size_mb * 8 * 1e6  # bit

    def ul_delay_energy(dev_iot: Device):
        rate = dev_iot.bandwidth  # Mbps
        prop_delay = dev_iot.delay  # ms
        gw_row = gw_matrix[dev_iot.id - 1]
        gw_idx = int(np.where(gw_row > 0)[0][0]) if np.any(gw_row > 0) else -1

        transmission_time_s = _get_transmission_time_s(data_size_mb, rate)
        total_delay_ms = transmission_time_s * 1000 + prop_delay
        energy_j = (P_TX_IOT + P_RX_AP) * transmission_time_s
        return total_delay_ms, rate, energy_j, gw_idx

    def dl_delay_energy(dev_iot: Device, gw_idx: int):
        rate = dev_iot.bandwidth
        prop_delay = dev_iot.delay

        transmission_time_s = _get_transmission_time_s(data_size_mb, rate)
        total_delay_ms = transmission_time_s * 1000 + prop_delay
        energy_j = (P_TX_AP + P_RX_IOT) * transmission_time_s
        return total_delay_ms, rate, energy_j

    # IoT ↔ IoT
    if src_iot and dst_iot:
        T_ul, rate_ul, E_ul, gw_u = ul_delay_energy(src)
        T_dl, rate_dl, E_dl = dl_delay_energy(dst, gw_u)
        same_gw = gw_u == np.where(gw_matrix[dst.id - 1] > 0)[0][0]
        total_delay = T_ul + T_dl + (0 if same_gw else edge_inter_delay)
        total_energy = E_ul + E_dl
        bottleneck_rate = min(rate_ul, rate_dl)
        return total_delay, bottleneck_rate, total_energy

    # IoT → Edge
    if src_iot and dst_edge:
        T_ul, rate_ul, E_ul, _ = ul_delay_energy(src)
        E_wired = E_per_bit_wired * bits
        return T_ul + edge_inter_delay, rate_ul, E_ul + E_wired

    # IoT → Cloud
    if src_iot and dst_cloud:
        T_ul, rate_ul, E_ul, _ = ul_delay_energy(src)
        E_wired = E_per_bit_wired * bits
        return T_ul + cloud_edge_delay, rate_ul, E_ul + E_wired

    # Edge → IoT
    if src_edge and dst_iot:
        gw_idx = np.where(gw_matrix[dst.id - 1] > 0)[0][0]
        T_dl, rate_dl, E_dl = dl_delay_energy(dst, gw_idx)
        E_wired = E_per_bit_wired * bits
        return T_dl + edge_inter_delay, rate_dl, E_dl + E_wired

    # Cloud → IoT
    if src_cloud and dst_iot:
        gw_idx = np.where(gw_matrix[dst.id - 1] > 0)[0][0]
        T_dl, rate_dl, E_dl = dl_delay_energy(dst, gw_idx)
        E_wired = E_per_bit_wired * bits
        return T_dl + cloud_edge_delay, rate_dl, E_dl + E_wired

    # 有线场景（Edge/Cloud 之间）
    WIRED_BANDWIDTH = 1000.0  # 1 Gbps
    E_wired = E_per_bit_wired * bits

    if src_edge and dst_edge:
        return edge_inter_delay, WIRED_BANDWIDTH, E_wired

    if (src_cloud and dst_edge) or (src_edge and dst_cloud):
        return cloud_edge_delay, WIRED_BANDWIDTH, E_wired

    return float("inf"), 0.0, float("inf")


# ---------------- 能力相关工具 ----------------


def get_module_capabilities(G: nx.DiGraph, nodes: set) -> Tuple[set, set, set]:
    """
    聚合模块 nodes 内部需要的三类能力：
    - C_sense：所有 sense 节点的 idx 集合
    - C_act：所有 act 节点的 idx 集合
    - C_soft：所有 proc 节点的 idx 集合
    """
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


# ---------------- DAG 指纹 & Stage-1 计划序列化 ----------------


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
    for u, v, _ in G.edges(data=True):
        edges_sig.append((str(u), str(v)))
    edges_sig.sort()

    m = hashlib.sha256()
    m.update(json.dumps(nodes_sig).encode("utf-8"))
    m.update(json.dumps(edges_sig).encode("utf-8"))
    return m.hexdigest()


def serialize_partition_plan(G: nx.DiGraph, plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    标准化 Stage-1 分区方案格式，便于保存与后续使用（包括 ABGSK 放置）。
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
    返回 JSON friendly 结构:
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
        return sorted([str(x) for x in seq])

    out_modules: List[Dict[str, Any]] = []
    for m in plan.get("modules", []):
        out_modules.append(
            {
                "module_id": int(m["module_id"]),
                "agent_id": int(m.get("agent_id", -1)),
                "nodes": _norm_nodes(m["nodes"]),
            }
        )
    out_modules.sort(key=lambda x: x["module_id"])

    return {
        "fingerprint": _dag_fingerprint(G),
        "task_id": int(plan.get("task_id", -1)),
        "objective": None if plan.get("objective") is None else float(plan["objective"]),
        "modules": out_modules,
    }


def save_partition_plans(plans_by_task: Dict[int, Dict[str, Any]], path: str):
    """
    plans_by_task: { task_id -> serialized_plan_dict }
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plans_by_task, f, ensure_ascii=False, indent=2)


def load_partition_plans(path: str) -> Dict[int, Dict[str, Any]]:
    """
    支持以下格式：
    - {task_id: plan}
    - {"plans": [plan1, plan2, ...]}
    - [plan1, plan2, ...]
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return {int(p["task_id"]): p for p in raw}
    elif isinstance(raw, dict):
        if "plans" in raw and isinstance(raw["plans"], list):
            return {int(p["task_id"]): p for p in raw["plans"]}
        return {int(k): v for k, v in raw.items()}
    else:
        raise ValueError("Unknown JSON format for precomputed plans")


def realize_plan_to_modules_queue(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    将序列化的 plan 转为模块列表：
    [
      {
        "module_id": int,
        "agent_id": int,
        "nodes": set(...)
      }, ...
    ]

    供 ABGSK 自己管理 module_id -> agent_id / nodes 映射。
    """
    out = []
    for m in sorted(plan["modules"], key=lambda x: x["module_id"]):
        out.append(
            {
                "module_id": int(m["module_id"]),
                "agent_id": int(m["agent_id"]),
                "nodes": set(m["nodes"]),
            }
        )
    return out


# ---------------- 简单贪心划分（可选，用于 Stage-1 初始种子） ----------------


def _calculate_module_fitness_collab(
    nodes: set,
    G: nx.DiGraph,
    agent_lookup: Dict[int, AgentTemplate],
    cut_penalty_coef: float = 0.02,
) -> float:
    """
    模块适应度（越小越好），用于贪心生长：
    - C_soft / C_act 缺失：不可协作 => 大罚
    - C_sense 缺失：可跨模块协作 => 小罚
    - cut 惩罚：跨模块边少更好
    - size 惩罚：鼓励模块稍大一些
    """
    if not nodes:
        return float("inf")

    C_sense, C_act, C_soft = get_module_capabilities(G, nodes)
    if not (C_sense or C_act or C_soft):
        return float("inf")

    best_agent_penalty = float("inf")
    for agent in agent_lookup.values():
        miss_soft = len(C_soft - agent.C_soft)
        miss_act = len(C_act - agent.C_act)
        miss_sense_collab = len(C_sense - agent.C_sense)
        penalty = (
            (miss_soft + miss_act) * UNSOLVABLE_MISMATCH_PENALTY
            + miss_sense_collab * COLLABORATION_PENALTY_PER_CAP
        )
        best_agent_penalty = min(best_agent_penalty, penalty)

    # cut 惩罚
    cut_edges = 0
    for u in nodes:
        for v in G.successors(u):
            if v not in nodes:
                cut_edges += 1
        for v in G.predecessors(u):
            if v not in nodes:
                cut_edges += 1
    possible_edges = max(1, len(nodes) * max(1, len(G.nodes) - len(nodes)))
    cut_penalty = cut_penalty_coef * (cut_edges / possible_edges)

    size_penalty = (1.0 / len(nodes)) * 0.1

    return best_agent_penalty + cut_penalty + size_penalty


def partition_with_greedy_algorithm(
    task_graph: TaskGraph,
    agent_lookup: Dict[int, AgentTemplate],
    max_module_size: int = 8,
) -> List[set]:
    """
    协作友好的贪心模块划分算法，作为 GRASP/LNS 的简单替代或初始解。
    """
    G = task_graph.G
    unassigned = set(G.nodes())
    final_modules: List[set] = []

    while unassigned:
        # 1) 选种子：优先 proc
        proc_nodes = [n for n in unassigned if G.nodes[n].get("type") == "proc"]
        if proc_nodes:
            seed = random.choice(proc_nodes)
        else:
            seed = next(iter(unassigned))

        current = {seed}
        unassigned.remove(seed)

        while len(current) < max_module_size:
            neighbors = set()
            for n in current:
                neighbors.update(G.successors(n))
                neighbors.update(G.predecessors(n))
            candidates = neighbors & unassigned
            if not candidates:
                break

            base_fit = _calculate_module_fitness_collab(current, G, agent_lookup)
            best_fit = base_fit
            best_cand = None

            for c in candidates:
                tmp = current | {c}
                fit = _calculate_module_fitness_collab(tmp, G, agent_lookup)
                if fit < best_fit:
                    best_fit = fit
                    best_cand = c

            if best_cand is None:
                break

            current.add(best_cand)
            unassigned.remove(best_cand)

        final_modules.append(current)

    return final_modules


def match_agent_for_module(
    G: nx.DiGraph,
    nodes: set,
    agent_lookup: Dict[int, AgentTemplate],
    strict: bool = True,
) -> Tuple[int, Dict[str, Any]]:
    """
    模块 -> AgentTemplate 的匹配函数，可用于 Stage-1 或 ABGSK 一体化求解。
    """
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
            best_info = {
                "perfect": (miss_soft == 0 and miss_act == 0 and miss_sense == 0),
                "miss_soft": miss_soft,
                "miss_act": miss_act,
                "miss_sense": miss_sense,
            }

    return best_agent_id, best_info


# ---------------- makespan 计算 & 评价函数 ----------------


def compute_task_finish_time(
    task_id: int,
    agent_dag_edges: List[Tuple[Any, Any, Dict]],
    exec_time_map: Dict[Tuple[int, Any], float],
    edge_delay_map: Dict[Tuple[int, Any, Any], float],
) -> float:
    """
    给定“模块三相位图”计算任务 makespan。
    agent_dag_edges: [(u, v, attr), ...]，u/v 为 phase node
    exec_time_map: (task_id, phase_node) -> 执行时间(s)
    edge_delay_map: (task_id, u, v) -> 传输延迟(s)
    """
    succ = defaultdict(list)
    indeg = defaultdict(int)
    modules_in_task = set()

    for u_mod, v_mod, _ in agent_dag_edges:
        succ[u_mod].append(v_mod)
        indeg[v_mod] += 1
        modules_in_task.update([u_mod, v_mod])

    q = deque()
    earliest_start = {}

    for mod in modules_in_task:
        if indeg[mod] == 0:
            earliest_start[mod] = 0.0
            q.append(mod)

    while q:
        u = q.popleft()
        start_time = earliest_start.get(u, 0.0)
        finish_time = start_time + exec_time_map.get((task_id, u), 0.0)

        for v in succ[u]:
            edge_delay = edge_delay_map.get((task_id, u, v), 0.0)
            candidate_start = finish_time + edge_delay
            earliest_start[v] = max(earliest_start.get(v, 0.0), candidate_start)

            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    max_finish = 0.0
    for mod in modules_in_task:
        finish_time = earliest_start.get(mod, 0.0) + exec_time_map.get((task_id, mod), 0.0)
        max_finish = max(max_finish, finish_time)

    return max_finish


def evaluation_func_rl(
    placement: Dict[Tuple[int, int], Dict],
    task_graph: TaskGraph,
    agent_lookup: Dict[int, AgentTemplate],
    device_map: Dict[int, Device],
    gw_matrix: np.ndarray,
) -> Tuple[float, float, Dict[str, float]]:
    """
    **核心评价函数**：给定一个 placement 方案，计算：
    - makespan（任务总完成时间）
    - total_energy（总能耗）
    - energy_bd（包含 exec, comm, total）

    其中 placement 的结构（每个 key 为 (task_id, module_id)）：
    {
      (tid, module_id): {
         "module_id": int,
         "agent_id": int,
         "nodes": set(node_id),
         "soft_device": int,
         "sense_map": {sense_cap -> dev_id},
         "act_map": {act_cap -> dev_id},
      },
      ...
    }

    ABGSK 直接在搜索“模块 -> 设备/协作设备”的编码，然后调用本函数做评价即可。
    """
    # ---------- 建立 node -> module 映射 ----------
    node_to_module_map: Dict[str, int] = {}
    task_id = -1
    for (tid, mod_id), mod_info in placement.items():
        task_id = tid
        for node in mod_info["nodes"]:
            node_to_module_map[node] = mod_id
    if task_id == -1:
        return -1.0, 0.0, {"exec": 0.0, "comm": 0.0, "total": 0.0}

    # ---------- 辅助：查询节点所在设备 ----------
    def get_device_for_node(node_id: str) -> Optional[int]:
        module_id = node_to_module_map.get(node_id)
        if module_id is None:
            return None
        mod_place = placement.get((task_id, module_id))
        if mod_place is None:
            return None
        ninfo = task_graph.G.nodes[node_id]
        ntype = ninfo.get("type", "proc")
        idx = ninfo.get("idx")
        if ntype == "proc":
            return mod_place["soft_device"]
        if ntype == "sense":
            return mod_place["sense_map"].get(idx)
        if ntype == "act":
            return mod_place["act_map"].get(idx)
        return None

    # ---------- 1) 每个模块三相位执行时间 ----------
    exec_time_map: Dict[Tuple[int, Any], float] = {}
    for (tid, module_id), info in placement.items():
        agent = agent_lookup[info["agent_id"]]
        res = device_map[info["soft_device"]].resource
        r_cpu, _, r_gpu, _, _, _ = agent.r
        cpu_cap = float(getattr(res, "cpu", 0.0))
        gpu_cap = float(getattr(res, "gpu", 0.0))
        cpu_time = (r_cpu / cpu_cap) if cpu_cap > 0 else 0.0
        gpu_time = (r_gpu / gpu_cap) if gpu_cap > 0 else 0.0
        core_time = max(cpu_time, gpu_time)

        n_pre = (module_id, "pre")
        n_core = (module_id, "core")
        n_post = (module_id, "post")

        exec_time_map[(task_id, n_pre)] = 0.0
        exec_time_map[(task_id, n_core)] = float(core_time)
        exec_time_map[(task_id, n_post)] = 0.0

    # ---------- 2) 模块内 S->P / P->A 延迟 ----------
    intra_pre_ms = defaultdict(float)
    intra_post_ms = defaultdict(float)

    for u, v, attr in task_graph.get_dependencies():
        u_mod = node_to_module_map.get(u)
        v_mod = node_to_module_map.get(v)
        if u_mod != v_mod or u_mod is None:
            continue
        du = get_device_for_node(u)
        dv = get_device_for_node(v)
        if du is None or dv is None or du == dv:
            continue

        d_ms, _, _ = compute_transmission_delay(
            src=device_map[du],
            dst=device_map[dv],
            data_size_mb=attr.get("data_size", 0.0),
            bw_req=attr.get("bandwidth_req", 0.0),
            gw_matrix=gw_matrix,
        )
        u_type = task_graph.G.nodes[u].get("type")
        v_type = task_graph.G.nodes[v].get("type")

        if u_type == "sense" and v_type == "proc":
            intra_pre_ms[u_mod] = max(intra_pre_ms[u_mod], d_ms)
        elif u_type == "proc" and v_type == "act":
            intra_post_ms[u_mod] = max(intra_post_ms[u_mod], d_ms)

    # ---------- 3) 三相位图边 & 跨模块延迟 ----------
    agent_dag_edges: List[Tuple[Any, Any, Dict]] = []
    edge_delay_map: Dict[Tuple[int, Any, Any], float] = {}
    # 模块内 pre->core, core->post
    for (tid, module_id), _ in placement.items():
        n_pre = (module_id, "pre")
        n_core = (module_id, "core")
        n_post = (module_id, "post")

        agent_dag_edges.append((n_pre, n_core, {}))
        edge_delay_map[(task_id, n_pre, n_core)] = intra_pre_ms[module_id] / 1000.0

        agent_dag_edges.append((n_core, n_post, {}))
        edge_delay_map[(task_id, n_core, n_post)] = intra_post_ms[module_id] / 1000.0

    # 跨模块边：统一 U(core)->V(core)
    inter_core_delay_sec = defaultdict(float)
    incoming_trans_delay = defaultdict(list)
    for u, v, attr in task_graph.get_dependencies():
        u_mod = node_to_module_map.get(u)
        v_mod = node_to_module_map.get(v)
        if u_mod is None or v_mod is None or u_mod == v_mod:
            continue

        du = get_device_for_node(u)
        dv = get_device_for_node(v)
        if du is None or dv is None:
            continue

        d_ms, _, _ = compute_transmission_delay(
            src=device_map[du],
            dst=device_map[dv],
            data_size_mb=attr.get("data_size", 1.0),
            bw_req=attr.get("bandwidth_req", 0.0),
            gw_matrix=gw_matrix,
        )
        delay_sec = float(d_ms) / 1000.0

        src_exec = exec_time_map.get((task_id, (u_mod, "core")), 0.0)
        num_proc = len(
            [
                n
                for n in task_graph.G.nodes()
                if node_to_module_map.get(n) == u_mod
                and task_graph.G.nodes[n].get("type") == "proc"
            ]
        )
        block_wait = src_exec if num_proc > 1 else 0.0
        total_delay = delay_sec + block_wait

        u_core = (u_mod, "core")
        v_core = (v_mod, "core")
        inter_core_delay_sec[(u_core, v_core)] = max(
            inter_core_delay_sec[(u_core, v_core)], total_delay
        )
        incoming_trans_delay[v_core].append(total_delay)

    for (u_core, v_core), d_sec in inter_core_delay_sec.items():
        agent_dag_edges.append((u_core, v_core, {}))
        edge_delay_map[(task_id, u_core, v_core)] = d_sec

    # 同步启动等待
    sync_wait_map = {
        m: max(v) if v else 0.0 for m, v in incoming_trans_delay.items()
    }
    for (tid, module_id), _ in placement.items():
        n_core = (module_id, "core")
        if n_core in sync_wait_map:
            exec_time_map[(task_id, n_core)] += sync_wait_map[n_core]

    # ---------- 4) makespan ----------
    makespan = compute_task_finish_time(
        task_id=task_id,
        agent_dag_edges=agent_dag_edges,
        exec_time_map=exec_time_map,
        edge_delay_map=edge_delay_map,
    )

    # ---------- 5) 能耗 ----------
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
        du = get_device_for_node(u)
        dv = get_device_for_node(v)
        if du is None or dv is None or du == dv:
            continue
        src = device_map[du]
        dst = device_map[dv]
        _, _, ej = compute_transmission_delay(
            src=src,
            dst=dst,
            data_size_mb=attr.get("data_size", 1.0),
            bw_req=attr.get("bandwidth_req", 0.0),
            gw_matrix=gw_matrix,
        )
        if getattr(src, "conn_type", "wired") == "wired" and getattr(
            dst, "conn_type", "wired"
        ) == "wired":
            ej *= 0.1
        comm_energy_J += ej

    energy_bd = {
        "exec": float(exec_energy_J),
        "comm": float(comm_energy_J),
        "total": float(exec_energy_J + comm_energy_J),
    }

    return float(makespan), float(energy_bd["total"]), energy_bd


# ---------------- 部署方案报告（便于可视化与分析） ----------------


def generate_deployment_report(
    placement: Dict[Tuple[int, int], Dict],
    task_graph: TaskGraph,
    agent_lookup: Dict[int, AgentTemplate],
    device_map: Dict[int, Device],
    final_info: Dict[str, Any],
) -> Dict[str, Any]:
    """
    将最终的部署方案（placement）转换成一个详细、易读的字典/JSON 报告。
    """
    task_id = getattr(task_graph, "id", None)
    if task_id is None:
        # fallback：从 placement key 中取
        task_id = list(placement.keys())[0][0]

    report: Dict[str, Any] = {
        "task_id": int(task_id),
        "overall_performance": final_info,
        "deployment_plan": [],
    }

    # 预处理：传感能力 -> 提供者
    sensor_provider_map: Dict[int, Dict[str, int]] = {}
    for (tid, mod_id), mod_info in placement.items():
        agent = agent_lookup[mod_info["agent_id"]]
        for sensor_cap in agent.C_sense:
            sensor_provider_map[sensor_cap] = {
                "agent_id": agent.id,
                "module_id": mod_id,
            }

    sorted_modules = sorted(placement.items(), key=lambda item: item[0][1])

    for (tid, module_id), info in sorted_modules:
        agent = agent_lookup[info["agent_id"]]
        required_sense, _, _ = get_module_capabilities(task_graph.G, info["nodes"])

        missing_sensors = required_sense - agent.C_sense
        collaboration_details = []
        for missing_cap in missing_sensors:
            if missing_cap in sensor_provider_map:
                provider = sensor_provider_map[missing_cap]
                collaboration_details.append(
                    {
                        "capability_id": missing_cap,
                        "status": "Collaborative",
                        "provided_by_agent_id": provider["agent_id"],
                        "in_module_id": provider["module_id"],
                    }
                )
            else:
                collaboration_details.append(
                    {
                        "capability_id": missing_cap,
                        "status": "Missing (Unsolved)",
                        "provided_by_agent_id": None,
                        "in_module_id": None,
                    }
                )

        soft_dev_id = info["soft_device"]
        device_deployments = {
            "software_deployment": {
                "device_id": soft_dev_id,
                "device_type": device_map[soft_dev_id].type,
            },
            "sensor_deployments": [
                {
                    "capability_id": cap_id,
                    "device_id": dev_id,
                    "device_type": device_map[dev_id].type,
                }
                for cap_id, dev_id in info["sense_map"].items()
            ],
            "actuator_deployments": [
                {
                    "capability_id": cap_id,
                    "device_id": dev_id,
                    "device_type": device_map[dev_id].type,
                }
                for cap_id, dev_id in info["act_map"].items()
            ],
        }

        module_report = {
            "module_id": module_id,
            "nodes": sorted(list(info["nodes"])),
            "assigned_agent_template_id": agent.id,
            "deployment_devices": device_deployments,
            "collaboration_details": {
                "required_sensors": sorted(list(required_sense)),
                "provided_by_this_agent": sorted(list(agent.C_sense)),
                "collaborative_sensors_needed": collaboration_details,
            },
        }
        report["deployment_plan"].append(module_report)

    return report


# ---------------- 一次性生成 Stage-1 plans（可选工具） ----------------


def build_and_save_stage1_plans(
    save_path: str = "plans.json",
    max_module_size: int = 8,
    grasp_weights: Optional[Weights] = None,
    params: Optional[GRASPParams] = None,
    seed: int = 42,
):
    """
    一次性对所有任务图生成 Stage-1 模块划分 + Agent 匹配方案，并保存到 save_path。
    方便 ABGSK 在 Stage-2 直接使用 plans.json 作为模块输入。
    """
    cloud, device_list, edge_list, gw_matrix = load_infrastructure()
    device_map: Dict[int, Device] = {d.id: d for d in device_list}
    device_map.update({e.id: e for e in edge_list})
    cloud.id = max(device_map.keys()) + 1
    device_map[cloud.id] = cloud

    df = pd.read_csv("redundant_agent_templates.csv")
    agent_lookup = build_agent_lookup(df)

    task_graphs: List[TaskGraph] = []
    for i in range(1, 11):
        tg = TaskGraph()
        with open(f"task/dag{i}_typed_constraint.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            tg.load_from_json(data, i)
            task_graphs.append(tg)

    rng = np.random.default_rng(seed)
    all_plans = []

    for tg in task_graphs:
        plan = solve_stage1_partition_and_match(
            tg,
            agent_lookup,
            max_module_size=max_module_size,
            weights=grasp_weights,
            params=params,
            seed=int(rng.integers(0, 10_000)),
            verbose=True,
        )
        ser = serialize_partition_plan(tg.G, plan)
        all_plans.append(ser)
        print(
            f"[Stage-1] Task {tg.id}: modules={len(plan['modules'])}, "
            f"objective={plan['objective']:.2f}"
        )

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_plans, f, ensure_ascii=False, indent=2)
    print(f"[Stage-1] Saved {len(all_plans)} plans to {save_path}")
