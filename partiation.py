# pip install ortools
import ast
import time

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model
import re, json, hashlib
from typing import List, Dict, Set, Tuple, Any
import networkx as nx

from simulation import Device, Resource


class AgentTemplate:
    def __init__(self, agent_id, C_sense, C_act, C_soft, r=None):
        self.id = agent_id
        self.C_sense = set(C_sense)
        self.C_act = set(C_act)
        self.C_soft = set(C_soft)
        self.r = r  # 资源，可选

    def covers(self, C_sense, C_act, C_soft) -> bool:
        return (C_sense.issubset(self.C_sense) and
                C_act.issubset(self.C_act) and
                C_soft.issubset(self.C_soft))

    def total_capability_size(self):
        return len(self.C_sense | self.C_act | self.C_soft)



# 解析集合字符串字段
def parse_capability_field(cell):
    try:
        sets = ast.literal_eval(cell)
        if isinstance(sets, list) and len(sets) > 0:
            return set().union(*sets)
        return set()
    except Exception:
        return set()

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

def load_infrastructure(config_path="infrastructure_config.json", gw_path="gw_matrix.npy"):
    # 读取 JSON 文件
    with open(config_path, "r") as f:
        data = json.load(f)

    # 解析云服务器
    cloud_data = data["cloud_server"]
    cloud = Device(
        type=cloud_data["type"],
        id=cloud_data["id"],
        conn_type=cloud_data.get("conn_type", "wired"),
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
            conn_type=dev.get("conn_type", "wired"),
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
            conn_type=edge.get("conn_type", "wired"),
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


# ============ 工具：从节点属性/名称解析能力ID ============
def _to_int_set(x) -> Set[int]:
    if x is None:
        return set()
    if isinstance(x, (set, list, tuple)):
        out = set()
        for v in x:
            try:
                out.add(int(v))
            except:
                pass
        return out
    try:
        return {int(x)}
    except:
        pass
    if isinstance(x, str):
        s = x.strip()
        if not s: return set()
        try:
            v = json.loads(s)
            return _to_int_set(v)
        except:
            nums = re.findall(r"-?\d+", s)
            return set(int(n) for n in nums)
    return set()


def _cap_ids_from_node(G: nx.DiGraph, u: str, kind: str) -> Set[int]:
    nd = G.nodes[u]
    for key in ("cap", "cap_id", "caps", "C_sense", "C_act"):
        if key in nd and nd[key] is not None:
            s = _to_int_set(nd[key])
            if s:
                return s
    if "idx" in nd:
        try:
            return {int(nd["idx"])}
        except:
            pass
    nums = re.findall(r"-?\d+", str(u))
    if nums:
        return {int(n) for n in nums}
    h = hashlib.md5(f"{kind}|{u}".encode("utf-8")).hexdigest()
    return {int(h[:8], 16) & 0x7fffffff}


def _edge_weight(G: nx.DiGraph, u: str, v: str) -> int:
    data = G.get_edge_data(u, v, default={})
    w = data.get("w", data.get("weight", 1))
    try:
        w = float(w)
    except:
        w = 1.0
    w = max(0.0, w)
    return int(round(w)) or 1


def cp_sat_partition_and_match_strict_io(
        task_graph: "TaskGraph",
        agent_lookup: Dict[int, "AgentTemplate"],
        *,
        max_module_size: int = 5,  # 只按 proc 计数
        # 目标权重
        lambda_cut: float = 1.5,
        lambda_iface: float = 0.8,
        lambda_module: float = 2.0,
        lambda_template: float = 2.0,
        lambda_red: float = 0.0,
        w_red_sense: float = 0.5,
        w_red_act: float = 1.0,
        lambda_cost: float = 0.0,
        lambda_exit_multi: float = 1.2,
        time_limit_s: float = 10.0,
        workers: int = 8,
        debug: bool = True,
        debug_dump_modules: bool = True,  # 可行时：打印模块 IO/覆盖信息
        debug_quick_diag: bool = True,  # 不可行时：做轻量诊断（逐类约束放松重跑）
        # ==== 新增：传感能力协作 ====
        allow_sense_collab: bool = True,  # 开启传感能力协作
        L_sense_per_cap: int = 1,  # 每种传感能力全局最多协作次数（默认 1）
        lambda_collab: float = 0.8,  # 协作惩罚（越大越抑制协作）
) -> Dict[str, Any]:
    """
    一体化 CP-SAT（严格 IO + 有向可达 + 内部封闭）：
      - 划分包含 sense/proc/act 全部节点；
      - 严格 IO：仅 entry/exit 允许跨模块边；internal 对外零边；
      - 模板自覆盖 soft/sense/act；
      - 有向可达连通（从入口或模块内无前驱的根出发，单货流保证）；
      - 冗余/接口/切割/模块数/模板数/多出口等目标项。
    """
    G = task_graph.G
    nodes_all = list(G.nodes())
    types = {u: G.nodes[u].get("type", "proc") for u in nodes_all}

    V_sense = [u for u in nodes_all if types[u] == "sense"]
    V_proc = [u for u in nodes_all if types[u] == "proc"]
    V_act = [u for u in nodes_all if types[u] == "act"]
    V = V_sense + V_proc + V_act
    E = list(G.edges())

    if not V_proc:
        return {"modules": [], "objective": 0.0, "status": "FEASIBLE"}

    # ---- 能力 ID 解析 ----
    def cap_ids_from_node(u: str, kind: str) -> Set[int]:
        return _cap_ids_from_node(G, u, kind)

    # proc 的 soft 需求（默认从 idx；兜底用名称/哈希）
    soft_id: Dict[str, int] = {}
    for p in V_proc:
        nd = G.nodes[p]
        sid = None
        if "idx" in nd:
            try:
                sid = int(nd["idx"])
            except:
                sid = None
        if sid is None:
            sset = cap_ids_from_node(p, "soft")
            sid = min(sset) if sset else None
        if sid is None:
            raise ValueError(f"proc 节点 {p} 无法解析 soft/idx")
        soft_id[p] = sid

    sense_id = {s: list(cap_ids_from_node(s, "sense"))[0] for s in V_sense}
    act_id = {a: list(cap_ids_from_node(a, "act"))[0] for a in V_act}

    SOFT_IDS = sorted({soft_id[p] for p in V_proc})
    SENSE_IDS = sorted(set(sense_id.values()))
    ACT_IDS = sorted(set(act_id.values()))

    # ---- 模板库 ----
    T = sorted(agent_lookup.keys())
    C_soft = {t: set(int(x) for x in getattr(agent_lookup[t], "C_soft", set())) for t in T}
    C_sense = {t: set(int(x) for x in getattr(agent_lookup[t], "C_sense", set())) for t in T}
    C_act = {t: set(int(x) for x in getattr(agent_lookup[t], "C_act", set())) for t in T}
    costT = {t: float(getattr(agent_lookup[t], "cost", 0.0)) for t in T}

    # ---- 模块上限（稳妥上界；可按需改成 len(V_proc) 放宽）----
    P = len(V_proc)
    # Kmax = max(1, min(P, math.ceil(P / max_module_size) + 3))
    Kmax = 10
    K = list(range(Kmax))

    # ---- CP-SAT 变量 ----
    model = cp_model.CpModel()

    # 分配/启用
    x = {(v, k): model.NewBoolVar(f"x[{v},{k}]") for v in V for k in
         K}  # x[v,k]=1表示节点属于哪个模块 例如 x[proc2,2]=1,表示将proc2节点纳入到模块2
    m = {k: model.NewBoolVar(f"m[{k}]") for k in K}  # m[k]=1 表示模块k被启用，例如m[2]=1表示模块2被启动

    # 模板
    y = {(k, t): model.NewBoolVar(f"y[{k},{t}]") for k in K for t in T}  # y[k,t]=1 表示模块k选择了模板t，例如y[1,2]=1 表示模块1选择了模板2
    t_used = {t: model.NewBoolVar(f"tused[{t}]") for t in T}  # # 表示模板是否被使用过1次

    # I/O 角色 + cut
    entry = {(v, k): model.NewBoolVar(f"entry[{v},{k}]") for v in V for k in K}  # entry[v,k]=1 表示节点v是为模块k的入口节点
    exit = {(v, k): model.NewBoolVar(f"exit[{v},{k}]") for v in V for k in K}  # exit[v,k]=1 表示节点v为模块k的出口节点
    internal = {(v, k): model.NewBoolVar(f"in[{v},{k}]") for v in V for k in K}  # internal[v,k]=1 表示节点v为模块k的内部节点
    cut = {(u, v): model.NewBoolVar(f"cut[{u},{v}]") for (u, v) in E}  # cut[u,v]=1 表示 u,v节点不在一个模块，有模块间通信代价

    # 需求/可用/冗余
    req_soft = {(k, r): model.NewBoolVar(f"reqSft[{k},{r}]") for k in K for r in
                SOFT_IDS}  # req_soft[k,r]=1 表示模块k需要软件能力r
    req_sense = {(k, s): model.NewBoolVar(f"reqSen[{k},{s}]") for k in K for s in
                 SENSE_IDS}  # req_sense[k,s]=1 表示模块k需要传感能力s
    req_act = {(k, a): model.NewBoolVar(f"reqAct[{k},{a}]") for k in K for a in ACT_IDS}  # req_act[k,a]=1 表示模块k需要驱动能力a

    av_soft = {(k, r): model.NewBoolVar(f"avSft[{k},{r}]") for k in K for r in
               SOFT_IDS}  # av_soft[k,r]=1 表示 模块k 所选的 模板t 提供软件能力r
    av_sense = {(k, s): model.NewBoolVar(f"avSen[{k},{s}]") for k in K for s in
                SENSE_IDS}  # av_sense[k,s]=1 表示模块k 所选的 模板t 提供传感能力s
    av_act = {(k, a): model.NewBoolVar(f"avAct[{k},{a}]") for k in K for a in
              ACT_IDS}  # av_act[k,a]=1 表示模块k 所选的 模板t 提供驱动能力a

    red_soft = {(k, r): model.NewBoolVar(f"redSft[{k},{r}]") for k in K for r in
                SOFT_IDS}  # red_soft[k,r]=1 表示模块k所选的模板t 有冗余软件能力r
    red_sense = {(k, s): model.NewBoolVar(f"redSen[{k},{s}]") for k in K for s in
                 SENSE_IDS}  # red_sense[k,s]=1 表示模块k所选的模板t 有冗余传感能力s
    red_act = {(k, a): model.NewBoolVar(f"redAct[{k},{a}]") for k in K for a in
               ACT_IDS}  # red_act[k,a]=1 表示模块k所选的模板t 有冗余驱动能力a


    # ---- 1) 唯一分配 + 模块容量 + 启用 ----
    # 构建无向图以获得弱连通分量
    undirected_G = nx.Graph()
    undirected_G.add_nodes_from(G.nodes())
    undirected_G.add_edges_from(G.edges())
    undirected_G.add_edges_from([(v, u) for (u, v) in G.edges()])

    # 每个节点对应的连通分量编号
    cc_labels = {}
    for i, comp in enumerate(nx.connected_components(undirected_G)):
        for node in comp:
            cc_labels[node] = i

    # 生成不在同一连通分量的节点对
    unconnected_pairs = []
    V_list = list(G.nodes())
    for i in range(len(V_list)):
        for j in range(i + 1, len(V_list)):
            u, v = V_list[i], V_list[j]
            if cc_labels[u] != cc_labels[v]:
                unconnected_pairs.append((u, v))

    # 添加约束：不同连通分量的节点不能同属一个模块
    for (u, v) in unconnected_pairs:
        for k in K:
            model.Add(x[(u, k)] + x[(v, k)] <= 1)

    for v in V:
        model.Add(sum(x[(v, k)] for k in K) == 1)  # 确保x[v,k]=1 分配的唯一性，即对任意节点v只能分配给一个模块k
    for k in K:
        model.Add(sum(x[(p, k)] for p in V_proc) <= max_module_size * m[k])  # 确保对于任意模块k，其最大proc节点数量< max_module_size
        for v in V:
            model.Add(m[k] >= x[(v, k)])  # 确保对于任意节点 v ，其分配的模块k都被启动

    # ---- 2) 模板：每个模块选且仅选 1 个（或未启用=0）----
    for k in K:
        model.Add(sum(y[(k, t)] for t in T) == m[k])  # 确保模块k 只能选择一个模板t
        for t in T:
            model.Add(t_used[t] >= y[(k, t)])  # 确保 模板t 被启动

    # ---- 3) 严格 IO：entry/exit/internal + cut ----
    for (u, v) in E:
        for k in K:
            # 只要有一条来自其他模块的节点 u 指向 v，v 就必须被标记为入口节点。
            if types[v]== "proc":
                model.Add(entry[(v, k)] >= x[(v, k)] - x[(u, k)])
                # 如果 节点 v 在模块 k中 即 x[v,k]=1，而节点u不在模块k中，即x[u,k]=0, 则entry[v,k]>=1，即entry[v,k]=1;

            # 当一个节点u 有边指向外部节点v时，它必须被标记为出口节点。
            model.Add(exit[(u, k)] >= x[(u, k)] - x[(v, k)])
            # 如果 节点u 在模块k中，即 x[u,k]=1，而节点v不在模块k中，即x[v,k]=0,则exit[u,k]>=1，即exit[u,k]=1

            # 边 (u,v) 是 cut-edge，当且仅当它连接不同模块的节点。
            # 如果边的两端属于不同模块 (x[u,k] ≠ x[v,k])，那么 cut[(u,v)] ≥ 1 ⇒ cut[(u,v)] = 1。
            model.Add(cut[(u, v)] >= x[(u, k)] - x[(v, k)])
            model.Add(cut[(u, v)] >= x[(v, k)] - x[(u, k)])

    # internal = x - entry - exit  （互斥且覆盖）
    for v in V:
        for k in K:
            # 剩下的就都是内部节点了
            # 每个模块内的节点要么是入口、要么是出口、要么是内部节点，三者互斥。
            model.Add(internal[v, k] <= x[v, k])
            model.Add(internal[v, k] <= 1 - entry[v, k])
            model.Add(internal[v, k] <= 1 - exit[v, k])
            model.Add(internal[v, k] >= x[v, k] - entry[v, k] - exit[v, k])

    # ---- 4) 模块需求（存在性 OR）----
    for k in K:
        # soft：聚合模块内 proc 的 idx
        for r in SOFT_IDS:
            Ps = [p for p in V_proc if soft_id[p] == r]
            if Ps:
                for p in Ps:
                    model.Add(req_soft[(k, r)] >= x[(p, k)])
                model.Add(req_soft[(k, r)] <= sum(x[(p, k)] for p in Ps))
            else:
                model.Add(req_soft[(k, r)] == 0)

        # sense：出现即需求（若你要“独占 IO 才计入需求”，可替换为独占逻辑）
        for s in SENSE_IDS:
            Ss = [u for u in V_sense if sense_id[u] == s]
            if Ss:
                for u in Ss:
                    model.Add(req_sense[(k, s)] >= x[(u, k)])
                model.Add(req_sense[(k, s)] <= sum(x[(u, k)] for u in Ss))
            else:
                model.Add(req_sense[(k, s)] == 0)

        # act：出现即需求
        for a in ACT_IDS:
            As = [u for u in V_act if act_id[u] == a]
            if As:
                for u in As:
                    model.Add(req_act[(k, a)] >= x[(u, k)])
                model.Add(req_act[(k, a)] <= sum(x[(u, k)] for u in As))
            else:
                model.Add(req_act[(k, a)] == 0)

    # ---- 5) 模板能力可用（OR 线性化）----
    def feas(attr: str, cap_id: int):
        if attr == "soft":  return [t for t in T if cap_id in C_soft[t]]
        if attr == "sense": return [t for t in T if cap_id in C_sense[t]]
        if attr == "act":   return [t for t in T if cap_id in C_act[t]]
        return []

    for k in K:
        for r in SOFT_IDS:
            feas_t = feas("soft", r)
            if feas_t:
                model.Add(av_soft[(k, r)] <= sum(y[(k, t)] for t in feas_t))
                for t in feas_t: model.Add(av_soft[(k, r)] >= y[(k, t)])
            else:
                model.Add(av_soft[(k, r)] == 0)

        for s in SENSE_IDS:
            feas_t = feas("sense", s)
            if feas_t:
                model.Add(av_sense[(k, s)] <= sum(y[(k, t)] for t in feas_t))
                for t in feas_t: model.Add(av_sense[(k, s)] >= y[(k, t)])
            else:
                model.Add(av_sense[(k, s)] == 0)

        for a in ACT_IDS:
            feas_t = feas("act", a)
            if feas_t:
                model.Add(av_act[(k, a)] <= sum(y[(k, t)] for t in feas_t))
                for t in feas_t: model.Add(av_act[(k, a)] >= y[(k, t)])
            else:
                model.Add(av_act[(k, a)] == 0)

    # ========= 传感能力协作（新增） =========
    # c[k,kk,s] = 1 表示 “模块 k 从 模块 kk 借用 传感能力 s”
    # b[k,s] = OR_kk c[k,kk,s] 表示 “模块 k 对能力 s 发生了协作借用”
    if allow_sense_collab and SENSE_IDS:
        c = {(k, kk, s): model.NewBoolVar(f"collab[{k},{kk},{s}]")
             for k in K for kk in K for s in SENSE_IDS if k != kk}
        b = {(k, s): model.NewBoolVar(f"borrow[{k},{s}]") for k in K for s in SENSE_IDS}
        # 只能向“启用且具备 s 能力”的模块借用；借用方也必须启用
        for k in K:
            for kk in K:
                if k == kk: continue
                for s in SENSE_IDS:
                    # c[k,kk,s] <= m[k], m[kk]
                    model.Add(c[k, kk, s] <= m[k])
                    model.Add(c[k, kk, s] <= m[kk])
                    # c[k,kk,s] <= av_sense[kk,s]  (donor 模块具备该能力)
                    model.Add(c[k, kk, s] <= av_sense[kk, s])

            # b[k,s] = OR_{kk!=k} c[k,kk,s]
            for s in SENSE_IDS:
                donors = [c[(k, kk, s)] for kk in K if kk != k]
                if donors:
                    model.Add(b[(k, s)] <= sum(donors))
                    for d in donors: model.Add(b[(k, s)] >= d)
                else:
                    model.Add(b[(k, s)] == 0)
        # 全局限额：每个能力 s 的协作次数 ≤ L_sense_per_cap
        for s in SENSE_IDS:
            model.Add(sum(b[(k, s)] for k in K) <= max(0, int(L_sense_per_cap)))

        # 覆盖放宽：sense 维度允许使用协作
        for k in K:
            for s in SENSE_IDS:
                model.Add(req_sense[k, s] <= av_sense[k, s] + b[(k, s)])
    else:
        # 不启用协作：保持严格自覆盖
        for k in K:
            for s in SENSE_IDS:
                model.Add(req_sense[k, s] <= av_sense[k, s])

    # ---- 6) 覆盖（严格自覆盖）----
    for k in K:
        for r in SOFT_IDS:  model.Add(req_soft[(k, r)] <= av_soft[(k, r)])
        for a in ACT_IDS:   model.Add(req_act[(k, a)] <= av_act[(k, a)])

    # ---- 7) 冗余 ----
    for k in K:
        for r in SOFT_IDS:  model.Add(red_soft[(k, r)] >= av_soft[(k, r)] - req_soft[(k, r)])
        for s in SENSE_IDS: model.Add(red_sense[(k, s)] >= av_sense[(k, s)] - req_sense[(k, s)])
        for a in ACT_IDS:   model.Add(red_act[(k, a)] >= av_act[(k, a)] - req_act[(k, a)])

    # ---- 8) 多出口惩罚计数 ----
    exit_count = {k: model.NewIntVar(0, len(V), f"exit_count[{k}]") for k in K}
    over_exit = {k: model.NewIntVar(0, len(V), f"over_exit[{k}]") for k in K}
    for k in K:
        model.Add(exit_count[k] == sum(exit[(v, k)] for v in V))
        model.Add(over_exit[k] >= exit_count[k] - 1)
        model.Add(over_exit[k] >= 0)
    # ========= 9) 连通性约束（基于邻接连通性） =========
    # 若两个节点在原图中没有直接边连接，也没有共同邻居，则不能同属一个模块
    for k in K:
        for u in V:
            for v in V:
                if u == v:
                    continue
                # 若 u,v 不相邻且也不共享邻居，则禁止在同模块
                neigh_u = set([x for (x, y) in E if y == u] + [y for (x, y) in E if x == u])
                neigh_v = set([x for (x, y) in E if y == v] + [y for (x, y) in E if x == v])
                if v not in neigh_u and u not in neigh_v and len(neigh_u & neigh_v) == 0:
                    model.Add(x[(u, k)] + x[(v, k)] <= 1)
    # ========= 10) 内部节点对外零边（封闭性硬约束） =========
    # 只要一个点在模块里是 internal，则它所有前驱/后继也必须在该模块
    for (u, v) in E:
        for k in K:
            model.Add(internal[(v, k)] <= x[(u, k)])  # v internal ⇒ u 同模块
            model.Add(internal[(u, k)] <= x[(v, k)])  # u internal ⇒ v 同模块
    # ========= 模块同步延迟惩罚 =========
    # 若模块内部节点 v 对外输出，则整个模块的结果只能在模块结束后提供
    # 我们用 "内部节点对外出边数量" 来近似这个同步延迟成本
    lambda_sync = 0.8  # 可调：同步等待惩罚权重

    sync_penalty = {k: model.NewIntVar(0, len(E), f"sync_penalty[{k}]") for k in K}

    for k in K:
        model.Add(sync_penalty[k] == sum(exit[(v, k)] for v in V if types[v] == "proc"))
    # ---- 11) 目标 ----
    def edge_w(u, v):
        data = G.get_edge_data(u, v, default={})
        w = data.get("w", data.get("weight", 1))
        try:
            return float(w)
        except:
            return 1.0

    obj = 0
    obj += lambda_cut * sum(edge_w(u, v) * cut[(u, v)] for (u, v) in E)
    obj += lambda_iface * sum(entry[(v, k)] + exit[(v, k)] for v in V for k in K)
    obj += lambda_module * sum(m[k] for k in K)
    obj += lambda_template * sum(t_used[t] for t in T)
    obj += lambda_sync * sum(sync_penalty[k] for k in K)
    obj += lambda_red * (
            sum(red_soft[(k, r)] for k in K for r in SOFT_IDS) +
            w_red_sense * sum(red_sense[(k, s)] for k in K for s in SENSE_IDS) +
            w_red_act * sum(red_act[(k, a)] for k in K for a in ACT_IDS)
    )
    obj += lambda_cost * sum(costT[t] * y[(k, t)] for k in K for t in T)
    obj += lambda_exit_multi * sum(over_exit[k] for k in K)

    model.Minimize(obj)

    # ---- 求解 ----
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.log_search_progress = True
    solver.parameters.log_to_stdout = True
    solver.parameters.num_search_workers = int(workers)
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"modules": [], "objective": None, "status": str(status)}

        # ---- 解析 ----
    used_modules = [k for k in K if solver.Value(m[k]) == 1]
    result_modules = []
    for new_id, k in enumerate(used_modules):
        nodes_k = [v for v in V if solver.Value(x[(v, k)]) == 1]
        chosen_t = None
        for t in T:
            if solver.Value(y[(k, t)]) == 1:
                chosen_t = t
                break
        result_modules.append({
            "module_id": new_id,
            "nodes": sorted(nodes_k),
            "agent_id": int(chosen_t) if chosen_t is not None else -1,
            "exit_count": int(solver.Value(sum(exit[(v, k)] for v in V))),
        })
    # ---- 协作关系解析（仅在启用协作时） ----
    if allow_sense_collab and SENSE_IDS:
        collaborations = []
        for k in used_modules:
            for kk in used_modules:
                if k == kk:
                    continue
                for s in SENSE_IDS:
                    var_name = f"collab[{k},{kk},{s}]"
                    if (k, kk, s) in c and solver.Value(c[(k, kk, s)]) == 1:
                        collaborations.append({
                            "from_module": kk,
                            "to_module": k,
                            "cap": int(s),
                        })

        # 将协作信息按模块聚合
        for mod in result_modules:
            mod_id = mod["module_id"]
            mod["collaborations_in"] = [
                cinfo for cinfo in collaborations if cinfo["to_module"] == mod_id
            ]
            mod["collaborations_out"] = [
                cinfo for cinfo in collaborations if cinfo["from_module"] == mod_id
            ]

    return {
        "modules": result_modules,
        "objective": float(solver.ObjectiveValue()),
        "status": "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
    }


def solve_one_task_with_cpsat(
        task_graph: TaskGraph,
        agent_lookup: Dict[int, AgentTemplate],
        *,
        max_module_size: int = 5,
        # 下面这些是新的权重，给了默认值，方便直接跑
        lambda_cut: float = 1.5,
        lambda_iface: float = 0.8,
        lambda_module: float = 2.0,
        lambda_template: float = 2.0,
        lambda_red: float = 0.0,
        w_red_sense: float = 0.5,
        w_red_act: float = 1.0,
        lambda_cost: float = 0.0,
        lambda_exit_multi: float = 1.2,
        time_limit_s: float = 30.0,
        workers: int = 8,
        verbose: bool = True,
) -> Dict[str, Any]:
    """
    调用 CP-SAT 一体化方法，返回与你原先 Stage-1 兼容的结构：
    {
      "task_id": ...,
      "modules": [
        {"module_id": i, "nodes": [...], "agent_id": ...}
      ],
      "objective": <float>,
      "status": "OPTIMAL"/"FEASIBLE"
    }
    """
    if verbose:
        print(f"\n[CP-SAT][Task {getattr(task_graph, 'id', -1)}] Start...")

    t0 = time.perf_counter()
    plan0 = cp_sat_partition_and_match_strict_io(
        task_graph,
        agent_lookup,
        max_module_size=max_module_size,
        lambda_cut=lambda_cut,
        lambda_iface=lambda_iface,
        lambda_module=lambda_module,
        lambda_template=lambda_template,
        lambda_red=lambda_red,
        w_red_sense=w_red_sense,
        w_red_act=w_red_act,
        lambda_cost=lambda_cost,
        lambda_exit_multi=lambda_exit_multi,
        time_limit_s=time_limit_s,
        workers=workers,
    )
    t1 = time.perf_counter()

    if verbose:
        print(f"[CP-SAT][Task {getattr(task_graph, 'id', -1)}] "
              f"status={plan0.get('status')} modules={len(plan0.get('modules', []))} "
              f"obj={plan0.get('objective')} time={t1 - t0:.3f}s")

    return {
        "task_id": getattr(task_graph, "id", -1),
        "modules": plan0["modules"],  # 这里 nodes 是完整集合（含 sense/proc/act）
        "objective": plan0.get("objective"),
        "status": plan0.get("status", "UNKNOWN"),
    }


def batch_build_and_save(
        task_graphs: List[TaskGraph],
        agent_lookup: Dict[int, AgentTemplate],
        out_path: str = "plans_cpsat.json",
        **kwargs,
) -> None:
    """
    批量运行并保存到 JSON。
    kwargs 会透传给 solve_one_task_with_cpsat（例如 max_module_size、lambda_* 等）
    """
    all_plans = []
    for tg in task_graphs:
        plan = solve_one_task_with_cpsat(tg, agent_lookup, **kwargs)
        all_plans.append(plan)
        print(f"[CP-SAT] Task {plan['task_id']}: modules={len(plan['modules'])}, "
              f"objective={plan.get('objective')}, status={plan.get('status')}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_plans, f, ensure_ascii=False, indent=2)
    print(f"[CP-SAT] Saved {len(all_plans)} plans -> {out_path}")


# ============ 演示入口 ============
if __name__ == "__main__":
    try:
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
    except Exception:
        # 如果没有加载器，留空/抛错；你可以改为你的实际数据渠道
        task_graphs = []
        agent_lookup = {}
        print("[CP-SAT] 请在 __main__ 中替换为你的数据加载逻辑。")

    if task_graphs and agent_lookup:
        # 批量调用；kwargs 直接透传到 solve_one_task_with_cpsat
        batch_build_and_save(
            task_graphs,
            agent_lookup,
            out_path="plans_cpsat.json",
            max_module_size=5,
            lambda_cut=1.5,
            lambda_iface=0.8,
            lambda_module=2.0,
            lambda_template=2.0,
            lambda_red=0.8,
            w_red_sense=0.5,
            w_red_act=1.0,
            lambda_cost=0.8,
            lambda_exit_multi=1.2,
            time_limit_s=30.0,
            workers=8,
        )
