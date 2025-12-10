import json
import os
import random
from functools import lru_cache
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from simulation import Device, Resource
from partiation import AgentTemplate, parse_capability_field
import networkx as nx
from SAC import evaluation_func_rl, compute_transmission_delay


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


class TaskGraph:
    def __init__(self):
        self.G = nx.DiGraph()

    def load_from_json(self, json_data):
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


def _normalize_module_entry(module: dict) -> Tuple[Optional[int], Optional[int], List[str]]:
    """兼容读取 Stage-1 输出的 module 信息."""

    def _get_value(d: dict, keys):
        for k in keys:
            if k in d:
                return d[k]
        return None

    module_id = _get_value(module, ["Module ID", "module_id"])
    agent_id = _get_value(module, ["Agent ID", "agent_id"])
    nodes = _get_value(module, ["Nodes", "nodes"]) or []
    return module_id, agent_id, nodes


def build_agent_instances(
    folder: str = "partiation",
    prefix: str = "agents_",
    count: int = 10,
    cpsat_path: str = "plans_cpsat.json",
):
    """
    返回：
      instances       : [(task_id, module_id, agent_id), ...]
      module_node_map : {(task_id, node_id): (task_id, module_id)}

    支持两种 Stage-1 输出格式：
    1) GRASP/LNS：partiation/agents_{task_id}.json，每个文件包含模块列表。
    2) CP-SAT：plans_cpsat.json，可为：
       - 顶层 list: [ {task_id, modules:[...]} , ... ]
       - 顶层 dict: { task_id: { ... }, ... } 或 { "plans":[...]}
    """
    instances: List[Tuple[int, int, int]] = []
    node_map: Dict[Tuple[int, str], Tuple[int, int]] = {}

    # ---------- 优先使用 CP-SAT 结果 ----------
    if os.path.exists(cpsat_path):
        with open(cpsat_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # 统一成一个可迭代的 plans 列表
        if isinstance(raw, list):
            plans_iter = raw
        elif isinstance(raw, dict):
            # 兼容 {"plans":[...]} 或 {task_id: plan, ...}
            if "plans" in raw and isinstance(raw["plans"], list):
                plans_iter = raw["plans"]
            else:
                plans_iter = list(raw.values())
        else:
            raise ValueError(f"Unsupported CP-SAT JSON format: {type(raw)}")

        for idx, plan in enumerate(plans_iter, start=1):
            task_id = int(plan.get("task_id", idx))

            for module in plan.get("modules", []):
                module_id, agent_id, nodes = _normalize_module_entry(module)
                if module_id is None or agent_id is None:
                    continue
                module_id = int(module_id)
                agent_id = int(agent_id)

                # 记录 (task, module, agent)
                instances.append((task_id, module_id, agent_id))

                # 记录每个 node 属于哪个模块
                for node in nodes or []:
                    node_map[(task_id, str(node))] = (task_id, module_id)

        return instances, node_map
from collections import defaultdict

def attach_nodes_to_placement(
    placement: Dict[Tuple[int, int], Dict],
    node_map: Dict[Tuple[int, str], Tuple[int, int]],
) -> Dict[Tuple[int, int], Dict]:
    """
    给每个 (task_id, module_id) 的 placement 填充 "nodes" 字段。

    node_map: {(task_id, node_id_str) -> (task_id, module_id)}
    """
    # 先根据 node_map 建一个反向表: (task_id, module_id) -> [nodes]
    module_nodes: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    for (tid, node), (tid2, mid) in node_map.items():
        if tid != tid2:
            continue
        module_nodes[(tid2, mid)].append(node)

    new_placement: Dict[Tuple[int, int], Dict] = {}
    for key, info in placement.items():
        tid, mid = key
        info2 = dict(info)  # 浅拷贝，避免修改原 dict
        # 如果原来没有 nodes，则从 module_nodes 补上；有的话就保留原来的
        if "nodes" not in info2:
            info2["nodes"] = module_nodes.get((tid, mid), [])
        new_placement[key] = info2

    return new_placement




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


def decode_global_placement(chromosome: List[int],
                            agent_instances: List[Tuple[int, int, int]],
                            agent_lookup: Dict[int, AgentTemplate]):
    """
    将整数向量 → { (task,module): {soft: h, sense:{τ:h}, act:{τ:h}} }
    """
    cursor = 0
    placement = {}
    for (task_id, module_id, agent_id) in agent_instances:
        tpl = agent_lookup[agent_id]

        soft_device = chromosome[cursor]
        cursor += 1

        sense_map = {}
        for τ in tpl.C_sense:
            sense_map[τ] = chromosome[cursor]
            cursor += 1

        act_map = {}
        for τ in tpl.C_act:
            act_map[τ] = chromosome[cursor]
            cursor += 1

        placement[(task_id, module_id)] = {
            "agent_id": agent_id,
            "soft_device": soft_device,
            "sense_map": sense_map,
            "act_map": act_map
        }
    return placement


AL_PARAMS = {
    "latency": {"lam": 0.0, "mu": 1e6},
    "bandwidth": {"lam": 0.0, "mu": 1e6},
    "resource": {"lam": 0.0, "mu": 1e7},
    "capability": {"lam": 0.0, "mu": 1e8}  # 新增
}


def _augmented_penalty(h_val: float, lam: float, mu: float) -> float:
    """根据增广拉格朗日形式计算单个约束的惩罚值。"""

    return lam * h_val + 0.5 * mu * (h_val ** 2)


# ---------------------- 归一化辅助函数 ----------------------
N_RES_DIM = 4  # cpu_num, gpu_mem, ram, disk 四维


def evaluation_func(chromosome: list,
                    agent_instances: list,
                    agent_lookup: Dict[int, AgentTemplate],
                    device_map: Dict[int, Device],
                    gw_matrix: np.ndarray,
                    node_map: Dict[Tuple[int, str], Tuple[int, int]],
                    task_graphs: List[TaskGraph],
                    alpha: float = 0.5,  # 时间权重
                    beta: float = 0.5,   # 能耗权重
                    return_detail: bool = False,
                    _placement_override: Optional[Dict] = None,
                    exec_time_cache: Optional[Dict[Tuple[int, int], float]] = None):
    """
    使用 SAC 中的 ``evaluation_func_rl`` 评估个体，保持原有返回格式。

    因 SAC 的评估函数返回 (makespan, energy, detail)，此处将 detail 作为能耗细分，
    其他惩罚和约束项维持为 0 以匹配遗传算法的接口。
    """
    if _placement_override is not None:
        placement = _placement_override
    else:
        placement = decode_global_placement(chromosome, agent_instances, agent_lookup)

    total_makespan = 0.0
    total_energy = 0.0
    total_penalty = 0.0
    res_violation = 0.0
    cap_violation = 0.0
    bw_violation = 0.0
    lat_violation = 0.0
    detail_map: Dict[int, Dict[str, float]] = {}

    placement = attach_nodes_to_placement(placement, node_map)

    def get_device_for_node(node_id: str, task_graph: TaskGraph) -> Optional[int]:
        # 与 SAC.evaluation_func_rl 内部逻辑保持一致
        module_id = node_map.get((task_id, str(node_id)), (None, None))[1]
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

    @lru_cache(maxsize=200000)
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

    for idx, task_graph in enumerate(task_graphs, start=1):
        task_placement = {k: v for k, v in placement.items() if k[0] == idx}
        task_id = idx
        if not task_placement:
            continue
        mk, energy, detail = evaluation_func_rl(
            task_placement,
            task_graph,
            agent_lookup,
            device_map,
            gw_matrix,
            exec_time_cache=exec_time_cache,
        )
        total_makespan += mk
        total_energy += energy
        if isinstance(detail, dict):
            detail_map[idx] = detail

        # -------- 约束违反统计：增广拉格朗日 --------
        # 1) 资源约束：需求/供给超额部分按比例累计
        for (tid, module_id), info in task_placement.items():
            agent_tpl = agent_lookup[info["agent_id"]]
            dev = device_map[info["soft_device"]]
            demand = agent_tpl.r
            supply = (
                float(getattr(dev.resource, "cpu", 0.0)),
                float(getattr(dev.resource, "num", 0.0)),
                float(getattr(dev.resource, "gpu", 0.0)),
                float(getattr(dev.resource, "gpu_mem", 0.0)),
                float(getattr(dev.resource, "ram", 0.0)),
                float(getattr(dev.resource, "disk", 0.0)),
            )
            for req, cap in zip(demand, supply):
                if cap <= 0:
                    res_violation += 1.0
                else:
                    res_violation += max((req - cap) / cap, 0.0)

            # 2) 能力约束：所需能力设备缺失计入违反
            missing_soft = agent_tpl.C_soft - dev.soft_cap
            cap_violation += len(missing_soft)
            for sense_cap, dev_id in info["sense_map"].items():
                sense_dev = device_map.get(dev_id)
                if sense_dev is None or sense_cap not in sense_dev.sense_cap:
                    cap_violation += 1.0
            for act_cap, dev_id in info["act_map"].items():
                act_dev = device_map.get(dev_id)
                if act_dev is None or act_cap not in act_dev.act_cap:
                    cap_violation += 1.0

        # 3) 带宽/时延约束：遍历任务图边
        for u, v, attr in task_graph.get_dependencies():
            du = get_device_for_node(u, task_graph)
            dv = get_device_for_node(v, task_graph)
            if du is None or dv is None or du == dv:
                continue

            data_mb = attr.get("data_size", 0.0)
            bw_req = attr.get("bandwidth_req", 0.0)
            lat_req = attr.get("latency_req", 0.0)

            delay_ms, rate_mbps, _ = cached_transmission_delay(du, dv, data_mb, bw_req)

            if bw_req > 0:
                bw_violation += max((bw_req - rate_mbps) / max(bw_req, 1e-6), 0.0)
            if lat_req > 0:
                lat_violation += max((delay_ms - lat_req) / max(lat_req, 1e-6), 0.0)

    total_penalty += _augmented_penalty(res_violation, AL_PARAMS["resource"]["lam"], AL_PARAMS["resource"]["mu"])
    total_penalty += _augmented_penalty(cap_violation, AL_PARAMS["capability"]["lam"], AL_PARAMS["capability"]["mu"])
    total_penalty += _augmented_penalty(bw_violation, AL_PARAMS["bandwidth"]["lam"], AL_PARAMS["bandwidth"]["mu"])
    total_penalty += _augmented_penalty(lat_violation, AL_PARAMS["latency"]["lam"], AL_PARAMS["latency"]["mu"])

    if return_detail:
        metrics = (
            total_makespan,
            total_energy,
            total_penalty,
            res_violation,
            cap_violation,
            bw_violation,
            lat_violation,
        )
        return metrics, {"details": detail_map}

    return (
        total_makespan,
        total_energy,
        total_penalty,
        res_violation,
        cap_violation,
        bw_violation,
        lat_violation,
    )



WHITE_NOISE = 1e-9  # W

# ---------- 常量 ----------
P_TX_IOT = 0.10  # W   100 mW
P_RX_IOT = 0.10
P_TX_AP = 0.025  # W    25 mW
P_RX_AP = 0.025

h_e, h_c = 3, 5
E_edge, E_core = 37e-9, 12.6e-9  # J/bit
E_cent = 20e-6  # J/bit
E_per_bit_wired = h_e * E_edge + h_c * E_core + E_cent  # ≈2.075e‑5 J/bit

# <<< NEW: 辅助函数，正确计算传输时间 >>>
def _get_transmission_time_s(data_size_mb: float, rate_mbps: float) -> float:
    """计算数据传输所需的时间（秒），处理速率为0的情况。"""
    if rate_mbps <= 0:
        return float('inf') # 如果速率为0，传输时间为无穷大
    return (data_size_mb * 8) / rate_mbps


class ABGSK:
    def __init__(self, pop_size=None, max_g=None, p=0.05):
        self.pop_size = pop_size
        self.max_g = max_g
        self.p = p

        # ------------- 基础环境加载 -------------
        cloud, device_list, edge_list, gw_matrix = load_infrastructure()
        self.cloud = cloud
        self.device_list = device_list
        self.edge_list = edge_list
        self.gw_matrix = gw_matrix

        # 读取智能体模板
        df = pd.read_csv("redundant_agent_templates.csv")

        # 任务图 dag1~dag10
        self.task_graphs = []
        task_num = 10
        for dag in range(1, task_num + 1):
            path = f"task/dag{dag}_typed_constraint.json"
            with open(path, "r") as f:
                data = json.load(f)
            tg = TaskGraph()
            tg.load_from_json(data)
            self.task_graphs.append(tg)

        device_num = len(device_list)
        edge_num = len(edge_list)

        # 为 Edge 设备重新编号（接在 IoT 之后）
        for idx, edge in enumerate(edge_list, start=device_num + 1):
            edge.id = idx
        cloud.id = device_num + edge_num + 1  # Cloud 单独一个 id

        # 构建实例与模板
        self.instances, self.node_map = build_agent_instances()
        self.agent_lookup = build_agent_lookup(df)
        self.all_devices = list(range(1, cloud.id + 1))

        # device_id -> Device
        self.device_map = {d.id: d for d in device_list}
        self.device_map.update({e.id: e for e in edge_list})
        self.device_map[cloud.id] = cloud

        # 预计算 (agent_id, device_id) 的核心执行时间，避免评估时重复算
        self.exec_time_cache: Dict[Tuple[int, int], float] = {}
        for agent_id, tpl in self.agent_lookup.items():
            r_cpu, _, r_gpu, _, _, _ = tpl.r
            for dev_id, dev in self.device_map.items():
                res = dev.resource
                cpu_cap = float(getattr(res, "cpu", 0.0))
                gpu_cap = float(getattr(res, "gpu", 0.0))
                cpu_time = (r_cpu / cpu_cap) if cpu_cap > 0 else float('inf')
                gpu_time = (r_gpu / gpu_cap) if gpu_cap > 0 else 0.0
                self.exec_time_cache[(agent_id, dev_id)] = max(cpu_time, gpu_time)

        # ---- 构建基因边界（low/high）----
        low_list = []
        high_list = []
        for _, _, aid in self.instances:  # 顺序必须和 encode_global_placement 一致
            tpl = self.agent_lookup[aid]

            # 1) soft：1～cloud.id
            low_list.append(1)
            high_list.append(self.cloud.id)

            # 2) sense：1～device_num（只 IoT）
            for _ in tpl.C_sense:
                low_list.append(1)
                high_list.append(len(self.device_list))

            # 3) act：1～device_num（只 IoT）
            for _ in tpl.C_act:
                low_list.append(1)
                high_list.append(len(self.device_list))

        self.low = np.array(low_list, dtype=int)
        self.high = np.array(high_list, dtype=int)
        self.problem_size = len(self.low)
        assert self.low.shape == self.high.shape

        self.alpha = 0.5
        self.beta = 0.5

        # ---- 贪心 seed 染色体 ----
        devices = device_list + edge_list + [cloud]
        self.seed_chrom = encode_global_placement_greedy(
            self.instances, self.agent_lookup, devices, self.device_map, verbose=False
        )

        # 打印一些调试信息
        print(f"Device boundaries - IoT: 1-{device_num}, Edge: {device_num + 1}-{device_num + edge_num}, Cloud: {cloud.id}")
        print(f"problem_size={self.problem_size}, low[:10]={self.low[:10]}, high[:10]={self.high[:10]}")

        mp, energy, penalty, res_sum, cap_sum, bw_sum, lat_sum = evaluation_func(
            self.seed_chrom,
            self.instances,
            self.agent_lookup,
            self.device_map,
            self.gw_matrix,
            self.node_map,
            self.task_graphs,
            alpha=self.alpha,
            beta=self.beta,
            exec_time_cache=self.exec_time_cache,
        )
        fval = self.alpha * mp + self.beta * energy + penalty
        print(f"[SEED] fitness {fval:.3f} | makespan {mp:.3f} | energy {energy:.3f} | "
              f"penalty {penalty:.3f} | res {res_sum:.3f} | cap {cap_sum:.3f} | "
              f"bw {bw_sum:.3f} | lat {lat_sum:.3f}")

    # ---------- 批量评估 ----------
    def evaluation_func(self, pop: np.ndarray):
        """
        输入:
            pop: (N, problem_size) int ndarray，整个种群
        输出:
            fval: (N,) 适应度 (越小越好)
            以及各个指标的向量
        """
        N = pop.shape[0]
        fval = np.zeros(N)
        makespan = np.zeros(N)
        total_energy = np.zeros(N)
        L_penalty = np.zeros(N)
        h_res_sum = np.zeros(N)
        h_cap_sum = np.zeros(N)
        h_bw_sum = np.zeros(N)
        h_lat_sum = np.zeros(N)

        for i in range(N):
            chrom = pop[i]
            mp, energy, penalty, res_val, cap_val, bw_val, lat_val = evaluation_func(
                chrom,
                self.instances,
                self.agent_lookup,
                self.device_map,
                self.gw_matrix,
                self.node_map,
                self.task_graphs,
                alpha=self.alpha,
                beta=self.beta,
                exec_time_cache=self.exec_time_cache,
            )
            makespan[i] = mp
            total_energy[i] = energy
            L_penalty[i] = penalty
            h_res_sum[i] = res_val
            h_cap_sum[i] = cap_val
            h_bw_sum[i] = bw_val
            h_lat_sum[i] = lat_val
            fval[i] = self.alpha * mp + self.beta * energy + penalty

        return fval, makespan, total_energy, L_penalty, h_res_sum, h_cap_sum, h_bw_sum, h_lat_sum

    def _update_al_multipliers(self, res_violation: float, cap_violation: float, bw_violation: float, lat_violation: float):
        """根据当前种群的平均违反度自适应更新增广拉格朗日乘子。"""

        tol = 1e-4
        rho = 1.5
        stats = {
            "resource": res_violation,
            "capability": cap_violation,
            "bandwidth": bw_violation,
            "latency": lat_violation,
        }
        for key, h in stats.items():
            AL_PARAMS[key]["lam"] = max(0.0, AL_PARAMS[key]["lam"] + AL_PARAMS[key]["mu"] * h)
            if h > tol:
                AL_PARAMS[key]["mu"] = min(AL_PARAMS[key]["mu"] * rho, 1e12)

    # ---------- sense/act 约束检查（调试用） ----------
    def check_sense_act_constraints(self, chromosome):
        placement = decode_global_placement(chromosome, self.instances, self.agent_lookup)
        violations = []
        device_num = len(self.device_list)

        for (task_id, module_id), info in placement.items():
            for sense_cap, dev_id in info["sense_map"].items():
                if dev_id > device_num:
                    violations.append(
                        f"[sense] Task {task_id}, Module {module_id}: cap {sense_cap} -> dev {dev_id} (> {device_num})"
                    )
            for act_cap, dev_id in info["act_map"].items():
                if dev_id > device_num:
                    violations.append(
                        f"[act]   Task {task_id}, Module {module_id}: cap {act_cap} -> dev {dev_id} (> {device_num})"
                    )
        return violations

    # ---------- 训练主循环 ----------
    def train(self):
        # ---- 基本参数 ----
        if self.pop_size is None:
            self.pop_size = 40 * self.problem_size if self.problem_size > 5 else 100
        max_nfes = 1000 * self.problem_size
        if self.max_g is None:
            self.max_g = max_nfes  # 和原论文一致

        KF_pool = np.array([0.1, 1.0, 0.5, 0.8])
        KR_pool = np.array([0.02, 0.1, 0.9, 0.9])
        max_pop_size = self.pop_size
        min_pop_size = 12

        # ---- helper: 轻量随机扰动 ----
        def mutate(arr: np.ndarray, p=0.05, step: int = 3) -> np.ndarray:
            new = arr.copy()
            device_num = len(self.device_list)
            for i in range(new.size):
                if random.random() < p:
                    lo = max(self.low[i], int(new[i] - step))
                    hi = min(self.high[i], int(new[i] + step))
                    new[i] = random.randint(lo, hi)
            return new

        # ---- 初始化种群：seed_chrom + 扰动 ----
        seed_np = np.array(self.seed_chrom, dtype=np.int32)
        popold = np.stack([seed_np.copy() for _ in range(self.pop_size)], axis=0)
        for i in range(1, self.pop_size):
            popold[i] = mutate(popold[i])
        pop = popold.copy()

        best_indv = [popold[0].copy()]
        loss = []

        fitness, _, _, _, _, _, _, _ = self.evaluation_func(pop)
        nfes = 0
        bsf_fit_var = 1e+300
        bsf_solution = popold[0].copy()
        no_improve_gen = 0

        for i in range(self.pop_size):
            nfes += 1
            if nfes > max_nfes:
                break
            if fitness[i] < bsf_fit_var:
                bsf_fit_var = fitness[i]

        # ---- GSK 参数 ----
        K = np.full((self.pop_size, 1), 10.0, dtype=float)
        Kind = np.random.rand(self.pop_size, 1)
        K[Kind < 0.5] = np.random.rand((Kind < 0.5).sum())
        K[Kind >= 0.5] = np.ceil(np.random.rand((Kind >= 0.5).sum()) * 20)

        KW_ind = None
        All_Imp = np.zeros(4)
        KF = None
        KR = None

        g = 0
        while nfes < max_nfes and g < self.max_g:
            g += 1
            if nfes < 0.1 * max_nfes:
                KW_ind = np.array([0.75, 0.10, 0.10, 0.05])
            else:
                KW_ind = 0.95 * KW_ind + 0.05 * All_Imp
                KW_ind = KW_ind / np.sum(KW_ind)

            # 选择哪一种 KF / KR 组合
            K_rand_ind = np.random.rand(self.pop_size, 1)
            K_rand_ind[(K_rand_ind > sum(KW_ind[0:3])) & (K_rand_ind <= sum(KW_ind[0:4]))] = 3
            K_rand_ind[(K_rand_ind > sum(KW_ind[0:2])) & (K_rand_ind <= sum(KW_ind[0:3]))] = 2
            K_rand_ind[(K_rand_ind > sum(KW_ind[0:1])) & (K_rand_ind <= sum(KW_ind[0:2]))] = 1
            K_rand_ind[(K_rand_ind > 0) & (K_rand_ind <= KW_ind[0])] = 0
            K_rand_ind = K_rand_ind.astype(np.int16)

            KF = KF_pool[K_rand_ind]
            KR = KR_pool[K_rand_ind]

            D_Gained_Shared_Junior = np.ceil(self.problem_size * ((1 - nfes / max_nfes) ** K))

            pop = popold  # 引用

            indBest = np.argsort(fitness)
            Rg1, Rg2, Rg3 = self.Gained_Shared_Junior_R1R2R3(indBest)
            R1, R2, R3 = self.Gained_Shared_Senior_R1R2R3(indBest, p=0.05)

            R0 = range(self.pop_size)
            Gained_Shared_Junior = np.zeros((self.pop_size, self.problem_size))
            ind1 = fitness[R0] > fitness[Rg3]
            if np.sum(ind1) > 0:
                Gained_Shared_Junior[ind1, :] = pop[ind1, :] + KF[ind1] * (
                        pop[Rg1[ind1], :] - pop[Rg2[ind1], :] +
                        pop[Rg3[ind1], :] - pop[ind1, :]
                )
            ind1 = ~ind1
            if np.sum(ind1) > 0:
                Gained_Shared_Junior[ind1, :] = pop[ind1, :] + KF[ind1] * (
                        pop[Rg1[ind1], :] - pop[Rg2[ind1], :] +
                        pop[ind1, :] - pop[Rg3[ind1], :]
                )

            Gained_Shared_Senior = np.zeros((self.pop_size, self.problem_size))
            ind = fitness[R0] > fitness[R2]
            if np.sum(ind) > 0:
                Gained_Shared_Senior[ind, :] = pop[ind, :] + KF[ind] * (
                        pop[R1[ind], :] - pop[ind, :] +
                        pop[R2[ind], :] - pop[R3[ind], :]
                )
            ind = ~ind
            if np.sum(ind) > 0:
                Gained_Shared_Senior[ind, :] = pop[ind, :] + KF[ind] * (
                        pop[R1[ind], :] - pop[R2[ind], :] +
                        pop[ind, :] - pop[R3[ind], :]
                )

            self.boundConstraint(Gained_Shared_Junior, pop, self.low, self.high)
            self.boundConstraint(Gained_Shared_Senior, pop, self.low, self.high)

            D_mask_junior = np.random.rand(self.pop_size, self.problem_size) <= (
                    D_Gained_Shared_Junior[:] / self.problem_size
            )
            D_mask_senior = ~D_mask_junior

            rand_mask_junior = np.random.rand(self.pop_size, self.problem_size) <= KR
            rand_mask_senior = np.random.rand(self.pop_size, self.problem_size) <= KR

            D_mask_junior &= rand_mask_junior
            D_mask_senior &= rand_mask_senior

            ui = pop.copy()
            ui[D_mask_junior] = Gained_Shared_Junior[D_mask_junior]
            ui[D_mask_senior] = Gained_Shared_Senior[D_mask_senior]

            self.boundConstraint(ui, pop, self.low, self.high)

            # 每 100 代可选做一次约束检查
            if g % 100 == 0:
                viol = self.check_sense_act_constraints(ui[0])
                if viol:
                    print(f"[Gen {g}] sample constraint violations (first 3):")
                    for v in viol[:3]:
                        print("   ", v)

            child_fit, child_ms, child_energy, child_pen, child_res, child_cap, child_bw, child_lat = self.evaluation_func(ui)

            # 基于当前种群平均违反度自适应调整增广拉格朗日参数
            self._update_al_multipliers(
                float(np.mean(child_res)),
                float(np.mean(child_cap)),
                float(np.mean(child_bw)),
                float(np.mean(child_lat)),
            )

            for i in range(self.pop_size):
                nfes += 1
                if nfes > max_nfes:
                    break
                if child_fit[i] < bsf_fit_var:
                    bsf_fit_var = child_fit[i]
                    bsf_solution = ui[i, :].copy()
                    no_improve_gen = 0
                    print(f"[Gen {g}] best_f={bsf_fit_var:.3f} | "
                          f"ms={child_ms[i]:.3f} | E={child_energy[i]:.3f} | "
                          f"pen={child_pen[i]:.3f} | res={child_res[i]:.3f} | "
                          f"cap={child_cap[i]:.3f} | bw={child_bw[i]:.3f} | lat={child_lat[i]:.3f}")
                else:
                    no_improve_gen += 1

            # 统计各策略贡献
            dif = np.abs(fitness - child_fit)
            child_better = (fitness > child_fit)
            All_Imp = np.zeros(4)
            for i_case in range(4):
                mask = child_better & (K_rand_ind.reshape(-1) == i_case)
                if np.sum(mask) > 0:
                    All_Imp[i_case] = np.sum(dif[mask])
            if np.sum(All_Imp) != 0:
                All_Imp = All_Imp / np.sum(All_Imp)
                Imp_Ind = np.argsort(All_Imp)
                for ii in range(len(All_Imp) - 1):
                    All_Imp[Imp_Ind[ii]] = max(All_Imp[Imp_Ind[ii]], 0.05)
                All_Imp[Imp_Ind[-1]] = 1 - np.sum(All_Imp[Imp_Ind[0:-1]])
            else:
                All_Imp[:] = 1.0 / len(All_Imp)

            conc = np.concatenate((fitness.reshape(-1, 1), child_fit.reshape(-1, 1)), axis=1)
            choose_child = conc.argmin(axis=1)  # 0: parent, 1: child
            fitness = conc[np.arange(conc.shape[0]), choose_child]
            popold = pop.copy()
            popold[choose_child == 1, :] = ui[choose_child == 1, :]

            best_indv.append(bsf_solution.copy())
            loss.append(bsf_fit_var)

            if g % 50 == 0 and no_improve_gen >= 50:
                print(f"[EARLY STOP] no improvement for {no_improve_gen} generations.")
                break

            # 自适应缩减种群规模
            plan_pop_size = round(
                (min_pop_size - max_pop_size) * ((nfes / max_nfes) ** (1 - nfes / max_nfes)) + max_pop_size
            )
            if self.pop_size > plan_pop_size:
                reduction_ind_num = self.pop_size - plan_pop_size
                if self.pop_size - reduction_ind_num < min_pop_size:
                    reduction_ind_num = self.pop_size - min_pop_size
                self.pop_size -= reduction_ind_num
                for _ in range(reduction_ind_num):
                    indBest = np.argsort(fitness)
                    worst_ind = indBest[-1]
                    popold = np.delete(popold, worst_ind, axis=0)
                    pop = np.delete(pop, worst_ind, axis=0)
                    fitness = np.delete(fitness, worst_ind, axis=0)
                    K = np.delete(K, worst_ind, axis=0)

        return best_indv, bsf_fit_var, loss

    # ---------- 边界裁剪 ----------
    def boundConstraint(self, vi, pop, low, high):
        vi_clipped = np.clip(vi, low[None, :], high[None, :])
        vi[:] = np.round(vi_clipped).astype(np.int32)

    # ---------- GSK 辅助 ----------
    def Gained_Shared_Junior_R1R2R3(self, indBest=None):
        pop_size = len(indBest)
        R1 = np.zeros(pop_size, dtype='int32')
        R2 = np.zeros(pop_size, dtype='int32')

        tmp = indBest.copy()
        tmp[1:] = tmp[0:-1]
        tmp[0] = indBest[1]
        tmp[-1] = indBest[pop_size - 3]
        R1[indBest] = tmp

        tmp = indBest.copy()
        tmp[0:-1] = tmp[1:]
        tmp[0] = indBest[2]
        tmp[-1] = indBest[pop_size - 2]
        R2[indBest] = tmp

        choice = np.arange(pop_size)
        R3 = [np.random.choice(choice[(R1[i] != choice) &
                                      (R2[i] != choice) &
                                      (indBest[i] != choice)]) for i in range(pop_size)]
        return R1, R2, np.array(R3, dtype='int32')

    def Gained_Shared_Senior_R1R2R3(self, indBest=None, p=0.05):
        pop_size = len(indBest)
        R1 = [indBest[random.randint(0, max(0, round(pop_size * p) - 1))] for _ in range(pop_size)]
        R2 = [indBest[random.randint(round(pop_size * p), max(0, round(pop_size * (1 - p)) - 1))] for _ in range(pop_size)]
        R3 = [indBest[random.randint(max(0, round(pop_size * (1 - p))), pop_size - 1)] for _ in range(pop_size)]
        return np.array(R1, dtype='int32'), np.array(R2, dtype='int32'), np.array(R3, dtype='int32')




def encode_global_placement_greedy(
        agent_instances: List[Tuple[int, int, int]],
        agent_lookup: Dict[int, AgentTemplate],
        devices: List[Device],
        device_map: Dict[int, Device],
        verbose: bool = False,
) -> List[int]:
    id_set = {d.id for d in devices}
    cloud_id = max(id_set)
    device_number = sum(getattr(d, "type", "Device") == "Device" for d in devices)

    remaining = {
        d.id: [
            d.resource.cpu,
            d.resource.num,
            d.resource.gpu,
            d.resource.gpu_mem,
            d.resource.ram,
            d.resource.disk,
        ]
        for d in device_map.values()
    }

    chrom: List[int] = []
    CHECK_IDX = (1, 3, 4, 5)

    def fits(need, remain):
        return all(need[i] <= remain[i] for i in CHECK_IDX)

    def deduct(need, remain):
        for i in range(len(remain)):
            remain[i] -= need[i]

    for (task_id, module_id, agent_id) in agent_instances:
        tpl = agent_lookup[agent_id]
        need = tpl.r

        cand_soft = [
            d for d in devices
            if tpl.C_soft.issubset(d.soft_cap) and fits(need, remaining[d.id])
        ]
        cand_soft.sort(key=lambda d: remaining[d.id][0], reverse=True)
        soft_dev = cand_soft[0].id if cand_soft else cloud_id
        chrom.append(soft_dev)
        deduct(need, remaining[soft_dev])

        if verbose:
            print(f"[SOFT] Task {task_id}-M{module_id} Agent {agent_id} -> Dev {soft_dev}")

        for τ in tpl.C_sense:
            cand_sense = [d for d in devices if d.id <= device_number and (τ in d.sense_cap)]
            sense_dev = cand_sense[0].id if cand_sense else soft_dev
            chrom.append(sense_dev)
            if verbose:
                print(f"   [SENSE {τ}] -> Dev {sense_dev}")

        for τ in tpl.C_act:
            cand_act = [d for d in devices if d.id <= device_number and (τ in d.act_cap)]
            act_dev = cand_act[0].id if cand_act else soft_dev
            chrom.append(act_dev)
            if verbose:
                print(f"   [ACT {τ}] -> Dev {act_dev}")

    return chrom



if __name__ == "__main__":
    abgsk = ABGSK(pop_size=100, max_g=300000)
    aa = abgsk.train()
    print(1)
    pass