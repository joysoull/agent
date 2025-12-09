import ast
import copy
import json
import os
import random
from typing import List, Tuple, Dict, Optional
import math
import numpy as np
import pandas as pd
from simulation import Device, Resource, near_wireless_rate, near_wireless_delay, far_wireless_rate, far_wireless_delay, \
    edge_inter_delay, cloud_edge_delay, device_number

from partiation import AgentTemplate, parse_capability_field
import networkx as nx
from collections import defaultdict, deque


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
      instances         : [(task_id, module_id, agent_id), ...]
      module_node_map   : {(task_id, node_id): (task_id, module_id)}

    支持两种 Stage-1 输出格式：
    1) GRASP/LNS：partiation/agents_{task_id}.json，每个文件包含模块列表。
    2) CP-SAT：plans_cpsat.json，顶层列表，每个元素带 task_id + modules。
    """
    instances: List[Tuple[int, int, int]] = []
    node_map: Dict[Tuple[int, str], Tuple[int, int]] = {}

    if os.path.exists(cpsat_path):
        with open(cpsat_path, "r") as f:
            plans = json.load(f)

        for idx, plan in enumerate(plans, start=1):
            task_id = plan.get("task_id", idx)
            for module in plan.get("modules", []):
                module_id, agent_id, nodes = _normalize_module_entry(module)
                if module_id is None or agent_id is None:
                    continue
                instances.append((task_id, int(module_id), int(agent_id)))
                for node in nodes:
                    node_map[(task_id, node)] = (task_id, int(module_id))
        return instances, node_map

    for task_id in range(1, count + 1):
        path = os.path.join(folder, f"{prefix}{task_id}.json")
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            modules = json.load(f)
        for module in modules:
            module_id, agent_id, nodes = _normalize_module_entry(module)
            if module_id is None or agent_id is None:
                continue
            instances.append((task_id, int(module_id), int(agent_id)))

            # 建立 node -> module 的映射
            for node in nodes:
                node_map[(task_id, node)] = (task_id, int(module_id))
    return instances, node_map


def find_module_of_node(node: str,
                        task_id: int,
                        module_node_map: dict):
    """
    返回 (task_id, module_id) 或 None
    """
    return module_node_map.get((task_id, node))


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


def encode_global_placement(agent_instances: List[Tuple[int, int, int]],
                            agent_lookup: Dict[int, AgentTemplate],
                            all_devices: List[int]) -> List[int]:
    """
    返回整数列表:
      [ soft₀ , sense₀₁ , … , act₀₁ , … ,  soft₁ , sense₁₁ , … ]
    """
    chromosome = []
    for (_, _, agent_id) in agent_instances:
        tpl = agent_lookup[agent_id]

        # soft (整体部署到一个设备)
        chromosome.append(random.choice(all_devices))

        # 每个 sense 能力一个设备
        for _ in tpl.C_sense:
            chromosome.append(random.choice(all_devices[1:device_number]))

        # 每个 act 能力一个设备
        for _ in tpl.C_act:
            chromosome.append(random.choice(all_devices[1:device_number]))
    return chromosome


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


def main():
    cloud, device_list, edge_list, gw_matrix = load_infrastructure()

    # 读取智能体模板
    # 加载数据
    csv_path = "redundant_agent_templates.csv"
    df = pd.read_csv(csv_path)

    task_graphs = []
    task_num = 10  # 假设有 10 个任务图 dag1~dag10
    tg = TaskGraph()
    for dag in range(1, task_num + 1):
        path = f"task/dag{dag}_typed_constraint.json"
        with open(path, "r") as f:
            data = json.load(f)
            tg.load_from_json(data)
            task_graphs.append(tg)

    device_num = len(device_list)
    edge_num = len(edge_list)

    # 给 Edge 服务器分配连续 id
    for idx, edge in enumerate(edge_list, start=device_num + 1):
        edge.id = idx  # 例如 IoT 有 25 台，则第一台 Edge 从 26 开始

    # Cloud 服务器 id
    cloud.id = device_num + edge_num + 1  # 唯一编号

    # 构建实例与模板
    instances, node_map = build_agent_instances()
    agent_lookup = build_agent_lookup(df)
    all_devices = list(range(1, cloud.id + 1))

    device_map = {d.id: d for d in device_list}
    device_map.update({e.id: e for e in edge_list})
    device_map[cloud.id] = cloud

    # 2. 随机生成一个染色体（初始化个体）
    chrom = encode_global_placement(instances, agent_lookup, all_devices)
    # 3. 解码查看放置方案
    placement_dict = decode_global_placement(chrom, instances, agent_lookup)

    # 打印其中一个实例的放置详情
    sample_key = instances[0][:2]  # (task_id,module_id)
    print(sample_key, placement_dict[sample_key])


# ---------------------- 归一化辅助函数 ----------------------
N_RES_DIM = 4  # cpu_num, gpu_mem, ram, disk 四维


def norm_latency(actual, req):
    return max(0.0, actual - req) / req  # ∈[0,∞) 但一般<1


def norm_bandwidth(actual, req):
    return max(0.0, req - actual) / req


def norm_resource(violation_cnt):
    return violation_cnt / N_RES_DIM  # 最多=1


def norm_capability(missing, total):
    return 0.0 if total == 0 else missing / total


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
                    _placement_override: Optional[Dict] = None):
    """
    返回:
      - 默认: makespan, energy, L_penalty, h_res_sum, h_cap_sum, h_bw_sum, h_lat_sum
      - return_detail=True: (metrics, violations)
    """
    # ---- 0. 解码/覆写 placement ----
    if _placement_override is not None:
        placement = _placement_override
        # 用 placement 重建 node_map（健壮性）
        node_map = {}
        for (task_id, module_id), mod_info in placement.items():
            for node in mod_info.get("nodes", []):
                node_map[(task_id, node)] = (task_id, module_id)
    else:
        placement = decode_global_placement(chromosome, agent_instances, agent_lookup)

    resource_used = defaultdict(lambda: [0, 0, 0, 0, 0, 0])  # cpu, num, gpu, gpu_mem, ram, disk
    exec_time_map: Dict[Tuple[int, int], float] = {}
    edge_delay_map: Dict[Tuple[int, int, int], float] = {}

    total_exec_energy = 0.0
    comm_energy = 0.0
    makespan = 0.0

    # AL 相关
    h_lat_sum = 0.0
    h_bw_sum = 0.0
    h_res_sum = 0.0
    h_cap_sum = 0.0

    used_sense_on_dev = defaultdict(set)
    used_act_on_dev = defaultdict(set)

    # ---------- 1. 每个模块的执行时间 / 能力 / 资源 ----------
    for (task_id, module_id), info in placement.items():
        agent = agent_lookup[info["agent_id"]]
        soft_dev_id = info["soft_device"]
        dev_soft = device_map[soft_dev_id]
        need = agent.r

        # 能力缺失
        miss_soft = len([τ for τ in agent.C_soft if τ not in dev_soft.soft_cap])
        miss_sense = 0
        miss_act = 0
        dup_cnt = 0

        # sense
        for t, dev_id in info["sense_map"].items():
            miss_sense += t not in device_map[dev_id].sense_cap
            dup_cnt += t in used_sense_on_dev[dev_id]
            used_sense_on_dev[dev_id].add(t)

        # act
        for t, dev_id in info["act_map"].items():
            miss_act += t not in device_map[dev_id].act_cap
            dup_cnt += t in used_act_on_dev[dev_id]
            used_act_on_dev[dev_id].add(t)

        total_cap = len(agent.C_soft) + len(agent.C_sense) + len(agent.C_act)
        miss_total = miss_soft + miss_sense + miss_act + dup_cnt
        h_cap_sum += norm_capability(miss_total, total_cap)

        # 资源累加（只对 num/gpu_mem/ram/disk 计约束）
        r_cpu, r_num, r_gpu, r_gpu_mem, r_ram, r_disk = need
        res_list = resource_used[soft_dev_id]
        res_list[1] += r_num
        res_list[3] += r_gpu_mem
        res_list[4] += r_ram
        res_list[5] += r_disk

        # 执行时间
        cpu_time = r_cpu / dev_soft.resource.cpu
        gpu_time = (r_gpu / dev_soft.resource.gpu) if dev_soft.resource.gpu else 0.0
        T_ma = max(cpu_time, gpu_time)
        exec_time_map[(task_id, module_id)] = T_ma

        # 计算能耗（简单线性）
        E_cpu = cpu_time * dev_soft.resource.cpu_power / 3600
        E_gpu = gpu_time * dev_soft.resource.gpu_power / 3600 if dev_soft.resource.gpu else 0.0
        total_exec_energy += (E_cpu + E_gpu)

    # ---------- 2. 资源约束 ----------
    for dev_id, used in resource_used.items():
        cap = device_map[dev_id].resource
        violation_cnt = sum([
            used[1] > cap.num,
            used[3] > cap.gpu_mem,
            used[4] > cap.ram,
            used[5] > cap.disk,
        ])
        h_res_sum += norm_resource(violation_cnt)

    # ---------- 3. 每个任务的通信 & makespan ----------
    agent_dag_edges = defaultdict(list)

    # **关键：task_id 从 1 开始，对齐 agents_{task_id}.json**
    for task_id, tg in enumerate(task_graphs, start=1):
        for u, v, attr in tg.get_dependencies():
            u_type = tg.G.nodes[u].get("type", "unknown")
            v_type = tg.G.nodes[v].get("type", "unknown")

            mod_u = find_module_of_node(u, task_id, node_map)
            mod_v = find_module_of_node(v, task_id, node_map)
            if mod_u is None or mod_v is None:
                # 说明该节点没有模块记录，直接跳过这条边（理论上不应发生）
                continue
            _, u_module = mod_u
            _, v_module = mod_v

            # 解析节点部署设备
            def get_dev_for_node(node_id, node_type, module_id):
                mod_place = placement[(task_id, module_id)]
                if node_type == "proc":
                    return mod_place["soft_device"]
                idx_val = tg.G.nodes[node_id].get("idx")
                if node_type == "sense":
                    return mod_place["sense_map"].get(idx_val)
                if node_type == "act":
                    return mod_place["act_map"].get(idx_val)
                return None

            u_dev = get_dev_for_node(u, u_type, u_module)
            v_dev = get_dev_for_node(v, v_type, v_module)

            edge_delay_ms = 0.0
            bw_actual = 0.0

            # proc→proc 同模块：无通信
            if u_type == "proc" and v_type == "proc":
                if u_module != v_module:
                    agent_dag_edges[task_id].append((u_module, v_module, attr))
            else:
                if u_dev is not None and v_dev is not None:
                    edge_delay_ms, bw_actual, e_comm_j = compute_transmission_delay(
                        src=device_map[u_dev],
                        dst=device_map[v_dev],
                        data_size_mb=attr["data_size"],
                        bw_req=attr["bandwidth_req"],
                        gw_matrix=gw_matrix,
                        edge_inter_delay=edge_inter_delay,
                        cloud_edge_delay=cloud_edge_delay
                    )
                    comm_energy += e_comm_j

                    # 为 agent-level DAG 记录边
                    if u_type == "sense" and v_type == "proc":
                        src_virtual = u
                        agent_dag_edges[task_id].append((src_virtual, v_module, attr))
                        edge_delay_map[(task_id, src_virtual, v_module)] = edge_delay_ms
                    elif u_type == "proc" and v_type == "act":
                        dst_virtual = v
                        agent_dag_edges[task_id].append((u_module, dst_virtual, attr))
                        edge_delay_map[(task_id, u_module, dst_virtual)] = edge_delay_ms
                    else:
                        agent_dag_edges[task_id].append((u_module, v_module, attr))
                        edge_delay_map[(task_id, u_module, v_module)] = edge_delay_ms

            # 时延违约
            h_lat_sum += norm_latency(edge_delay_ms, attr["latency_req"])
            # 带宽违约（需要的 > 实际可用）
            h_bw_sum += norm_bandwidth(attr["bandwidth_req"] - bw_actual, attr["bandwidth_req"])

        # 该任务的完成时间
        T_m = compute_task_finish_time(
            task_id=task_id,
            agent_dag_edges=agent_dag_edges[task_id],
            exec_time_map=exec_time_map,
            edge_delay_map=edge_delay_map
        )
        makespan = max(makespan, T_m)

    # ---------- 4. 增广拉格朗日惩罚 ----------
    L_penalty = 0.0
    for name, h_sum in [
        ("latency", h_lat_sum),
        ("bandwidth", h_bw_sum),
        ("resource", h_res_sum),
        ("capability", h_cap_sum),
    ]:
        lam = AL_PARAMS[name]["lam"]
        mu = AL_PARAMS[name]["mu"]
        L_penalty += lam * h_sum + 0.5 * mu * h_sum ** 2

    total_energy = total_exec_energy + comm_energy
    fitness = alpha * makespan + beta * total_energy + L_penalty

    if return_detail:
        return (
            {"makespan": makespan, "energy": total_energy},
            {"lat": h_lat_sum, "bw": h_bw_sum, "res": h_res_sum, "cap": h_cap_sum},
        )

    return makespan, total_energy, L_penalty, h_res_sum, h_cap_sum, h_bw_sum, h_lat_sum



# 取均值或随机
def mean_or_rand(rng):  # 传入 [low, high]
    return random.randint(*rng)


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


def shannon_capacity(b_mhz, P_tx, h_gain, w0=WHITE_NOISE):
    return b_mhz * 1e6 * math.log2(1 + P_tx * h_gain / w0)  # bit/s


def compute_ul_dl_delay(data_mb, rate_mbps, prop_ms):
    """data_mb / rate_mbps  (秒→ms) + 传播"""
    return data_mb * 8 / rate_mbps * 1e3 + prop_ms

# <<< NEW: 辅助函数，正确计算传输时间 >>>
def _get_transmission_time_s(data_size_mb: float, rate_mbps: float) -> float:
    """计算数据传输所需的时间（秒），处理速率为0的情况。"""
    if rate_mbps <= 0:
        return float('inf') # 如果速率为0，传输时间为无穷大
    return (data_size_mb * 8) / rate_mbps

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
        rate = dev_iot.bandwidth
        prop_delay = dev_iot.delay
        gw_row = gw_matrix[dev_iot.id - 1]
        gw_idx = np.where(gw_row > 0)[0][0]

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


def compute_task_finish_time(task_id,
                             agent_dag_edges,  # [(from_mod, to_mod, data_attr), ...]
                             exec_time_map,
                             edge_delay_map):
    """
    返回 T_m_total （该任务 makespan）
    agent_dag_edges: task 内的 agent 级 DAG 边列表
    """
    # 1. 构建邻接表 & 入度
    succ = defaultdict(list)
    indeg = defaultdict(int)
    modules_in_task = set()

    for u_mod, v_mod, attr in agent_dag_edges:
        succ[u_mod].append(v_mod)
        indeg[v_mod] += 1
        modules_in_task.update([u_mod, v_mod])

    # 源节点 = 入口虚拟 sensor 节点 (f=0)  或 indeg==0
    earliest_finish = {}  # f_mi for modules & virtual sink
    q = deque()

    for mod in modules_in_task:
        if indeg[mod] == 0:
            earliest_finish[mod] = 0.0
            q.append(mod)

    # 2. 拓扑遍历
    while q:
        u = q.popleft()
        base_finish = earliest_finish[u] + exec_time_map.get((task_id, u), 0.0)

        for v in succ[u]:
            # 累加延迟
            edge_delay = edge_delay_map.get((task_id, u, v), 0.0)
            candidate_time = base_finish + edge_delay
            earliest_finish[v] = max(earliest_finish.get(v, 0.0), candidate_time)
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    # 3. 找到汇节点（没有后继）最大的 f
    max_finish = 0.0
    for mod in modules_in_task:
        if not succ[mod]:  # 没有后继 == sink/drive
            max_finish = max(max_finish,
                             earliest_finish[mod])  # sink 不再加 exec（=0）
    return max_finish


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
        max_nfes = 10000 * self.problem_size
        if self.max_g is None:
            self.max_g = max_nfes  # 和原论文一致

        KF_pool = np.array([0.1, 1.0, 0.5, 0.8])
        KR_pool = np.array([0.02, 0.1, 0.9, 0.9])
        max_pop_size = self.pop_size
        min_pop_size = 12

        # ---- helper: 轻量随机扰动 ----
        def mutate(arr: np.ndarray, p=0.05) -> np.ndarray:
            new = arr.copy()
            device_num = len(self.device_list)
            for i in range(new.size):
                if random.random() < p:
                    # 根据索引位置判断类型：soft or sense/act
                    if i == 0:
                        # 第一个一定是 soft
                        new[i] = random.randint(1, self.cloud.id)
                    else:
                        # 统一用边界裁剪，不再手动区别；后面会用 boundConstraint
                        new[i] = random.randint(self.low[i], self.high[i])
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

            for i in range(self.pop_size):
                nfes += 1
                if nfes > max_nfes:
                    break
                if child_fit[i] < bsf_fit_var:
                    bsf_fit_var = child_fit[i]
                    bsf_solution = ui[i, :].copy()
                    print(f"[Gen {g}] best_f={bsf_fit_var:.3f} | "
                          f"ms={child_ms[i]:.3f} | E={child_energy[i]:.3f} | "
                          f"pen={child_pen[i]:.3f} | res={child_res[i]:.3f} | "
                          f"cap={child_cap[i]:.3f} | bw={child_bw[i]:.3f} | lat={child_lat[i]:.3f}")

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
