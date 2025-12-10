import json
import random
from dataclasses import dataclass
from typing import Set, List, Dict

import numpy as np

# 任务数量(M)
task_number = 10
# IoT设备数量
device_number = 30
# 云服务器数量
cloud_number = 1
# 边缘服务器数量
edge_number = 10
# 网关数量
gateway_number = 3

# ====== 智慧工厂层级参数（工厂 / 车间 / 生产线） ======
factory_number = 1               # 当前只建 1 个工厂
workshop_per_factory = 3         # 每个工厂下的车间数量
lines_per_workshop = 3           # 每个车间下的生产线数量
# 总生产线数
total_lines = factory_number * workshop_per_factory * lines_per_workshop

# ✅ 设备类型比例
wired_ratio = 0.7   # 70%为有线设备
wireless_ratio = 0.3

# ✅ 有线通信参数 (工业以太网)
wired_bandwidth_range = [80, 100]     # Mbps
wired_delay_range = [1, 2]            # ms
# ✅ 无线通信参数 (Wi-Fi6 / 5G)
near_wireless_rate = [15, 30]         # Mbps
near_wireless_delay = [2, 5]          # ms
far_wireless_rate = [10, 20]          # Mbps
far_wireless_delay = [8, 12]          # ms

edge_inter_delay = 1
cloud_edge_delay = 20

# 资源模板
config_map = {
    'tiny': {
        'cpu': 10e9,  # 10GFLOPS 单核算力
        'num': 1,
        'gpu': 0,  # FP32 FLOPS
        'gpu_mem': 0,
        'ram': 2,  # 内存大小
        'disk': 20,  # 存储
        'cpu_power': 1,  # 轻量 IoT 单核功耗 1.5W
        'gpu_power': 0,
    },
    'small': {
        'cpu': 15e9,  # 10GFLOPS 单核算力
        'num': 2,
        'gpu': 0,  # FP32 FLOPS
        'gpu_mem': 0,
        'ram': 4,  # 内存大小
        'disk': 40,  # 存储
        'cpu_power': 1,
        'gpu_power': 0,
    },
    'medium': {
        'cpu': 40e9,
        'num': 8,
        'gpu': 2e12,
        'gpu_mem': 8,
        'ram': 8,
        'disk': 100,
        'cpu_power': 1,
        'gpu_power': 10.0,
    },
    'large': {
        'cpu': 60e9,
        'num': 32,
        'gpu': 82e12,
        'gpu_mem': 24,
        'ram': 128,
        'disk': 500,
        'cpu_power': 6.0,
        'gpu_power': 400,
    }
}

# 假设总共的能力编号范围如下：
soft_capability_pool = list(range(1, 11))       # soft 能力编号 1~10
soft_capability_pool_cpu = list(range(8, 11))   # 弱算力设备只给一部分 soft
sense_capability_pool = list(range(1, 4))       # sense 能力编号 1~3
act_capability_pool = list(range(1, 4))         # act 能力编号 1~3

# 云服务器
cloud_server = {
    'type': 'Cloud',
    'id': 0,
    'resource': {
        'cpu': 100e9,
        'num': 64,
        'gpu': 200e12,
        'gpu_mem': 80,
        'ram': 256,
        'disk': 4000,
        'cpu_power': 7.5,
        'gpu_power': 350,
    },
    'act_cap': [],
    'sense_cap': [],
    'soft_cap': soft_capability_pool,
    'bandwidth': -1,
    'delay': -1,

    # <<< 层级信息：云不在工厂层级内，单独作为 "cloud" >>>
    'level': 'cloud',         # 'line' / 'workshop' / 'factory' / 'cloud'
    'factory_id': None,
    'workshop_id': None,
    'line_id': None,
}


def _assign_line_hierarchy(dev_index: int):
    """
    根据设备索引，把 IoT 设备映射到 (工厂, 车间, 生产线).
    dev_index: 从 0 开始的索引（0..device_number-1）
    """
    # 做一个简单的轮转映射: 先按生产线，再展开到车间和工厂
    line_global_idx = dev_index % total_lines  # 0..total_lines-1

    factory_id = 1 + (line_global_idx // (workshop_per_factory * lines_per_workshop))
    within_factory = line_global_idx % (workshop_per_factory * lines_per_workshop)
    workshop_id = 1 + (within_factory // lines_per_workshop)
    line_id = 1 + (within_factory % lines_per_workshop)

    return factory_id, workshop_id, line_id


def _sample_task_deploy_domains() -> List[Dict]:
    """
    为每个任务随机生成一个部署域:
      - line_only         : 只能在生产线级设备上部署
      - line_workshop     : 生产线 + 车间
      - workshop_factory  : 车间 + 工厂
      - factory_cloud     : 工厂 + 云
      - all               : 全层级通用
    """
    domain_patterns = [
        ("line_only", ["line"]),
        ("line_workshop", ["line", "workshop"]),
        ("workshop_factory", ["workshop", "factory"]),
        ("factory_cloud", ["factory", "cloud"]),
        ("all", ["line", "workshop", "factory", "cloud"]),
    ]

    task_domains = []
    for tid in range(1, task_number + 1):
        pattern_name, levels = random.choice(domain_patterns)
        task_domains.append({
            "task_id": tid,
            "pattern": pattern_name,
            "allowed_levels": levels
        })
    return task_domains


def create_device():
    # IoT设备列表
    device_list = []

    # 创建 IOT 设备，并添加到设备列表中
    for i in range(1, device_number + 1):
        device_type = random.choice(['tiny', 'small', 'medium'])

        # ✅ 判断设备是有线还是无线
        conn_type = 'wired' if random.random() < wired_ratio else 'wireless'

        if device_type == 'medium':
            soft_pool = soft_capability_pool
        else:
            soft_pool = soft_capability_pool_cpu

        # ✅ 根据连接类型分配速率与延迟
        if conn_type == 'wired':
            bandwidth = random.uniform(*wired_bandwidth_range)
            delay = random.uniform(*wired_delay_range)
        else:
            # 无线设备区分近/远距离
            is_near = (random.random() < 0.7)
            bandwidth = random.uniform(*near_wireless_rate) if is_near else random.uniform(*far_wireless_rate)
            delay = random.uniform(*near_wireless_delay) if is_near else random.uniform(*far_wireless_delay)

        # <<< NEW: 生产线层级映射 >>>
        factory_id, workshop_id, line_id = _assign_line_hierarchy(i - 1)

        device = {
            'type': 'Device',
            'id': i,
            'conn_type': conn_type,
            'resource': config_map[device_type],
            'act_cap': random.sample(act_capability_pool, random.randint(2, 3)),
            'sense_cap': random.sample(sense_capability_pool, random.randint(2, 3)),
            'soft_cap': soft_pool,
            'bandwidth': bandwidth,
            'delay': delay,

            # 物理层级：IoT 设备都挂在具体生产线上
            'level': 'line',              # 生产线级
            'factory_id': factory_id,
            'workshop_id': workshop_id,
            'line_id': line_id,
        }
        device_list.append(device)

    edge_server_list = []

    # 创建边缘服务器，并添加到边缘服务器列表中
    for i in range(1, edge_number + 1):
        # 一部分部署在“车间级”，一部分部署在“工厂级”
        if i <= edge_number // 2:
            # 车间级边缘服务器：接入多个生产线
            factory_id = 1
            workshop_id = random.randint(1, workshop_per_factory)
            line_id = None
            level = 'workshop'
        else:
            # 工厂级边缘服务器：汇聚多个车间
            factory_id = 1
            workshop_id = None
            line_id = None
            level = 'factory'

        edge_server = {
            'type': 'Edge',
            'id': i,
            'resource': config_map['large'],
            'act_cap': [],
            'sense_cap': [],
            'soft_cap': soft_capability_pool,
            'bandwidth': -1,
            'delay': -1,

            # <<< 层级信息：车间级 / 工厂级边缘 >>>,
            'level': level,               # 'workshop' 或 'factory'
            'factory_id': factory_id,
            'workshop_id': workshop_id,
            'line_id': line_id,
        }
        edge_server_list.append(edge_server)

    # <<< NEW: 为每个任务生成部署域约束 >>>
    task_deploy_domains = _sample_task_deploy_domains()

    # 合并所有内容保存到 JSON（除了 gw_matrix）
    data = {
        "cloud_server": cloud_server,
        "device_list": device_list,
        "edge_server_list": edge_server_list,
        # 新增：任务部署域
        "task_deploy_domains": task_deploy_domains,
        # 新增：工厂拓扑元数据（方便后续建模使用）
        "factory_meta": {
            "factory_number": factory_number,
            "workshop_per_factory": workshop_per_factory,
            "lines_per_workshop": lines_per_workshop
        }
    }

    # 保存 JSON 文件
    with open("infrastructure_config.json", "w") as f:
        json.dump(data, f, indent=2)

    # ✅ 创建网关连接矩阵：仅无线设备通过网关连接
    gw_matrix = np.zeros((device_number, gateway_number))
    for i, dev in enumerate(device_list):
        if dev['conn_type'] == 'wireless':
            gw_id = np.random.randint(0, gateway_number)
            gw_matrix[i][gw_id] = 1
    np.save("gw_matrix.npy", gw_matrix)


@dataclass
class Resource:
    cpu: float
    num: int
    gpu: float
    gpu_mem: float
    ram: float
    disk: float
    cpu_power: float
    gpu_power: float


@dataclass
class Device:
    type: str  # "Device", "Edge", "Cloud"
    id: int
    conn_type: str
    resource: Resource
    act_cap: Set[int]
    sense_cap: Set[int]
    soft_cap: Set[int]
    bandwidth: float
    delay: float

    # <<< NEW: 为物理实体补充工厂层级信息 >>>
    level: str = "line"          # 'line' / 'workshop' / 'factory' / 'cloud'
    factory_id: int | None = 1
    workshop_id: int | None = 1
    line_id: int | None = 1


@dataclass
class Gateway:
    id: int
    b_DL: float  # 下行信道带宽 (MHz)
    P_tx: float  # 网关发送功率 (W)
    P_rx: float  # 网关接收功耗 (W)
    h_gain: float  # 信道增益 (无量纲)


create_device()
