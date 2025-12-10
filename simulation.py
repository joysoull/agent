import json
import random
from dataclasses import dataclass
from typing import Set

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


# ✅ 设备类型比例
wired_ratio = 0.7   # 70%为有线设备
wireless_ratio = 0.3

# ✅ 有线通信参数 (工业以太网)
wired_bandwidth_range = [80, 100]     # Mbps
wired_delay_range = [1, 2]            # ms
# ✅ 无线通信参数 (Wi-Fi6 / 5G)
near_wireless_rate = [15, 30]         # Mbps
near_wireless_delay = [2, 5]          # ms
far_wireless_rate = [10, 20]           # Mbps
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
        'cpu_power': 1,  # 轻量 IoT 单核功耗 1.5W
        'gpu_power': 0,
    },
    'medium': {
        'cpu': 40e9,  # 10GFLOPS 单核算力
        'num': 8,
        'gpu': 2e12,  # FP32 FLOPS
        'gpu_mem': 8,
        'ram': 8,  # 内存大小
        'disk': 100,  # 存储
        'cpu_power': 1,  # IoT 单核功耗 1W
        'gpu_power': 10.0,
    },
    'large': {
        'cpu': 60e9,  # 10GFLOPS 单核算力
        'num': 32,
        'gpu': 82e12,  # FP32 FLOPS
        'gpu_mem': 24,  # 24G显存
        'ram': 128,  # 内存大小
        'disk': 500,  # 存储
        'cpu_power': 6.0,
        'gpu_power': 400,
    }
}

# 假设总共的能力编号范围如下：
soft_capability_pool = list(range(1, 11))  # soft 能力编号 1~15
soft_capability_pool_cpu = list(range(8, 11))
sense_capability_pool = list(range(1, 4))  # sense 能力编号 1~3
act_capability_pool = list(range(1, 4))  # act 能力编号 1~3

# 云服务器
cloud_server = {
    'type': 'Cloud',
    'id': 0,
    'resource': {
        'cpu': 100e9,  # 100GFLOPS 单核算力
        'num': 64,  # CPU 核心数量
        'gpu': 200e12,  # FP32 FLOPS
        'gpu_mem': 80,  # 80G显存
        'ram': 256,  # 内存大小
        'disk': 4000,  # 存储
        'cpu_power': 7.5,
        'gpu_power': 350,
    },
    'act_cap': [],
    'sense_cap': [],
    'soft_cap': soft_capability_pool,
    'bandwidth': -1,
    'delay': -1,
}


def create_device():
    # IoT设备列表
    device_list = []
    # 创建IOT设备，从tiny，small,medium随机选择创建，并添加到设备列表中
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

        device = {
            'type': 'Device',
            'id': i,
            'conn_type': conn_type,  # ✅ 新增字段
            'resource': config_map[device_type],
            'act_cap': random.sample(act_capability_pool, random.randint(2, 3)),
            'sense_cap': random.sample(sense_capability_pool, random.randint(2, 3)),
            'soft_cap': soft_pool,
            'bandwidth': bandwidth,
            'delay': delay,
        }
        device_list.append(device)

    edge_server_list = []
    # 创建边缘服务器，large，并添加到边缘服务器列表中
    for i in range(1, edge_number + 1):
        edge_server = {
            'type': 'Edge',
            'id': i,
            'resource': config_map['large'],
            'act_cap': [],
            'sense_cap': [],
            'soft_cap': soft_capability_pool,
            'bandwidth': -1,
            'delay': -1,
        }
        edge_server_list.append(edge_server)

    # 合并所有内容保存到 JSON（除了 gw_matrix）
    data = {
        "cloud_server": cloud_server,
        "device_list": device_list,
        "edge_server_list": edge_server_list
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

@dataclass
class Gateway:
    id: int
    b_DL: float  # 下行信道带宽 (MHz)
    P_tx: float  # 网关发送功率 (W)
    P_rx: float  # 网关接收功耗 (W)
    h_gain: float  # 信道增益 (无量纲)

create_device()