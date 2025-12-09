import os
import json
import re
import random

import networkx as nx
from matplotlib import pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.readwrite import json_graph
def add_random_type_and_index_to_nodes(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)

    proc_nodes = []
    sense_nodes = []
    act_nodes = []

    for node in data["nodes"]:
        node_id = node["id"]
        match = re.match(r"(proc|sense|act)(\d+)", node_id)
        if not match:
            node["type"] = "unknown"
            node["idx"] = -1
            continue

        node_type = match.group(1)
        node["type"] = node_type

        if node_type == "proc":
            proc_nodes.append(node)
        elif node_type == "sense":
            sense_nodes.append(node)
        elif node_type == "act":
            act_nodes.append(node)

    # 为每类节点随机分配 idx
    for node in proc_nodes:
        node["idx"] = random.randint(1, 10)
    for node in sense_nodes:
        node["idx"] = random.randint(1, 3)
    for node in act_nodes:        node["idx"] = random.randint(1, 3)

    # 添加通信相关参数（所有节点都加）
    for node in data["links"]:
        node["data_size"] = round(random.uniform(2, 6), 1)  # MB
        node["bandwidth_req"] = round(random.uniform(0.1, 1.0), 2)  # Mbps
        node["latency_req"] = random.randint(100, 1000)  # ms

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✅ Saved: {output_path}")
def tweak_existing_link_constraints(
    input_dir=".",                # 文件所在目录
    prefix="dag",                 # 文件前缀
    start=1, end=10,              # 编号范围
    mode="random",                # "random" 或 "scale"
    # random 模式的取值范围
    data_size_range=(2.0, 6.0),   # MB
    bw_range=(0.1, 1.0),          # Mbps
    lat_range=(100, 1000),        # ms
    # scale 模式的缩放与裁剪
    scale_factors=None,           # 例: {"data_size": 0.8, "bandwidth_req": 1.2, "latency_req": 0.5}
    clip_ranges=None,             # 例: {"data_size": (1.0, 10.0), "bandwidth_req": (0.05, 5.0), "latency_req": (20, 2000)}
    seed=None,                    # 设定随机种子，便于重复
    overwrite=True,               # 是否覆盖原文件
    output_dir=None               # 若不覆盖，输出到该目录；None 时与输入目录相同
):
    """
    对已生成的 {prefix}{i}_typed_constraint.json 批量修改 links 里的通信参数。

    mode="random":   在给定范围内重新随机赋值
    mode="scale":    按比例缩放原值，并裁剪到 clip_ranges
    """
    if seed is not None:
        random.seed(seed)

    if output_dir is None:
        output_dir = input_dir

    if mode not in ("random", "scale"):
        raise ValueError("mode 必须为 'random' 或 'scale'")

    if mode == "scale":
        # 默认缩放因子和裁剪范围
        if scale_factors is None:
            scale_factors = {"data_size": 1.0, "bandwidth_req": 1.0, "latency_req": 1.0}
        if clip_ranges is None:
            clip_ranges = {
                "data_size": data_size_range,
                "bandwidth_req": bw_range,
                "latency_req": lat_range
            }

        def _scale_and_clip(val, key):
            sf = scale_factors.get(key, 1.0)
            lo, hi = clip_ranges.get(key, (val, val))
            return max(lo, min(hi, val * sf))

    changed_files = 0
    changed_edges = 0

    for i in range(start, end + 1):
        in_path = os.path.join(input_dir, f"{prefix}{i}_typed_constraint.json")
        if not os.path.isfile(in_path):
            print(f"⚠️ 跳过：未找到 {in_path}")
            continue

        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "links" not in data or not isinstance(data["links"], list):
            print(f"⚠️ 跳过：{in_path} 中没有 links 列表")
            continue

        file_edge_updates = 0
        for link in data["links"]:
            # 兼容键名不存在的情况，给默认值
            if mode == "random":
                link["data_size"]     = round(random.uniform(*data_size_range), 1)
                link["bandwidth_req"] = round(random.uniform(*bw_range), 2)
                link["latency_req"]   = int(round(random.uniform(*lat_range)))
            else:  # mode == "scale"
                # 原值不存在就当成范围下限
                cur_data_size = float(link.get("data_size", data_size_range[0]))
                cur_bw_req    = float(link.get("bandwidth_req", bw_range[0]))
                cur_lat       = float(link.get("latency_req", lat_range[0]))

                link["data_size"]     = round(_scale_and_clip(cur_data_size, "data_size"), 1)
                link["bandwidth_req"] = round(_scale_and_clip(cur_bw_req, "bandwidth_req"), 2)
                link["latency_req"]   = int(round(_scale_and_clip(cur_lat, "latency_req")))

            file_edge_updates += 1

        # 输出路径
        if overwrite:
            out_path = in_path
        else:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, os.path.basename(in_path))

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        changed_files += 1
        changed_edges += file_edge_updates
        print(f"✅ 已更新 {out_path} | 修改边数: {file_edge_updates}")

    print(f"\n完成：共处理文件 {changed_files} 个，累计修改边 {changed_edges} 条。")

def batch_process_dags(input_dir="../dag", prefix="dag", start=1, end=10):
    for i in range(start, end + 1):
        input_file = os.path.join(input_dir, f"{prefix}{i}.json")
        output_file = os.path.join(f"{prefix}{i}_typed_constraint.json")
        add_random_type_and_index_to_nodes(input_file, output_file)
def load_dag_from_json(filename="dag.json"):
    with open(filename, 'r') as f:
        data = json.load(f)
    G = json_graph.node_link_graph(data, directed=True)
    return G

def draw_dag(G, filename):
    pos = graphviz_layout(G, prog='dot')  # 层次布局

    # 分类节点
    sense_nodes = [n for n in G.nodes if str(n).startswith("sense")]
    act_nodes = [n for n in G.nodes if str(n).startswith("act")]
    proc_nodes = [n for n in G.nodes if n not in sense_nodes and n not in act_nodes]

    # 绘制不同类别的节点
    nx.draw_networkx_nodes(G, pos, nodelist=proc_nodes, node_color='skyblue', node_size=400, label='proc')
    nx.draw_networkx_nodes(G, pos, nodelist=sense_nodes, node_color='lightgreen', node_size=400, label='sense')
    nx.draw_networkx_nodes(G, pos, nodelist=act_nodes, node_color='salmon', node_size=400, label='act')

    nx.draw_networkx_edges(G, pos, arrows=True)
    nx.draw_networkx_labels(G, pos)

    plt.title("DAG with Node Types Highlighted")
    plt.axis("off")
    plt.legend()
    # 保存为 PNG 图像
    plt.savefig(filename, format='png', bbox_inches='tight', dpi=300)
    plt.show()
# 执行批量处理
tweak_existing_link_constraints(
    input_dir=".", prefix="dag", start=1, end=10,
    mode="random",
    data_size_range=(0.5, 2),
    bw_range=(0.1, 1.0),
    lat_range=(600, 1000),
    seed=42,
    overwrite=True
)

# for i in range(1, 11):
#     G = load_dag_from_json(f"dag{i}_typed.json")
#     draw_dag(G, f"dag{i}_typed.png")
