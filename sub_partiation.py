

import networkx as nx
from typing import List, Tuple, Dict, Set, Optional

import ast
import pandas as pd
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout



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


class ModuleCandidate:
    def __init__(self, proc_nodes: Set[str], sense_nodes: Set[str], act_nodes: Set[str],
                 entry_nodes: Set[str], exit_nodes: Set[str]):
        self.proc_nodes = proc_nodes
        self.sense_nodes = sense_nodes
        self.act_nodes = act_nodes
        self.entry_nodes = entry_nodes
        self.exit_nodes = exit_nodes
        self.all_nodes = proc_nodes | sense_nodes | act_nodes

    def size(self) -> int:
        return len(self.all_nodes)

    def get_capabilities(self, G: nx.DiGraph) -> Tuple[Set[int], Set[int], Set[int]]:
        """提取模块所需的能力"""
        C_sense, C_act, C_soft = set(), set(), set()

        for node in self.all_nodes:
            node_type = G.nodes[node].get("type")
            idx = G.nodes[node].get("idx")

            if node_type == "sense":
                C_sense.add(idx)
            elif node_type == "act":
                C_act.add(idx)
            elif node_type == "proc":
                C_soft.add(idx)

        return C_sense, C_act, C_soft


class AgentCapabilityMatcher:
    def __init__(self, agents: List[AgentTemplate]):
        self.agents = agents
        # 为快速查找创建索引
        self.agents_by_capability = self._build_capability_index()

    def _build_capability_index(self) -> Dict[str, List[AgentTemplate]]:
        """构建能力索引以加速匹配"""
        index = defaultdict(list)
        for agent in self.agents:
            # 按照能力类型建立索引
            for cap in agent.C_sense:
                index[f"sense_{cap}"].append(agent)
            for cap in agent.C_act:
                index[f"act_{cap}"].append(agent)
            for cap in agent.C_soft:
                index[f"soft_{cap}"].append(agent)
        return index

    def find_matching_agents(self, C_sense: Set[int], C_act: Set[int], C_soft: Set[int]) -> List[AgentTemplate]:
        """查找能够覆盖所有能力的智能体"""
        matching_agents = []
        for agent in self.agents:
            if agent.covers(C_sense, C_act, C_soft):
                matching_agents.append(agent)
        return matching_agents

    def find_collaborative_agents(self, C_sense: Set[int], C_act: Set[int], C_soft: Set[int],
                                  allowed_agents: Set[AgentTemplate]) -> List[
        Tuple[AgentTemplate, AgentTemplate]]:
        """查找可以协作覆盖能力的智能体对（最多1次协作）"""
        collaborative_pairs = []
        allowed_list = list(allowed_agents)

        for i, agent1 in enumerate(allowed_list):
            for j, agent2 in enumerate(allowed_list):
                if i >= j:  # 避免重复检查
                    continue

                # 检查两个智能体能否协作覆盖所有能力
                combined_sense = agent1.C_sense | agent2.C_sense
                combined_act = agent1.C_act | agent2.C_act
                combined_soft = agent1.C_soft | agent2.C_soft

                if (C_sense.issubset(combined_sense) and
                        C_act.issubset(combined_act) and
                        C_soft.issubset(combined_soft)):
                    # 确保proc能力只能由单个agent提供
                    if C_soft.issubset(agent1.C_soft) or C_soft.issubset(agent2.C_soft):
                        collaborative_pairs.append((agent1, agent2))

        return collaborative_pairs


class OptimizedGraphPartitioner:
    def __init__(self, G: nx.DiGraph, agents: List[AgentTemplate]):
        self.G = G
        self.agents = agents
        self.matcher = AgentCapabilityMatcher(agents)
        self.topo_order = list(nx.topological_sort(G))
        self.collaboration_records = []  # 每项记录：dict格式，字段如下

    def _get_connected_sense_act_nodes(self, proc_node: str) -> Tuple[Set[str], Set[str]]:
        """获取与proc节点直接连接的sense和act节点"""
        sense_nodes = set()
        act_nodes = set()

        # 检查所有邻居节点
        for pred in self.G.predecessors(proc_node):
            if self.G.nodes[pred].get("type") == "sense":
                sense_nodes.add(pred)

        for succ in self.G.successors(proc_node):
            if self.G.nodes[succ].get("type") == "act":
                act_nodes.add(succ)

        return sense_nodes, act_nodes

    def _is_valid_module(self, module: ModuleCandidate) -> bool:
        """检查模块是否满足约束条件"""
        # 1. 大小约束
        if module.size() > 3:
            return False

        # 2. 连通性约束：内部节点不能与外部节点直接通信
        for node in module.all_nodes:
            if node in module.entry_nodes or node in module.exit_nodes:
                continue

            # 检查是否有与模块外部的连接
            for neighbor in self.G.neighbors(node):
                if neighbor not in module.all_nodes:
                    return False

        return True

    def _generate_module_candidates(self, start_proc: str, assigned_nodes: Set[str]) -> List[ModuleCandidate]:
        """从给定的proc节点开始生成模块候选"""
        candidates = []

        if start_proc in assigned_nodes:
            return candidates

        # 获取直接连接的sense和act节点
        sense_nodes, act_nodes = self._get_connected_sense_act_nodes(start_proc)

        # 基础模块：只包含当前proc节点及其直接邻居
        base_module = ModuleCandidate(
            proc_nodes={start_proc},
            sense_nodes=sense_nodes,
            act_nodes=act_nodes,
            entry_nodes=sense_nodes,
            exit_nodes=act_nodes if act_nodes else {start_proc}
        )

        if self._is_valid_module(base_module):
            candidates.append(base_module)

        # 尝试扩展模块：添加后续的proc节点
        for next_proc in self.G.successors(start_proc):
            if (self.G.nodes[next_proc].get("type") == "proc" and
                    next_proc not in assigned_nodes and
                    len(base_module.all_nodes) < 3):

                next_sense, next_act = self._get_connected_sense_act_nodes(next_proc)

                extended_module = ModuleCandidate(
                    proc_nodes={start_proc, next_proc},
                    sense_nodes=sense_nodes | next_sense,
                    act_nodes=act_nodes | next_act,
                    entry_nodes=sense_nodes,
                    exit_nodes=next_act if next_act else {next_proc}
                )

                if self._is_valid_module(extended_module):
                    candidates.append(extended_module)

        return candidates

    def _evaluate_module_assignment(self, module: ModuleCandidate, selected_agents: Set[AgentTemplate]) -> Tuple[
        float, Optional[AgentTemplate], Optional[Tuple[AgentTemplate, AgentTemplate]]]:
        """
        1. 先尝试单智能体完全覆盖（可选用任何智能体，以减少后续协作需求）
        2. 如无法覆盖，则 **仅在 selected_agents 内部** 尝试协作覆盖
        3. 若两种方式都失败，则返回 0 分
        """
        # ----------- 能力需求 -----------
        C_sense, C_act, C_soft = module.get_capabilities(self.G)

        # ----------- 单智能体覆盖 ----------
        single_agents = self.matcher.find_matching_agents(C_sense, C_act, C_soft)
        if single_agents:
            best_agent = min(single_agents, key=lambda ag: ag.total_capability_size())
            return 1.0, best_agent, None

        # ----------- 协作覆盖（仅限已选智能体） ----------
        collaborative_pairs: List[Tuple[AgentTemplate, AgentTemplate]] = []
        collaborative_pairs = self.matcher.find_collaborative_agents(C_sense, C_act, C_soft, selected_agents)
        # if collaborative_pairs:
        #     best_pair = min(collaborative_pairs,
        #                     key=lambda pair: pair[0].total_capability_size() + pair[1].total_capability_size())
        #     return 0.5, None, best_pair
        if selected_agents:
            # matcher 内部应只在 selected_agents 中做组合。若原实现无法过滤，可在此处再做一次过滤。
            all_pairs = self.matcher.find_collaborative_agents(C_sense, C_act, C_soft, selected_agents)
            collaborative_pairs = [
                pair for pair in all_pairs
                if pair[0] in selected_agents and pair[1] in selected_agents
            ]

        if collaborative_pairs:
            best_pair = min(
                collaborative_pairs,
                key=lambda p: p[0].total_capability_size() + p[1].total_capability_size()
            )
            return 0.5, None, best_pair  # 分数 0.5 仅次于单智能体

            # ----------- 无法覆盖 ----------
        return 0.0, None, None

    def _create_fallback_module(self, proc_node: str, assigned_nodes: Set[str]) -> ModuleCandidate:
        """为无法匹配的proc节点创建最小模块"""
        if proc_node in assigned_nodes:
            return None

        sense_nodes, act_nodes = self._get_connected_sense_act_nodes(proc_node)

        # 只包含未分配的邻居节点
        available_sense = {n for n in sense_nodes if n not in assigned_nodes}
        available_act = {n for n in act_nodes if n not in assigned_nodes}

        return ModuleCandidate(
            proc_nodes={proc_node},
            sense_nodes=available_sense,
            act_nodes=available_act,
            entry_nodes=available_sense,
            exit_nodes=available_act if available_act else {proc_node}
        )

    def _find_best_agent_with_relaxed_constraints(self, C_sense: Set[int], C_act: Set[int], C_soft: Set[int]) -> \
            Optional[AgentTemplate]:
        """使用放宽约束寻找最佳智能体"""
        # 1. 首先尝试完全匹配
        matching_agents = self.matcher.find_matching_agents(C_sense, C_act, C_soft)
        if matching_agents:
            return min(matching_agents, key=lambda ag: ag.total_capability_size())

        # 2. 必须能覆盖proc能力，sense和act可以放宽
        proc_capable_agents = [ag for ag in self.agents if C_soft.issubset(ag.C_soft)]
        if proc_capable_agents:
            # 选择能覆盖最多其他能力的智能体
            def coverage_score(agent):
                sense_coverage = len(C_sense.intersection(agent.C_sense)) / max(len(C_sense), 1)
                act_coverage = len(C_act.intersection(agent.C_act)) / max(len(C_act), 1)
                return sense_coverage + act_coverage

            return max(proc_capable_agents, key=coverage_score)

        # 3. 最后的备选：选择总体能力最强的智能体
        if self.agents:
            return max(self.agents, key=lambda ag: ag.total_capability_size())

        return None

    def _assign_orphaned_nodes(self, assigned_nodes: Set[str],
                               result_modules: List[Tuple[ModuleCandidate, AgentTemplate]]) -> List[
        Tuple[ModuleCandidate, AgentTemplate]]:
        """分配孤立的sense和act节点到现有模块或创建新模块"""
        orphaned_nodes = []

        for node in self.G.nodes:
            if node not in assigned_nodes:
                orphaned_nodes.append(node)

        if not orphaned_nodes:
            return result_modules

        # 尝试将孤立节点合并到现有模块
        updated_modules = []
        remaining_orphans = set(orphaned_nodes)

        for module, agent in result_modules:
            # 检查是否可以将孤立节点加入此模块
            candidates_for_merge = []

            for orphan in list(remaining_orphans):
                if module.size() >= 3:  # 模块已满
                    continue

                # 检查是否与模块有连接
                has_connection = False
                for module_node in module.all_nodes:
                    if (self.G.has_edge(orphan, module_node) or
                            self.G.has_edge(module_node, orphan)):
                        has_connection = True
                        break

                if has_connection:
                    candidates_for_merge.append(orphan)

            # 选择最多一个孤立节点合并（保持模块大小≤3）
            if candidates_for_merge and module.size() < 3:
                orphan_to_merge = candidates_for_merge[0]
                orphan_type = self.G.nodes[orphan_to_merge].get("type")

                # 创建扩展模块
                new_sense_nodes = module.sense_nodes.copy()
                new_act_nodes = module.act_nodes.copy()

                if orphan_type == "sense":
                    new_sense_nodes.add(orphan_to_merge)
                elif orphan_type == "act":
                    new_act_nodes.add(orphan_to_merge)

                extended_module = ModuleCandidate(
                    proc_nodes=module.proc_nodes,
                    sense_nodes=new_sense_nodes,
                    act_nodes=new_act_nodes,
                    entry_nodes=new_sense_nodes,
                    exit_nodes=new_act_nodes if new_act_nodes else module.proc_nodes
                )

                # 检查智能体是否仍能覆盖
                C_sense, C_act, C_soft = extended_module.get_capabilities(self.G)
                if agent.covers(C_sense, C_act, C_soft):
                    updated_modules.append((extended_module, agent))
                    remaining_orphans.remove(orphan_to_merge)
                    assigned_nodes.add(orphan_to_merge)
                else:
                    updated_modules.append((module, agent))
            else:
                updated_modules.append((module, agent))

        # 为剩余的孤立节点创建新模块
        for orphan in remaining_orphans:
            orphan_type = self.G.nodes[orphan].get("type")
            orphan_idx = self.G.nodes[orphan].get("idx")

            if orphan_type == "sense":
                orphan_module = ModuleCandidate(
                    proc_nodes=set(),
                    sense_nodes={orphan},
                    act_nodes=set(),
                    entry_nodes={orphan},
                    exit_nodes={orphan}
                )
                C_sense, C_act, C_soft = {orphan_idx}, set(), set()
            elif orphan_type == "act":
                orphan_module = ModuleCandidate(
                    proc_nodes=set(),
                    sense_nodes=set(),
                    act_nodes={orphan},
                    entry_nodes={orphan},
                    exit_nodes={orphan}
                )
                C_sense, C_act, C_soft = set(), {orphan_idx}, set()
            else:
                continue

            # 寻找能覆盖的智能体
            suitable_agent = self._find_best_agent_with_relaxed_constraints(C_sense, C_act, C_soft)
            if suitable_agent:
                updated_modules.append((orphan_module, suitable_agent))

        return updated_modules

    def partition_graph(self) -> Tuple[List[Tuple[ModuleCandidate, AgentTemplate]], List[str]]:
        """执行图划分，确保所有节点都被分配"""
        assigned_nodes = set()
        result_modules = []
        unmatched_nodes = []
        selected_agents = set()
        # 第一阶段：按拓扑顺序处理proc节点
        proc_nodes = [node for node in self.topo_order if self.G.nodes[node].get("type") == "proc"]

        for proc_node in proc_nodes:
            if proc_node in assigned_nodes:
                continue

            # 生成候选模块
            candidates = self._generate_module_candidates(proc_node, assigned_nodes)

            # 评估每个候选模块
            best_score = 0.0
            best_module = None
            best_agent = None
            best_pair = None

            for candidate in candidates:
                score, agent, pair = self._evaluate_module_assignment(candidate, selected_agents)
                if score > best_score:
                    best_score = score
                    best_module = candidate
                    best_agent = agent
                    best_pair = pair

            # 如果找到了合适的分配
            if best_score > 0.0:
                if best_agent:
                    result_modules.append((best_module, best_agent))
                    selected_agents.add(best_agent)
                elif best_pair:
                    # chosen_agent = best_pair[0] if C_soft.issubset(best_pair[0].C_soft) else best_pair[1]
                    # result_modules.append((best_module, chosen_agent))
                    # selected_agents.update(best_pair)

                    # 重新计算模块能力，确定哪一个协作者覆盖了 proc 的 soft 能力
                    C_sense, C_act, C_soft = best_module.get_capabilities(self.G)
                    chosen_agent = best_pair[0] if C_soft.issubset(best_pair[0].C_soft) else best_pair[1]

                    result_modules.append((best_module, chosen_agent))
                    # 注意：这里不再把第二个协作者加入 selected_agents，
                    # 因为协作仅限已选集合，两个协作者本身就已在 selected_agents
                    self.collaboration_records.append({
                        "module": best_module,
                        "agents": best_pair,
                    })
                assigned_nodes.update(best_module.all_nodes)
            else:
                # 创建fallback模块
                fallback_module = self._create_fallback_module(proc_node, assigned_nodes)
                if fallback_module:
                    C_sense, C_act, C_soft = fallback_module.get_capabilities(self.G)
                    fallback_agent = self._find_best_agent_with_relaxed_constraints(C_sense, C_act, C_soft)

                    if fallback_agent:
                        result_modules.append((fallback_module, fallback_agent))
                        assigned_nodes.update(fallback_module.all_nodes)
                    else:
                        unmatched_nodes.append(proc_node)

        # 第二阶段：处理剩余的孤立节点
        result_modules = self._assign_orphaned_nodes(assigned_nodes, result_modules)

        # 检查最终的未分配节点
        final_unmatched = []
        for node in self.G.nodes:
            if node not in assigned_nodes:
                # 再次检查是否已被分配到更新的模块中
                found_in_module = False
                for module, _ in result_modules:
                    if node in module.all_nodes:
                        found_in_module = True
                        break
                if not found_in_module:
                    final_unmatched.append(node)

        return result_modules, final_unmatched


def draw_partitioned_dag_optimized(G: nx.DiGraph, modules: List[ModuleCandidate], filename: str):
    """绘制划分后的图"""
    pos = graphviz_layout(G, prog='dot')

    # 构造模块颜色映射
    module_colors = {}
    color_palette = plt.get_cmap("tab20").colors
    for i, module in enumerate(modules):
        color = color_palette[i % len(color_palette)]
        for node in module.all_nodes:
            module_colors[node] = color

    # 分类节点
    all_nodes = list(G.nodes)
    sense_nodes = [n for n in all_nodes if G.nodes[n].get("type") == "sense"]
    act_nodes = [n for n in all_nodes if G.nodes[n].get("type") == "act"]
    proc_nodes = [n for n in all_nodes if G.nodes[n].get("type") == "proc"]

    # 为每类节点生成颜色子列表
    proc_colors = [module_colors.get(n, "gray") for n in proc_nodes]
    sense_colors = [module_colors.get(n, "gray") for n in sense_nodes]
    act_colors = [module_colors.get(n, "gray") for n in act_nodes]

    # 绘图
    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, nodelist=proc_nodes, node_color=proc_colors,
                           node_size=500, node_shape='s', label='proc')
    nx.draw_networkx_nodes(G, pos, nodelist=sense_nodes, node_color=sense_colors,
                           node_size=400, node_shape='o', label='sense')
    nx.draw_networkx_nodes(G, pos, nodelist=act_nodes, node_color=act_colors,
                           node_size=400, node_shape='^', label='act')
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title("Optimized Graph Partitioning with Agent Assignment", fontsize=16)
    plt.axis("off")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(filename, format='png', bbox_inches='tight', dpi=300)
    plt.show()


import networkx as nx
from typing import List, Dict, Set, Tuple
from collections import defaultdict, Counter
import json
from networkx.readwrite import json_graph




def get_node_capability(node_attrs: Dict) -> Tuple[str, int]:
    return node_attrs.get("type"), node_attrs.get("idx")


def load_task_graph(filename: str) -> nx.DiGraph:
    with open(filename, 'r') as f:
        data = json.load(f)
    return json_graph.node_link_graph(data, directed=True)


# ---------- 协作统计工具函数 ----------
def _build_collab_records(req_set, agent_caps, cap_name, used_agents, main_agent_id):
    """
    req_set       : 模块需要的某类能力集合（Set[int]）
    agent_caps    : 主智能体实际拥有的该类能力集合
    cap_name      : 字符串 "sense" or "act"，仅用于字段名拼接
    used_agents   : {agent_id: AgentTemplate, ...}，已在模块中用过的智能体
    main_agent_id : 本模块主智能体 ID
    返回 list[dict] => [{'capability_idx': c, 'helpers':[{'agent_id': x, 'capability_idx': c}, ...]}, ...]
    """
    collab_records = []
    for cap_idx in req_set:
        if cap_idx in agent_caps:  # 主智能体已覆盖
            continue
        helpers = [
            {"agent_id": aid, "capability_idx": cap_idx}
            for aid, helper in used_agents.items()
            if aid != main_agent_id and cap_idx in getattr(helper, f"C_{cap_name}")
        ]
        if helpers:
            collab_records.append({
                "capability_idx": cap_idx,
                "helpers": helpers
            })
    return collab_records


# 主要执行函数
def run_optimized_partitioning(dag):
    # 加载数据
    csv_path = "../redundant_agent_templates.csv"
    df = pd.read_csv(csv_path)

    # 构建 AgentTemplate 列表
    agents = []
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
        agent = AgentTemplate(agent_id, C_sense, C_act, C_soft, r)
        agents.append(agent)


    # 加载任务图
    G = load_task_graph(f"../task/dag{dag}_typed.json")

    # 执行优化划分
    partitioner = OptimizedGraphPartitioner(G, agents)
    result_modules, unmatched_nodes = partitioner.partition_graph()

    # 生成结果摘要
    module_summaries = []
    used_agents = {ag.id: ag for _, ag in result_modules}
    for i, (module, agent) in enumerate(result_modules):
        C_sense, C_act, C_soft = module.get_capabilities(G)

        # === 新增：检查协作需求 ===
        # === 重新计算 sense / act 协作 ===
        collab_sense = _build_collab_records(
            C_sense, agent.C_sense, "sense", used_agents, agent.id
        )
        collab_act = _build_collab_records(
            C_act, agent.C_act, "act", used_agents, agent.id
        )

        module_info = {
            "Module ID": i + 1,
            "Agent ID": agent.id,
            "Nodes": sorted(list(module.all_nodes)),
            "Proc Nodes": sorted(list(module.proc_nodes)),
            "Sense Nodes": sorted(list(module.sense_nodes)),
            "Act Nodes": sorted(list(module.act_nodes)),
            "Entry Nodes": sorted(list(module.entry_nodes)),
            "Exit Nodes": sorted(list(module.exit_nodes)),
            "Module Size": module.size(),
            "Required Sense": sorted(C_sense),
            "Required Act": sorted(C_act),
            "Required Soft": sorted(C_soft),
            "Agent Sense": sorted(agent.C_sense),
            "Agent Act": sorted(agent.C_act),
            "Agent Soft": sorted(agent.C_soft),
            "Sense Collaboration": collab_sense,
            "Act Collaboration": collab_act
        }

        module_summaries.append(module_info)

    summary_df = pd.DataFrame(module_summaries)

    # 绘制结果
    modules_only = [module for module, _ in result_modules]
    draw_partitioned_dag_optimized(G, modules_only, f"optimized_partitioned_dag{dag}.png")

    print(f"划分完成！")
    print(f"成功划分模块数: {len(result_modules)}")
    print(f"未匹配节点数: {len(unmatched_nodes)}")

    # 验证所有节点都被分配
    total_assigned = 0
    for module, _ in result_modules:
        total_assigned += len(module.all_nodes)

    total_nodes = len(G.nodes)
    print(f"总节点数: {total_nodes}")
    print(f"已分配节点数: {total_assigned}")
    print(f"分配覆盖率: {(total_assigned / total_nodes) * 100:.1f}%")

    if unmatched_nodes:
        print(f"未匹配节点: {unmatched_nodes}")
        print("警告：仍有节点未被分配！")
    else:
        print("✓ 所有节点都已成功分配到模块中")

    # 详细验证
    all_assigned_nodes = set()
    for module, _ in result_modules:
        all_assigned_nodes.update(module.all_nodes)

    missing_nodes = set(G.nodes) - all_assigned_nodes
    if missing_nodes:
        print(f"缺失节点详情: {missing_nodes}")

    # === 计算能力冗余率 ===
    used_sense = set()
    used_act = set()
    used_soft = set()
    agent_sense = set()
    agent_act = set()
    agent_soft = set()

    for module, agent in result_modules:
        C_sense, C_act, C_soft = module.get_capabilities(G)
        used_sense.update(C_sense)
        used_act.update(C_act)
        used_soft.update(C_soft)

        agent_sense.update(agent.C_sense)
        agent_act.update(agent.C_act)
        agent_soft.update(agent.C_soft)

    total_used = len(used_sense) + len(used_act) + len(used_soft)
    total_provided = len(agent_sense) + len(agent_act) + len(agent_soft)
    total_redundant = total_provided - total_used
    redundancy_rate = (total_redundant / total_provided) * 100 if total_provided else 0.0

    print(f"\n=== 能力冗余率统计 ===")
    print(f"Agent 提供能力总数: {total_provided}")
    print(f"模块实际所需能力总数: {total_used}")
    print(f"冗余能力数量: {total_redundant}")
    print(f"冗余率: {redundancy_rate:.2f}%")

    return result_modules, unmatched_nodes, summary_df


# 运行示例
if __name__ == "__main__":
    for i in range(1, 11):
        result_modules, unmatched_nodes, summary_df = run_optimized_partitioning(i)
        # 保存 JSON 文件
        summary_df.to_json(f"agents_{i}.json", orient="records", indent=2)
        print("\n模块摘要:")
        print(summary_df.to_string(index=False))
