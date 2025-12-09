import networkx as nx
# 需要你工程里已有的这些对象/函数/常量
from SAC import TaskGraph
from sub_partiation import AgentTemplate
from SAC import COLLABORATION_PENALTY_PER_CAP, UNSOLVABLE_MISMATCH_PENALTY
import time
from collections import Counter
from statistics import mean

# ---------- 全局权重（可按需调参） ----------
GRASP_DEFAULT_WEIGHTS = {
    "w_unsolv": 10.0,
    "w_collab": 0.8,
    "w_redund": 0.60,
    "w_cut":    0.60,
    "w_size":   0.15,
    "w_module": 0.50,
}

# ===================== Stage-1: IO-Constrained GRASP + LS + LNS =====================
# 直接可用：复制到 stage1_grasp_lns_io.py 或你的现有文件
# 依赖：networkx, typing, json（标准库）, 以及你的类型/函数：
#   - AgentTemplate (具有 C_sense/C_act/C_soft 三个集合)
#   - TaskGraph     (具有属性: G: nx.DiGraph, id: int；并提供 get_dependencies()/get_sense_nodes() 等亦可)
#   - get_module_capabilities(G, nodes) -> (req_sense, req_act, req_soft)
# 如果你的项目里已存在这些函数/类型，import 对应模块即可；否则把 get_module_capabilities 也贴进去。

from dataclasses import dataclass
from typing import List, Set, Dict, Tuple, Optional, Any
import random
import json
import hashlib


# ---------------------- 权重与参数 ----------------------
@dataclass
class Weights:
    w_module: float = 0.20  # 模块个数惩罚
    w_interface: float = 1.00  # entry/exit/cut 接口规模
    w_redund: float = 0.30  # 能力冗余惩罚
    w_collab: float = 0.60  # 协作成本（缺失传感由已提交模块提供）
    w_balance: float = 0.15  # 模块大小均衡（越均匀越好）


@dataclass
class GRASPParams:
    iters: int = 30
    alpha: float = 0.3  # RCL 阈值 [0,1]，0=贪心，1=随机
    ls_iters: int = 50  # 本地搜索步数
    lns_rounds: int = 10  # LNS 轮次
    lns_destroy_frac: float = 0.25  # 每轮破坏比例（按模块数）


# ---------------------- IO 模块化：entry/exit/internal 与校验 ----------------------
def compute_entry_exit_sets(
        G: nx.DiGraph, modules: List[Set[str]]
) -> Tuple[List[Set[str]], List[Set[str]], List[Set[str]]]:
    K = len(modules)
    entries: List[Set[str]] = [set() for _ in range(K)]
    exits: List[Set[str]] = [set() for _ in range(K)]
    intern: List[Set[str]] = [set() for _ in range(K)]

    for i, M in enumerate(modules):
        for u in M:
            has_in_outside, has_out_outside = False, False
            # 入边来自模块外？
            for p in G.predecessors(u):
                if p not in M:
                    has_in_outside = True
                    break
            # 出边指向模块外？
            for s in G.successors(u):
                if s not in M:
                    has_out_outside = True
                    break
            if has_in_outside:
                entries[i].add(u)
            if has_out_outside:
                exits[i].add(u)
            if (not has_in_outside) and (not has_out_outside):
                intern[i].add(u)
    return entries, exits, intern


def is_io_partition(G: nx.DiGraph, modules: List[Set[str]]) -> bool:
    """IO 原则硬校验：内部节点不得有跨边；跨模块边必须 exit->entry。"""
    node2mod: Dict[str, int] = {}
    for i, M in enumerate(modules):
        for u in M:
            node2mod[u] = i

    entries, exits, internals = compute_entry_exit_sets(G, modules)

    # 1) 内部节点不得有跨模块边
    for i, M in enumerate(modules):
        for u in internals[i]:
            # 任一外部入/出边 → 违规
            for p in G.predecessors(u):
                if p not in M: return False
            for s in G.successors(u):
                if s not in M: return False

    # 2) 跨模块边必须由 exit 指向 entry
    for u, v in G.edges():
        i = node2mod.get(u, -1)
        j = node2mod.get(v, -1)
        if i == -1 or j == -1:  # 游离节点无需理会
            continue
        if i != j:
            if (u not in exits[i]) or (v not in entries[j]):
                return False

    return True


# ---------------------- 目标函数组件 ----------------------
def interface_cost(G: nx.DiGraph, modules: List[Set[str]], w_entry=1.0, w_exit=1.0, w_cut=0.5) -> float:
    entries, exits, _ = compute_entry_exit_sets(G, modules)
    node2mod = {}
    for i, M in enumerate(modules):
        for u in M:
            node2mod[u] = i
    cut = 0
    for u, v in G.edges():
        if node2mod.get(u, -1) != node2mod.get(v, -1):
            cut += 1
    return w_entry * sum(len(s) for s in entries) + w_exit * sum(len(s) for s in exits) + w_cut * cut


def size_balance_cost(modules: List[Set[str]]) -> float:
    """用方差衡量大小均衡；模块太不均匀要惩罚。"""
    if not modules: return 0.0
    sizes = [len(M) for M in modules if len(M) > 0]
    if not sizes: return 0.0
    mean = sum(sizes) / len(sizes)
    var = sum((s - mean) ** 2 for s in sizes) / len(sizes)
    return var


# ---------------------- 匹配与代价（协作仅来自已提交模块） ----------------------
def match_agent_for_module_with_collab(
    req_s: Set[int],
    req_a: Set[int],
    req_soft: Set[int],
    agent_lookup: Dict[int, "AgentTemplate"],
    pool_sense: Set[int],
    *,
    allow_sense_collab: bool = True,
    allow_act_collab: bool = False,
) -> Tuple[int, Dict[str, Any]]:
    """
    选择一个 agent 模板使模块三类能力都覆盖：
      - soft：必须自覆盖
      - act ：默认必须自覆盖（allow_act_collab=False）
      - sense：自身覆盖或（若允许）由 pool_sense 补齐

    返回:
      (agent_id, info)；不可行返回 (-1, {})
    """
    best_aid = -1
    best_cost = float("inf")
    best_info: Dict[str, Any] = {}

    for aid, agent in agent_lookup.items():
        # ---------- 覆盖可行性检查 ----------
        lack_soft  = req_soft - agent.C_soft
        if lack_soft:
            continue  # soft 必须自覆盖

        lack_act   = req_a - agent.C_act
        if lack_act:
            # 你目前不允许 act 协作；就算允许，这里也保持严格：仍不放行
            if not allow_act_collab:
                continue
            else:
                continue

        lack_sense = req_s - agent.C_sense
        covered_by_pool_sense: Set[int] = set()
        if lack_sense:
            if not allow_sense_collab:
                continue
            if not lack_sense.issubset(pool_sense):
                continue
            covered_by_pool_sense = set(lack_sense)  # 由协作池补齐
            lack_sense = set()

        # ---------- 计算冗余与协作并写入 info ----------
        # 自身承担的 sense 需求（扣掉协作池覆盖）
        self_sense_req = req_s - covered_by_pool_sense

        # 冗余（自身能力多于需求的部分）
        redund_soft  = len(agent.C_soft  - req_soft)
        redund_act   = len(agent.C_act   - req_a)
        redund_sense = len(agent.C_sense - self_sense_req)

        # 协作统计（sense 可协作；act 不可协作已在上面过滤）
        collab_ok  = len(covered_by_pool_sense)  # 由池补齐的 sense 项数
        collab_bad = 0                           # 这里不会出现 bad（已过滤）

        info = {
            "agent_id": aid,
            "req_soft": set(req_soft),
            "req_act": set(req_a),
            "req_sense": set(req_s),

            # 用于代价函数的关键字段（以前缺失）
            "redund_soft": redund_soft,
            "redund_act": redund_act,
            "redund_sense": redund_sense,
            "collab_ok": collab_ok,
            "collab_bad": collab_bad,

            # 记录覆盖分解，便于调试
            "covered_by_pool_sense": covered_by_pool_sense,
            "self_sense_req": self_sense_req,
        }

        # 你的 cost 使用的是下面两个函数（现在有数据可算了）
        cost = redundancy_cost(info) + collab_cost(info)

        # （可选）增加一点温和的体量惩罚，让“更精简”的模板优先
        # overcap = len(agent.C_soft | agent.C_act | agent.C_sense) - len(req_soft | req_a | self_sense_req)
        # cost += 0.01 * max(overcap, 0)

        if cost < best_cost:
            best_cost = cost
            best_aid = aid
            best_info = info

    return (best_aid, best_info) if best_aid != -1 else (-1, {})



def redundancy_cost(info: Dict[str, Any]) -> float:
    return float(info.get("redund_soft", 0) + info.get("redund_act", 0) + 0.5 * info.get("redund_sense", 0))


def collab_cost(info: Dict[str, Any], w_ok=1.0, w_bad=10.0) -> float:
    return float(w_ok * info.get("collab_ok", 0) + w_bad * info.get("collab_bad", 0))


# ---------------------- 总目标 ----------------------
def total_objective(
        G: nx.DiGraph,
        modules: List[Set[str]],
        module_agent: Dict[int, int],
        agent_lookup: Dict[int, "AgentTemplate"],
        weights: Weights
) -> float:
    """在已固定的“已提交模块顺序”下计算总目标。协作池按顺序逐步扩展。"""
    w = weights
    # 1) 接口规模
    f_interface = interface_cost(G, modules)
    # 2) 模块数
    f_module = len([M for M in modules if len(M) > 0])
    # 3) 冗余 + 协作（顺序评估）
    pool: Set[int] = set()
    f_redund, f_collab = 0.0, 0.0
    for i, M in enumerate(modules):
        if len(M) == 0:
            continue
        req_s, req_a, req_soft = get_module_capabilities(G, M)
        aid = module_agent.get(i, -1)
        if aid == -1:
            # 未匹配当作高罚（通常不会发生）
            f_collab += 100.0
            continue
        ag = agent_lookup[aid]
        # 缺口：参考匹配结果
        missing_sense = req_s - ag.C_sense
        collab_ok = len([c for c in missing_sense if c in pool])
        collab_bad = len(missing_sense) - collab_ok
        redund_soft = len(ag.C_soft - req_soft)
        redund_act = len(ag.C_act - req_a)
        redund_sense = len(ag.C_sense - req_s)

        f_redund += redund_soft + redund_act + 0.5 * redund_sense
        f_collab += (1.0 * collab_ok + 10.0 * collab_bad)

        # 扩展协作池（顺序很重要）
        pool |= set(ag.C_sense)

    # 4) 大小均衡
    f_bal = size_balance_cost(modules)

    return (w.w_module * f_module +
            w.w_interface * f_interface +
            w.w_redund * f_redund +
            w.w_collab * f_collab +
            w.w_balance * f_bal)


from typing import Set, Tuple, Iterable
import networkx as nx

def _assert_act_covered(G, modules, module_agent, agent_lookup):
    for i, M in enumerate(modules):
        req_s, req_a, req_soft = get_module_capabilities(G, M)
        aid = module_agent.get(i, -1)
        ag = agent_lookup.get(aid)
        assert ag is not None and (req_a - ag.C_act == set()), \
            f"[ACT COVERAGE] module {i} act uncovered: {req_a - (ag.C_act if ag else set())}"
import re, hashlib, json

def get_module_capabilities(
    G: nx.DiGraph,
    nodes: Iterable[str],
) -> Tuple[Set[int], Set[int], Set[int]]:
    """
    计算模块的能力需求 (req_sense, req_act, req_soft)。

    规则：
    1) 合并模块内 proc 节点显式声明的需求字段（如 req_sense/req_act/req_soft）。
    2) 识别“独占 IO”并纳入能力：
       - 对每个指向模块内 proc 的 sense 节点：若其所有后继都在该模块，则把它的传感能力并入 req_sense。
       - 对每个由模块内 proc 指向的 act 节点：若其所有前驱都在该模块，则把它的驱动能力并入 req_act。
       （这保证 act 的能力 **一定**计入 req_act，从而在匹配阶段强制 agent 自覆盖 act）
    3) 若模块集合里本身包含 sense/act（极少见），也会直接读取它们的能力并入。
    """

    def _to_int_set(x) -> Set[int]:
        """健壮解析：支持 set/list/tuple/单值；支持字符串形式（JSON、{1,2}、[1 2]、'1, 2' 等）。"""
        if x is None:
            return set()
        if isinstance(x, (set, list, tuple)):
            out = set()
            for v in x:
                try:
                    out.add(int(v))
                except Exception:
                    pass
            return out
        # 标量尝试
        try:
            return {int(x)}
        except Exception:
            pass
        # 字符串尝试 JSON + 正则
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return set()
            try:
                val = json.loads(s)
                return _to_int_set(val)
            except Exception:
                pass
            nums = re.findall(r"-?\d+", s)
            return set(int(n) for n in nums)
        return set()

    def _cap_ids_from_node(G: nx.DiGraph, u: str, *, kind: str) -> Set[int]:
        """
        从节点 u 推断“能力 id 集合”。优先顺序：
          1) 显式字段：cap / cap_id / caps / C_sense / C_act
          2) 若有整数型 idx 字段：使用 {idx}
          3) 若节点 id 里带数字：抽取所有数字作为集合（常见 'sense3' -> {3}）
          4) 最后兜底：稳定 hash 到一个非负整数（对同名节点稳定一致）
        kind ∈ {"sense","act"} 仅用于兜底 hash 的命名空间区分。
        """
        nd = G.nodes[u]
        # 1) 显式字段
        for key in ("cap", "cap_id", "caps", "C_sense", "C_act"):
            if key in nd and nd[key] is not None:
                s = _to_int_set(nd[key])
                if s:
                    return s

        # 2) idx 字段
        if "idx" in nd:
            try:
                return {int(nd["idx"])}
            except Exception:
                pass

        # 3) 从节点名提取数字
        nums = re.findall(r"-?\d+", str(u))
        if nums:
            return {int(n) for n in nums}

        # 4) 稳定 hash（避免运行间漂移）
        h = hashlib.md5(f"{kind}|{u}".encode("utf-8")).hexdigest()
        # 用 31bit 正整数。你也可以取模到更小空间，但要保证与 agent 的编码一致。
        return {int(h[:8], 16) & 0x7fffffff}
    # 支持的字段名（尽量兼容不同命名）


    IO_CAP_S_KEYS = ("cap", "cap_id", "caps", "C_sense")
    IO_CAP_A_KEYS = ("cap", "cap_id", "caps", "C_act")

    nodes = set(nodes)
    # 模块内的 proc（若传入的是 proc-only，这里等于 nodes）
    procs = {u for u in nodes if G.nodes[u].get("type") == "proc"}

    req_sense: Set[int] = set()
    req_act:   Set[int] = set()
    req_soft:  Set[int] = set()

    # 1) 先合并 proc 节点自身声明的需求
    for u in procs:
        nd = G.nodes[u]
        # proc 节点：把它的 idx 作为软件需求（保持你原有约定）
        if nd.get("type") == "proc" and "idx" in nd:
            try:
                req_soft.add(int(nd["idx"]))
            except Exception:
                pass

    # 2) 识别并纳入“独占 IO”
    #   独占 sense：来自该 sense 的所有后继都在本模块（且至少有一个后继）
    cand_sense = set()
    for u in procs:
        for s in G.predecessors(u):
            if G.nodes[s].get("type") == "sense":
                cand_sense.add(s)
    for s in cand_sense:
        succs = set(G.successors(s))
        if succs and succs.issubset(procs if procs else nodes):
            # 并入能力
            cap_set = set()
            nds = G.nodes[s]
            for k in IO_CAP_S_KEYS:
                cap_set |= _to_int_set(nds.get(k))
            # 如果依旧为空，兜底由节点自身推断
            if not cap_set:
                cap_set = _cap_ids_from_node(G, s, kind="sense")
            req_sense |= cap_set

    #   独占 act：该 act 的所有前驱都在本模块（且至少有一个前驱）
    cand_act = set()
    for u in procs:
        for a in G.successors(u):
            if G.nodes[a].get("type") == "act":
                cand_act.add(a)
    for a in cand_act:
        preds = set(G.predecessors(a))
        if preds and preds.issubset(procs if procs else nodes):
            cap_set = set()
            nda = G.nodes[a]
            for k in IO_CAP_A_KEYS:
                cap_set |= _to_int_set(nda.get(k))
            if not cap_set:
                # ✅ 关键修复：当没有 cap/caps/C_act 等字段时，回退到 idx 等
                cap_set = _cap_ids_from_node(G, a, kind="act")
            req_act |= cap_set

    # 3) 兜底：如果模块集合里已包含 sense/act，自身能力也计入
    for u in nodes:
        t = G.nodes[u].get("type")
        if t == "sense":
            caps = set()
            for k in IO_CAP_S_KEYS:
                caps |= _to_int_set(G.nodes[u].get(k))
            req_sense |= caps
        elif t == "act":
            cap_set = set()
            for k in IO_CAP_A_KEYS:
                cap_set |= _to_int_set(G.nodes[u].get(k))
            if not cap_set:
                cap_set = _cap_ids_from_node(G, u, kind="act")
            req_act |= cap_set

    return req_sense, req_act, req_soft


def _io_ok(G, modules: List[Set[str]]) -> bool:
    ok, _ = check_proc_only_partition_strict(G, modules)
    return ok
def _assert_proc_only_partition_strict(G, modules):
    ok, msg = check_proc_only_partition_strict(G, modules)
    assert ok, f"[IO STRICT] {msg}"
# ---------------------- GRASP 构造（IO 约束 + 大小上限 + 协作顺序） ----------------------
def grasp_construct_io(
        G: nx.DiGraph,
        agent_lookup: Dict[int, "AgentTemplate"],
        max_module_size: int,
        params: GRASPParams
) -> Tuple[List[Set[str]], Dict[int, int]]:
    """
    构造顺序：按提交先后为 0..K-1；协作池只来自已提交模块。
    规则：模块仅含 proc；soft/act 必须自覆盖；sense 允许从协作池补齐（可改为 False）。
    """

    # 仅对 proc 做划分
    unassigned = {n for n in G.nodes() if G.nodes[n].get("type") == "proc"}
    modules: List[Set[str]] = []
    module_agent: Dict[int, int] = {}
    sense_pool: Set[int] = set()  # 已提交模块提供的传感能力池

    alpha = getattr(params, "alpha", 0.25)
    watchdog_mult = 5  # 生长看门狗系数
    while unassigned:
        # 选种子（都为 proc 了）
        seed = next(iter(unassigned))
        cur = {seed}
        unassigned.remove(seed)  # 立刻移除，防止重复挑中
        # 匹配当前候选模块（严格：软/驱动必须覆盖；缺传感只允许来自 sense_pool）
        req_s, req_a, req_soft = get_module_capabilities(G, cur)
        aid, info = match_agent_for_module_with_collab(req_s, req_a, req_soft, agent_lookup, sense_pool)
        if aid == -1:
            # 单点都不可行 -> 强制新开模块失败；尝试把种子当单模块但不可行时，只能放宽策略
            continue
        best_score = redundancy_cost(info) + collab_cost(info)
        steps = 0

        # --- 生长（RCL） ---
        improved = True
        # FIX: 记录增长前的大小，若轮次结束时大小未变，则退出
        while improved and len(cur) < max_module_size and steps < watchdog_mult * max_module_size:
            steps += 1
            prev_size = len(cur)  # FIX
            candidates = set()
            # 候选：邻接 ∩ 未分配 ∩ proc；排除已在 cur 的点
            for u in cur:
                candidates |= set(G.predecessors(u))
                candidates |= set(G.successors(u))
            candidates &= unassigned
            candidates -= cur  # FIX: 明确排除已在 cur 的节点
            candidates = {c for c in candidates if G.nodes[c].get("type") == "proc"}
            if not candidates:
                break

            scored: List[Tuple[float, str, int, Dict[str, Any]]] = []
            for c in candidates:
                tmp = set(cur) | {c}
                # IO 约束（对“当前模块 + 已提交模块 + 其他未分配视为外部”做局部检查）
                # 这里仅确保“当前模块内部节点对外零边”的强约束：把 tmp 看成一个模块，检查内部节点合法性
                if len(tmp) > max_module_size:
                    continue
                # 构造临时模块集：已提交模块 + tmp +（忽略剩余未分配）
                temp_modules = [set(M) for M in modules] + [tmp]
                if not _io_ok(G, temp_modules):
                    continue

                req_s2, req_a2, req_soft2 = get_module_capabilities(G, tmp)
                aid2, info2 = match_agent_for_module_with_collab(req_s2, req_a2, req_soft2, agent_lookup, sense_pool)
                if aid2 == -1:
                    continue
                cost2 = redundancy_cost(info2) + collab_cost(info2)
                scored.append((cost2, c, aid2, info2))

            if not scored:
                break

            scored.sort(key=lambda x: x[0])
            best, worst = scored[0][0], scored[-1][0]
            thr = best + alpha * (worst - best)
            rcl = [item for item in scored if item[0] <= thr] or [scored[0]]  # 兜底
            pick_cost, pick_node, pick_aid, pick_info = random.choice(rcl)

            # GRASP 允许轻微非改进；这里保留你的策略
            cur.add(pick_node)
            # FIX: 立刻把新加入的节点从未分配集中移除，避免重复挑中
            if pick_node in unassigned:
                unassigned.remove(pick_node)  # FIX
            # 若没有改进，也可以接受（GRASP 通常允许轻微非改进），但我们偏保守：
            if pick_cost <= best_score:
                cur.add(pick_node)
                best_score = pick_cost
                aid = pick_aid
                info = pick_info
            # FIX: 若本轮没有让模块变大，则退出，防止死循环
            improved = (len(cur) > prev_size)  # FIX
        # ---- 提交前的最终匹配：确保当前 cur 的最终形态真的可覆盖 ----
        req_sF, req_aF, req_softF = get_module_capabilities(G, cur)
        aidF, infoF = match_agent_for_module_with_collab(
            req_sF, req_aF, req_softF, agent_lookup, sense_pool,
            allow_sense_collab=True, allow_act_collab=False  # 维持你的规则：act 不许协作
        )
        if aidF == -1:
            # 极端兜底：回退到最小可行子集（保守地只留种子）
            cur = {next(iter(cur))}
            req_sF, req_aF, req_softF = get_module_capabilities(G, cur)
            aidF, infoF = match_agent_for_module_with_collab(
                req_sF, req_aF, req_softF, agent_lookup, sense_pool,
                allow_sense_collab=True, allow_act_collab=False
            )
            if aidF == -1:
                # 罕见：直接放弃该模块，进入下一轮
                continue
                # 提交该模块
        mid = len(modules)
        modules.append(cur)
        # ------------------- 修正如下 -------------------
        module_agent[mid] = aidF
        sense_pool |= set(agent_lookup[aidF].C_sense)

    return modules, module_agent


def _exclusive_io_nodes_for_module(G, proc_set: Set[str]) -> Tuple[Set[str], Set[str]]:
    """
    返回 (sense_in, act_out)：
      - sense_in：所有指向模块内 proc 的 sense 节点，且它们的所有后继都在该模块内（独占）
      - act_out ：所有由模块内 proc 指向的 act 节点，且它们的所有前驱都在该模块内（独占）
    """
    sense_in, act_out = set(), set()

    # 入口 sense（入度=0，后继全部在模块内）
    cand_sense = set()
    for u in proc_set:
        for s in G.predecessors(u):
            if G.nodes[s].get("type") == "sense":
                cand_sense.add(s)
    for s in cand_sense:
        succs = set(G.successors(s))
        if succs and succs.issubset(proc_set):
            sense_in.add(s)

    # 出口 act（出度=0，前驱全部在模块内）
    cand_act = set()
    for u in proc_set:
        for a in G.successors(u):
            if G.nodes[a].get("type") == "act":
                cand_act.add(a)
    for a in cand_act:
        preds = set(G.predecessors(a))
        if preds and preds.issubset(proc_set):
            act_out.add(a)

    return sense_in, act_out


def make_full_modules_from_proc(G, modules_proc: List[Set[str]]) -> List[Set[str]]:
    """
    将“仅含 proc 的模块列表”扩展为“包含 sense/act 的完整模块”：
    对每个模块 i，添加其独占的入口 sense 与出口 act。
    """
    full = []
    for M in modules_proc:
        s_in, a_out = _exclusive_io_nodes_for_module(G, M)
        full.append(set(M) | s_in | a_out)
    return full



# ---------------------- 本地搜索（IO 约束 + 顺序协作） ----------------------
def local_search_improve(
        G: nx.DiGraph,
        modules: List[Set[str]],
        module_agent: Dict[int, int],
        agent_lookup: Dict[int, "AgentTemplate"],
        max_module_size: int,
        weights: Weights,
        iters: int
) -> Tuple[List[Set[str]], Dict[int, int]]:
    """
    邻域：单点移动（i->j）与双点交换（i<->j）。
    约束：IO（proc-only）、大小上限、动作后两个模块都非空。
    顺序固定，重匹配使用 rematch_sequential（顺序影响协作池）。
    """
    def eval_obj(MS, MA):
        return total_objective(G, MS, MA, agent_lookup, weights)

    best_MS = [set(M) for M in modules]
    best_MA = dict(module_agent)
    best_val = eval_obj(best_MS, best_MA)

    K = len(best_MS)
    for _ in range(iters):
        improved = False

        # 1) 单点移动
        for i in range(K):
            if len(best_MS[i]) <= 1:
                continue
            for j in range(K):
                if i == j:
                    continue
                for u in list(best_MS[i]):
                    if len(best_MS[j]) + 1 > max_module_size:
                        continue
                    MS2 = [set(M) for M in best_MS]
                    MS2[i].remove(u)
                    MS2[j].add(u)
                    if len(MS2[i]) == 0:   # 非空约束
                        continue
                    if not _io_ok(G, MS2):
                        continue
                    MA2 = dict(best_MA)
                    if not rematch_sequential(G, MS2, MA2, agent_lookup):
                        continue
                    v2 = eval_obj(MS2, MA2)
                    if v2 < best_val:
                        best_MS, best_MA, best_val = MS2, MA2, v2
                        improved = True
                        break
                if improved: break
            if improved: break
        if improved:
            continue

        # 2) 双点交换
        for i in range(K):
            for j in range(i + 1, K):
                for u in list(best_MS[i]):
                    for v in list(best_MS[j]):
                        if (len(best_MS[i]) - 1) < 1 or (len(best_MS[j]) - 1) < 1:
                            continue
                        # 交换不会改变两侧大小上限；但可保留这一行作为示例
                        MS2 = [set(M) for M in best_MS]
                        MS2[i].remove(u); MS2[j].add(u)
                        MS2[j].remove(v); MS2[i].add(v)
                        if not _io_ok(G, MS2):
                            continue
                        MA2 = dict(best_MA)
                        if not rematch_sequential(G, MS2, MA2, agent_lookup):
                            continue
                        v2 = eval_obj(MS2, MA2)
                        if v2 < best_val:
                            best_MS, best_MA, best_val = MS2, MA2, v2
                            improved = True
                            break
                    if improved: break
                if improved: break
            if improved: break

        if not improved:
            break

    return best_MS, best_MA



def rematch_sequential(
        G: nx.DiGraph,
        modules: List[Set[str]],
        module_agent: Dict[int, int],
        agent_lookup: Dict[int, "AgentTemplate"]
) -> bool:
    """
    按模块顺序重匹配（覆盖原 module_agent），
    协作池仅来自“之前已匹配成功”的模块。
    """
    pool: Set[int] = set()
    for i, M in enumerate(modules):
        if len(M) == 0:
            module_agent[i] = -1
            continue
        req_s, req_a, req_soft = get_module_capabilities(G, M)
        aid, info = match_agent_for_module_with_collab(req_s, req_a, req_soft, agent_lookup, pool)
        if aid == -1:
            return False
        module_agent[i] = aid
        pool |= set(agent_lookup[aid].C_sense)
    return True

def is_io_partition_proc_only(G: nx.DiGraph, modules: List[Set[str]]) -> bool:
    """
    模块仅含 proc；仅 entry/exit 可以与外部相连；内部节点不得与外部相连。
    """
    for M in modules:
        # 必须全是 proc
        for u in M:
            if G.nodes[u].get("type") != "proc":
                return False

        # 计算 entry / exit
        entries, exits = set(), set()
        for u in M:
            if any(v not in M for v in G.predecessors(u)):
                entries.add(u)
            if any(w not in M for w in G.successors(u)):
                exits.add(u)
        internals = M - entries - exits

        # 内部节点不得有任何对外连边
        for u in internals:
            if any(v not in M for v in G.predecessors(u)): return False
            if any(w not in M for w in G.successors(u)):   return False
    return True

def compute_entry_exit_sets_proc_only(
    G: nx.DiGraph,
    modules: List[Set[str]],
) -> Tuple[Dict[int, Set[str]], Dict[int, Set[str]], Dict[int, Set[str]]]:
    entries, exits, internals = {}, {}, {}
    for i, Mi in enumerate(modules):
        e_in, e_out = set(), set()
        for u in Mi:
            if G.nodes[u].get("type") != "proc":
                raise ValueError(f"Module {i} contains non-proc node: {u}")
            # 任何来自模块外的入边 => entry（不区分对方是 proc/sense/act）
            for v in G.predecessors(u):
                if v not in Mi:
                    e_in.add(u)
                    break
            # 任何指向模块外的出边 => exit（不区分对方是 proc/sense/act）
            for w in G.successors(u):
                if w not in Mi:
                    e_out.add(u)
                    break
        entries[i]   = e_in
        exits[i]     = e_out
        internals[i] = Mi - e_in - e_out
    return entries, exits, internals


def check_proc_only_partition_strict(
    G: nx.DiGraph,
    modules: List[Set[str]],
) -> Tuple[bool, str]:
    entries, exits, internals = compute_entry_exit_sets_proc_only(G, modules)
    node2mod = {u: i for i, Mi in enumerate(modules) for u in Mi}

    # A) internal 节点不得有任何跨模块边（无论对方类型）
    for i, Mi in enumerate(modules):
        for u in internals[i]:
            for v in G.predecessors(u):
                if v not in Mi:
                    return False, f"[A] module {i} internal node {u} has external predecessor {v}"
            for w in G.successors(u):
                if w not in Mi:
                    return False, f"[A] module {i} internal node {u} has external successor {w}"

    # B) 任何跨模块边都必须满足 exit->entry
    for u, v in G.edges():
        i = node2mod.get(u); j = node2mod.get(v)
        if i is None or j is None or i == j:
            continue
        if (u not in exits[i]) or (v not in entries[j]):
            return False, f"[B] cross edge {u}->{v} invalid: u∈exits[{i}]? {u in exits[i]} ; v∈entries[{j}]? {v in entries[j]}"

    return True, "OK"

# ---------------------- LNS：破坏-修复（保序） ----------------------
def lns_improve(
        G: nx.DiGraph,
        modules: List[Set[str]],
        module_agent: Dict[int, int],
        agent_lookup: Dict[int, "AgentTemplate"],
        max_module_size: int,
        weights: Weights,
        params: GRASPParams
) -> Tuple[List[Set[str]], Dict[int, int]]:
    """
    固定顺序的 LNS：每轮选择窗口，回收窗口内节点，仅对 proc 重建；
    前缀协作池固定；随后本地搜索微调。
    """
    def eval_obj(MS, MA):
        return total_objective(G, MS, MA, agent_lookup, weights)

    best_MS = [set(M) for M in modules]
    best_MA = dict(module_agent)
    best_val = eval_obj(best_MS, best_MA)

    K = len(best_MS)
    rounds = getattr(params, "lns_rounds", 200)
    frac = getattr(params, "lns_destroy_frac", 0.25)
    watchdog_mult = 5

    for _ in range(rounds):
        if K <= 1:
            break
        win = max(1, int(frac * K))
        start = random.randint(0, K - win)
        end = start + win  # [start, end)

        prefix = [set(M) for M in best_MS[:start]]
        window = [set(M) for M in best_MS[start:end]]
        suffix = [set(M) for M in best_MS[end:]]
        if not window:
            continue

        # 前缀协作池
        pool: Set[int] = set()
        for i, _M in enumerate(prefix):
            aid0 = best_MA.get(i, -1)
            if aid0 != -1:
                pool |= set(agent_lookup[aid0].C_sense)

        # 回收窗口节点，仅保留 proc 参与重建
        recovered = set().union(*window)
        unassigned = {n for n in recovered if G.nodes[n].get("type") == "proc"}

        rebuilt_modules: List[Set[str]] = []
        rebuilt_agents: Dict[int, int] = {}

        while unassigned:
            seed = next(iter(unassigned))
            cur = {seed}
            unassigned.remove(seed)  # 立刻移除 seed

            req_s, req_a, req_soft = get_module_capabilities(G, cur)
            aid, info = match_agent_for_module_with_collab(
                req_s, req_a, req_soft, agent_lookup, pool,
                allow_sense_collab=True, allow_act_collab=False
            )
            if aid == -1:
                # 不可行，丢弃该点
                continue

            steps = 0
            while len(cur) < max_module_size and steps < watchdog_mult * max_module_size:
                steps += 1
                cand = set()
                for u in cur:
                    cand |= set(G.predecessors(u))
                    cand |= set(G.successors(u))
                cand &= unassigned
                cand -= cur
                cand = {c for c in cand if G.nodes[c].get("type") == "proc"}
                if not cand:
                    break

                picked = None
                best_delta = float("inf")
                for c in cand:
                    tmp = set(cur) | {c}
                    temp_modules = [*prefix, *rebuilt_modules, tmp, *suffix]
                    if not _io_ok(G, temp_modules):
                        continue
                    req_s2, req_a2, req_soft2 = get_module_capabilities(G, tmp)
                    aid2, info2 = match_agent_for_module_with_collab(
                        req_s2, req_a2, req_soft2, agent_lookup, pool,
                        allow_sense_collab=True, allow_act_collab=False
                    )
                    if aid2 == -1:
                        continue
                    score = redundancy_cost(info2) + collab_cost(info2)
                    if score < best_delta:
                        best_delta = score
                        picked = (c, aid2)
                if picked is None:
                    break
                c, aid = picked
                cur.add(c)
                unassigned.remove(c)

            # 双保险：提交前把整个模块从未分配集中清除
            unassigned -= cur

            rebuilt_modules.append(cur)
            rebuilt_agents[len(prefix) + len(rebuilt_modules) - 1] = aid
            pool |= set(agent_lookup[aid].C_sense)

        # 空重建则跳过
        if not rebuilt_modules:
            continue

        # 拼接与重匹配
        MS2 = [*prefix, *rebuilt_modules, *suffix]
        MA2 = dict(best_MA)
        MA2.update(rebuilt_agents)

        if not _io_ok(G, MS2):
            continue
        if not rematch_sequential(G, MS2, MA2, agent_lookup):
            continue

        # 局部搜索微调
        MS2, MA2 = local_search_improve(G, MS2, MA2, agent_lookup, max_module_size, weights, iters=200)
        v2 = eval_obj(MS2, MA2)
        if v2 < best_val:
            best_MS, best_MA, best_val = MS2, MA2, v2

    return best_MS, best_MA


def analyze_sense_collaboration(
        G: nx.DiGraph,
        modules: List[Set[str]],
        module_agent: Dict[int, int],
        agent_lookup: Dict[int, "AgentTemplate"],
):
    """
    返回一个字典，按模块给出：
      - self_caps: 该模块智能体自身具备的传感能力（用于本模块）
      - required: 该模块根据节点需求汇总的传感能力全集
      - collab_from: 列出缺失的传感能力由哪个(更早)模块/智能体提供
      - unresolved: 仍无法被任何更早模块覆盖的传感能力（潜在问题）

    协作只能来自更早的模块（顺序即 modules 列表下标）。
    """
    out = {"by_module": [], "unresolved_any": []}

    # 先做一遍“前缀池”：到每个模块 i-1 为止，可用的 sense -> 最早提供者模块/智能体
    earliest_provider = {}  # cap -> (provider_module_id, provider_agent_id)
    providers_by_module = []  # 记录每个模块新增了哪些cap

    for i, M in enumerate(modules):
        aid = module_agent.get(i, -1)
        ag = agent_lookup.get(aid)
        if ag is None:
            providers_by_module.append(set())
            continue
        new_caps = set(ag.C_sense) - set(c for c in earliest_provider.keys())
        # 注意：上面是“相对最早”的新增；我们更常用“全部加入池子”，但 earliest 只记录第一次看到的提供者
        for cap in ag.C_sense:
            if cap not in earliest_provider:
                earliest_provider[cap] = (i, aid)
        providers_by_module.append(set(new_caps))

    # 正式逐模块分析（只允许从更早模块找提供者）
    for i, M in enumerate(modules):
        aid = module_agent.get(i, -1)
        ag = agent_lookup.get(aid)
        req_s, req_a, req_soft = get_module_capabilities(G, M)
        have_s = set() if ag is None else set(ag.C_sense)

        missing = req_s - have_s
        collab_from = []
        unresolved = []

        # 只能从更早模块找提供者
        # 为了只用更早模块，构建一个“到 i-1 为止的最早提供者表”
        earliest_before_i = {}
        for cap, (pm, pa) in earliest_provider.items():
            if pm < i:  # 只取更早的
                earliest_before_i[cap] = (pm, pa)

        for cap in sorted(missing):
            if cap in earliest_before_i:
                pm, pa = earliest_before_i[cap]
                collab_from.append({
                    "cap": int(cap),
                    "provider_module": int(pm),
                    "provider_agent": int(pa),
                })
            else:
                unresolved.append(int(cap))

        out["by_module"].append({
            "module_id": int(i),
            "agent_id": int(aid),
            "self_caps": sorted(int(x) for x in have_s),
            "required": sorted(int(x) for x in req_s),
            "collab_from": collab_from,  # 本模块需要由别人补的cap来自谁
            "unresolved": unresolved,  # 没人能补到的cap（潜在问题）
        })

        # 可选：你也可以把本模块实际“贡献给后继”的cap记下来（= have_s）
        # 但上面 providers_by_module / earliest_provider 已经能看到了

        # 收集全局未解决
        if unresolved:
            out["unresolved_any"].extend([(i, c) for c in unresolved])

    return out


def print_sense_collaboration_report(collab_info: dict):
    """
    漂亮地打印 analyze_sense_collaboration 的结果。
    """
    print("\n=== Sense Collaboration Report ===")
    for item in collab_info["by_module"]:
        mid = item["module_id"];
        aid = item["agent_id"]
        req = item["required"]
        self_caps = item["self_caps"]
        need = sorted(set(req) - set(self_caps))
        print(f"\n[Module {mid}] Agent {aid}")
        print(f"  - Required sense: {req}")
        print(f"  - Self-provided : {self_caps}")
        if not need:
            print("  - Missing sense : [] (no collaboration needed)")
        else:
            print(f"  - Missing sense : {need}")
            if item["collab_from"]:
                print("  - Collaborated from earlier modules:")
                for c in item["collab_from"]:
                    print(f"      cap {c['cap']}  <=  Module {c['provider_module']} (Agent {c['provider_agent']})")
            if item["unresolved"]:
                print(f"  - Unresolved    : {item['unresolved']}  <-- ⚠ 无人可提供（需调整划分/模板）")

    if collab_info["unresolved_any"]:
        print("\n[WARN] Unresolved caps exist:")
        for (mid, cap) in collab_info["unresolved_any"]:
            print(f"  - Module {mid} cap {cap}")
    else:
        print("\nAll missing senses are collaboratively satisfied by earlier modules. ✅")


# ---------------------- 统一入口：Stage-1 求解 ----------------------
def solve_stage1_partition_and_match(
        task_graph: "TaskGraph",
        agent_lookup: Dict[int, "AgentTemplate"],
        max_module_size: int = 8,
        weights: Optional[Weights] = None,
        params: Optional[GRASPParams] = None,
        seed: Optional[int] = None,
        verbose: bool = False,  # <--- 新增

) -> Dict[str, Any]:
    """
    返回字典：
      {
        "task_id": ...,
        "modules": [
          {
            "module_id": i,
            "nodes": [...],
            "agent_id": ...,
            "entry": [...], "exit": [...], "internal": [...]
          }, ...
        ],
        "objective": <float>
      }
    """
    if seed is not None:
        random.seed(seed)

    w = weights or Weights()
    p = params or GRASPParams()

    G = task_graph.G

    task_id = getattr(task_graph, "id", -1)

    if verbose:
        print(f"\n[Stage-1][Task {task_id}] ===== 开始 Stage-1 求解 =====")

    # 1) 初解（GRASP）
    if verbose:
        print(f"[Stage-1][Task {task_id}] Step 1: 运行 GRASP 构造初解...")
    t0 = time.perf_counter()
    # 1) 初解（GRASP）
    modules, module_agent = grasp_construct_io(G, agent_lookup, max_module_size, p)
    t1 = time.perf_counter()
    if verbose:
        print(f"[Stage-1][Task {task_id}] Step 1 完成: 模块={len(modules)}, "
              f"耗时={t1 - t0:.3f}s")
        obj1, p1 = objective_breakdown(G, modules, module_agent, agent_lookup, w)
        print(f"[Stage-1][Task {task_id}] Step 1 OBJ = {obj1:.3f} "
              f"(module={p1['module']:.2f}, interface={p1['interface']:.2f}, "
              f"redund={p1['redund']:.2f}, collab={p1['collab']:.2f}, balance={p1['balance']:.2f})")
    # 2) 本地搜索改进
    if verbose:
        print(f"[Stage-1][Task {task_id}] Step 2: 本地搜索改进...")
        obj2, p2 = objective_breakdown(G, modules, module_agent, agent_lookup, w)
        print(f"[Stage-1][Task {task_id}] Step 2 OBJ = {obj2:.3f} "
              f"(module={p2['module']:.2f}, interface={p2['interface']:.2f}, "
              f"redund={p2['redund']:.2f}, collab={p2['collab']:.2f}, balance={p2['balance']:.2f})")
    # 2) 本地搜索
    modules, module_agent = local_search_improve(G, modules, module_agent, agent_lookup, max_module_size, w, p.ls_iters)
    t2 = time.perf_counter()
    if verbose:
        print(f"[Stage-1][Task {task_id}] Step 2 完成: 模块={len(modules)}, "
              f"耗时={t2 - t1:.3f}s")
    # 3) LNS 改进
    # 3) LNS 改进
    if verbose:
        print(f"[Stage-1][Task {task_id}] Step 3: LNS 破坏-修复...")
    modules, module_agent = lns_improve(G, modules, module_agent, agent_lookup, max_module_size, w, p)
    t3 = time.perf_counter()
    if verbose:
        print(f"[Stage-1][Task {task_id}] Step 3 完成: 模块={len(modules)}, "
              f"耗时={t3 - t2:.3f}s")
    # 生成详情
    # entries, exits, internals = compute_entry_exit_sets(G, modules)
    obj = total_objective(G, modules, module_agent, agent_lookup, w)
    modules_full = make_full_modules_from_proc(G, modules)
    entries, exits, internals = compute_entry_exit_sets_proc_only(G, modules)
    # --- NEW: 详细摘要输出（可开关） ---
    if verbose:
        sizes = [len(M) for M in modules]
        over = [s for s in sizes if s > max_module_size]
        agent_used = sorted(set(module_agent.values()))
        # 模块大小分布（前 10 个）
        top_sizes = sizes[:10]
        # 简单的协作统计：若你有协作计数函数，可在此处替换为真实数据
        print(
            "[Stage-1][Task %s] modules=%d  avg_size=%.2f  max_size=%d  "
            "violations(>%d)=%d  unique_agents=%d  "
            "t_grasp=%.3fs  t_ls=%.3fs  t_lns=%.3fs  obj=%.3f"
            % (
                getattr(task_graph, "id", -1), len(modules),
                mean(sizes) if sizes else 0.0, max(sizes) if sizes else 0,
                max_module_size, len(over), len(agent_used),
                (t1 - t0), (t2 - t1), (t3 - t2), float(obj),
            )
        )
        # 大小直方图（精简打印）
        hist = Counter(sizes)
        print("[Stage-1][Task %s] size_hist=%s  sizes_head=%s"
              % (getattr(task_graph, "id", -1), dict(sorted(hist.items())), top_sizes))

    result_modules: List[Dict[str, Any]] = []
    for i, M in enumerate(modules):
        aid = module_agent.get(i, -1)
        M_full = modules_full[i]  # 使用“包含 sense/act”的集合来输出
        result_modules.append({
            "module_id": i,
            "nodes": sorted(list(M_full)),  # <--- 输出完整节点集合
            "agent_id": int(aid),
            "entry": sorted(list(entries[i])),
            "exit": sorted(list(exits[i])),
            "internal": sorted(list(internals[i])),
        })
    _assert_act_covered(G, modules, module_agent, agent_lookup)
    _assert_proc_only_partition_strict(G, modules)
    verify_and_report_partition(G, modules)
    # 假设你在 solve_stage1_partition_and_match 之后：
    collab = analyze_sense_collaboration(task_graph.G, modules, module_agent, agent_lookup)
    print_sense_collaboration_report(collab)
    return {
        "task_id": getattr(task_graph, "id", -1),
        "modules": result_modules,
        "objective": float(obj),
    }

def objective_breakdown(
    G: nx.DiGraph,
    modules: List[Set[str]],
    module_agent: Dict[int, int],
    agent_lookup: Dict[int, "AgentTemplate"],
    weights: Weights,
) -> Tuple[float, Dict[str, float]]:
    # 1) 接口规模
    f_interface = interface_cost(G, modules)
    # 2) 模块数
    f_module = len([M for M in modules if len(M) > 0])
    # 3) 冗余 + 协作（按顺序累积协作池）
    pool: Set[int] = set()
    f_redund, f_collab = 0.0, 0.0
    for i, M in enumerate(modules):
        if not M:
            continue
        req_s, req_a, req_soft = get_module_capabilities(G, M)
        aid = module_agent.get(i, -1)
        if aid == -1:
            f_collab += 100.0
            continue
        ag = agent_lookup[aid]
        missing_sense = req_s - ag.C_sense
        collab_ok = len([c for c in missing_sense if c in pool])
        collab_bad = len(missing_sense) - collab_ok
        redund_soft = len(ag.C_soft - req_soft)
        redund_act  = len(ag.C_act  - req_a)
        redund_sense= len(ag.C_sense- req_s)
        f_redund += redund_soft + redund_act + 0.5 * redund_sense
        f_collab += (1.0 * collab_ok + 10.0 * collab_bad)
        pool |= set(ag.C_sense)

    # 4) 大小均衡
    f_bal = size_balance_cost(modules)

    parts = {
        "module":   f_module,
        "interface":f_interface,
        "redund":   f_redund,
        "collab":   f_collab,
        "balance":  f_bal,
    }
    obj = (weights.w_module   * f_module +
           weights.w_interface* f_interface +
           weights.w_redund   * f_redund +
           weights.w_collab   * f_collab +
           weights.w_balance  * f_bal)
    return float(obj), parts

def verify_and_report_partition(G: nx.DiGraph, modules: List[Set[str]]) -> bool:
    ok, msg = check_proc_only_partition_strict(G, modules)
    if ok:
        print("[VERIFY] Partition OK.")
        return True
    print("[VERIFY][FAIL]", msg)

    entries, exits, internals = compute_entry_exit_sets_proc_only(G, modules)

    # 如果是 A 类（internal 有外边），打印它的邻居
    m = re.search(r"module (\d+) internal node ([^\s]+) has external (predecessor|successor)", msg)
    if m:
        i = int(m.group(1)); u = m.group(2)
        preds = list(G.predecessors(u))
        succs = list(G.successors(u))
        print(f"  -> module {i}, node={u}, preds={preds}, succs={succs}")
        print(f"  -> entries[{i}]={sorted(entries[i])}")
        print(f"  -> exits[{i}]  ={sorted(exits[i])}")
        print(f"  -> internals[{i}]={sorted(internals[i])}")
    else:
        # 如果是 B 类（跨边不是 exit->entry），定位边的两个端点和模块
        m2 = re.search(r"cross edge ([^\s]+)->([^\s]+) invalid", msg)
        if m2:
            u, v = m2.group(1), m2.group(2)
            i = next((k for k, M in enumerate(modules) if u in M), None)
            j = next((k for k, M in enumerate(modules) if v in M), None)
            print(f"  -> offending edge {u}->{v}; u in mod {i}, v in mod {j}")
            if i is not None:
                print(f"     exits[{i}]={sorted(exits[i])}")
            if j is not None:
                print(f"     entries[{j}]={sorted(entries[j])}")
    return False


# ---------------------- 序列化与落盘 ----------------------
def _dag_fingerprint(G: nx.DiGraph) -> str:
    """用于检查计划是否匹配同一 DAG（轻量指纹）"""
    h = hashlib.sha256()
    nodes = sorted(list(G.nodes()))
    edges = sorted(list(G.edges()))
    h.update(str(nodes).encode("utf-8"))
    h.update(str(edges).encode("utf-8"))
    return h.hexdigest()


def serialize_partition_plan(G: nx.DiGraph, plan: Dict[str, Any]) -> Dict[str, Any]:

    return {
        "fingerprint": _dag_fingerprint(G),
        "task_id": plan.get("task_id", -1),
        "objective": plan.get("objective", None),
        "modules": plan.get("modules", []),
    }


def build_and_save_stage1_plans(
        task_graphs: List["TaskGraph"],
        agent_lookup: Dict[int, "AgentTemplate"],
        out_path: str = "plans.json",
        max_module_size: int = 8,
        weights: Optional[Weights] = None,
        params: Optional[GRASPParams] = None,
) -> None:
    all_plans = []
    for tg in task_graphs:
        plan = solve_stage1_partition_and_match(
            tg, agent_lookup,
            max_module_size=max_module_size,
            weights=weights, params=params
        )
        ser = serialize_partition_plan(tg.G, plan)
        all_plans.append(ser)
        print(f"[Stage-1] Task {tg.id}: modules={len(plan['modules'])}, objective={plan['objective']:.2f}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_plans, f, ensure_ascii=False, indent=2)
    print(f"[Stage-1] Saved {len(all_plans)} plans to {out_path}")


if __name__ == "__main__":
    build_and_save_stage1_plans()
