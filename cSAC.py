# ============================
# Discrete Constrained SAC (cSAC) — Stage-2 Only
# ============================
from dataclasses import dataclass
from typing import Tuple, Dict, List, cast
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import gymnasium as gym
from gymnasium import spaces

# ---------- Utils ----------

def _masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对 logits 应用二值 mask（0=禁用）后做 log_softmax。
    - logits: (B, H, N)
    - mask  : (B, H, N)  布尔或 0/1
    返回 (log_probs, probs)，对“整头全 0”做退化保护（全 0 -> 全 1）。
    """
    tiny = torch.finfo(logits.dtype).min
    mask = mask.to(torch.bool)
    all_false = (~mask).all(dim=dim, keepdim=True)              # (B,H,1)
    safe_mask = torch.where(all_false, torch.ones_like(mask), mask)
    masked_logits = torch.where(safe_mask, logits, torch.full_like(logits, tiny))
    log_probs = torch.log_softmax(masked_logits, dim=dim)
    probs = torch.exp(log_probs)
    return log_probs, probs


def _to_torch_obs(obs: Dict[str, np.ndarray], device: torch.device):
    # 将单步 obs(dict of np) 转为 torch，外加 batch 维 (1,...)
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device).unsqueeze(0) for k, v in obs.items()}

def _to_torch_obs_batch(obs_list: List[Dict[str, np.ndarray]], device: torch.device):
    # 将 list[obs_dict] 合并为 batched torch dict
    keys = obs_list[0].keys()
    out = {}
    for k in keys:
        vs = [torch.as_tensor(obs[k], dtype=torch.float32, device=device).unsqueeze(0) for obs in obs_list]
        out[k] = torch.cat(vs, dim=0)
    return out

# ---------- Replay Buffer ----------

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.ptr = 0
        self.full = False
        self.data = []

    def add(self, obs, action, reward, next_obs, done, costs):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.ptr] = (obs, action, reward, next_obs, done, costs)
        self.ptr = (self.ptr + 1) % self.capacity
        self.full = self.full or self.ptr == 0

    def __len__(self):
        return len(self.data) if not self.full else self.capacity

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, len(self), size=batch_size)
        batch = [self.data[i] for i in idxs]
        return list(zip(*batch))  # (obs_b, act_b, rew_b, next_obs_b, done_b, costs_b)

# ---------- Backbone (Stage-2 flat obs only) ----------

class FlatObsBackbone(nn.Module):
    """
    仅基于第二阶段需要的平铺特征构建共享表示：
    - device_resources (D,6)
    - constraints (8,)
    注：mask（compute/sense/act）不参与编码，只用于动作屏蔽。
    """
    def __init__(self, observation_space: spaces.Dict, hid: int = 256):
        super().__init__()
        self.obs_space = observation_space

        # 计算平铺后的维度（不含 mask）
        flat_dim = 0
        for k, sp in observation_space.spaces.items():
            if k not in ["compute_mask", "sense_mask", "act_mask"]:
                flat_dim += spaces.utils.flatdim(sp)

        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU()
        )
        self.value_head = nn.Linear(hid, 1)
        self.latent_dim = hid

    def _flatten_obs(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        flats = []
        for k, t in obs.items():
            if k not in ["compute_mask", "sense_mask", "act_mask"]:
                flats.append(torch.flatten(t, start_dim=1))
        if not flats:
            bsz = next(iter(obs.values())).shape[0]
            flats = [torch.zeros((bsz, 1), device=next(self.parameters()).device)]
        return torch.cat(flats, dim=1)

    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._flatten_obs(obs)   # (B, flat_dim)
        h = self.encoder(x)          # (B, hid)
        value = self.value_head(h)   # (B, 1)
        return h, value

# ---------- Actor / Critics (multi-head discrete) ----------

class MultiHeadPolicySAC(nn.Module):
    """
    多头离散策略：输出 (heads, num_devices) 的 logits，逐头 masked softmax。
    使用 FlatObsBackbone（Stage-2）。
    """
    def __init__(self, observation_space: spaces.Dict, action_space: spaces.MultiDiscrete, hid: int = 256):
        super().__init__()
        assert isinstance(action_space, spaces.MultiDiscrete)
        # ✅ 记录每个 head 的候选数（[T, D, I, I, ..., I]）
        self.head_sizes = [int(x) for x in action_space.nvec.tolist()] if hasattr(action_space.nvec, "tolist") else [
            int(x) for x in action_space.nvec]
        self.num_heads = len(self.head_sizes)
        # ✅ 统一的候选槽位上限：所有 head 的最大候选数
        self.N_max = int(max(self.head_sizes))

        self.encoder = FlatObsBackbone(observation_space, hid=hid)
        # ✅ 让每个 head 都输出 N_max 个 logits，(B, H * N_max) -> (B, H, N_max)
        self.policy_head = nn.Linear(self.encoder.latent_dim, self.num_heads * self.N_max)

    @torch.no_grad()
    def act(self, obs: Dict[str, torch.Tensor], deterministic: bool = False):
        h, _ = self.encoder(obs)  # (B,Hid)
        logits = self.policy_head(h).view(-1, self.num_heads, self.N_max)  # (B,H,N_max)

        # ---- 取观测中的掩码 ----
        task_mask = obs["task_mask"]  # (B, T)
        comp_maskT = obs["compute_mask"]  # (B, T, D)
        sense_maskT = obs["sense_mask"]  # (B, T, K_s, I)
        act_maskT = obs["act_mask"]  # (B, T, K_a, I)

        B, T = task_mask.shape
        D = comp_maskT.shape[-1]
        K_s = sense_maskT.shape[2]
        I = sense_maskT.shape[-1]
        K_a = act_maskT.shape[2]

        # 1) 在任务维上汇总（OR/Max）
        comp_mask_any = comp_maskT.max(dim=1).values  # (B, D)
        sense_mask_any = sense_maskT.max(dim=1).values  # (B, K_s, I)
        act_mask_any = act_maskT.max(dim=1).values  # (B, K_a, I)
        task_mask_head = task_mask.unsqueeze(1).to(comp_maskT.dtype)  # (B, 1, T)

        # 2) 右侧 0 填充到统一长度 N_max（= self.N_max）
        N_max = self.N_max

        def pad_last(x, Nmax):
            pad = Nmax - x.shape[-1]
            if pad <= 0:
                return x
            return torch.nn.functional.pad(x, (0, pad), value=0.0)

        task_mask_pad = pad_last(task_mask_head, N_max)  # (B, 1,    N_max)
        comp_mask_pad = pad_last(comp_mask_any.unsqueeze(1), N_max)  # (B, 1,    N_max)
        sense_mask_pad = pad_last(sense_mask_any, N_max)  # (B, K_s,  N_max)
        act_mask_pad = pad_last(act_mask_any, N_max)  # (B, K_a,  N_max)

        # 3) 拼 head 维：顺序需与动作空间一致 -> [task, compute, sense... , act...]
        full_mask = torch.cat([task_mask_pad, comp_mask_pad, sense_mask_pad, act_mask_pad], dim=1)  # (B,H,N_max)

        # ---- masked softmax 采样 ----
        logp, probs = _masked_log_softmax(logits, full_mask, dim=-1)

        if deterministic:
            a = probs.argmax(dim=-1)  # (B,H)
        else:
            B, H, N = probs.shape
            a = torch.zeros(B, H, dtype=torch.long, device=probs.device)
            for h_id in range(H):
                a[:, h_id] = torch.distributions.Categorical(probs[:, h_id, :]).sample()

        picked = logp.gather(dim=-1, index=a.unsqueeze(-1)).squeeze(-1)  # (B,H)
        log_prob = picked.sum(dim=1)  # (B,)
        return a.cpu().numpy(), log_prob.cpu().numpy()

    def dist_all(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        h, _ = self.encoder(obs)
        logits = self.policy_head(h).view(-1, self.num_heads, self.N_max)  # (B,H,N_max)

        task_mask = obs["task_mask"]  # (B, T)
        comp_maskT = obs["compute_mask"]  # (B, T, D)
        sense_maskT = obs["sense_mask"]  # (B, T, K_s, I)
        act_maskT = obs["act_mask"]  # (B, T, K_a, I)

        comp_mask_any = comp_maskT.max(dim=1).values  # (B, D)
        sense_mask_any = sense_maskT.max(dim=1).values  # (B, K_s, I)
        act_mask_any = act_maskT.max(dim=1).values  # (B, K_a, I)
        task_mask_head = task_mask.unsqueeze(1).to(comp_maskT.dtype)  # (B,1,T)

        N_max = self.N_max

        def pad_last(x, Nmax):
            pad = Nmax - x.shape[-1]
            if pad <= 0:
                return x
            return torch.nn.functional.pad(x, (0, pad), value=0.0)

        task_mask_pad = pad_last(task_mask_head, N_max)  # (B,1,N_max)
        comp_mask_pad = pad_last(comp_mask_any.unsqueeze(1), N_max)  # (B,1,N_max)
        sense_mask_pad = pad_last(sense_mask_any, N_max)  # (B,K_s,N_max)
        act_mask_pad = pad_last(act_mask_any, N_max)  # (B,K_a,N_max)

        full_mask = torch.cat([task_mask_pad, comp_mask_pad, sense_mask_pad, act_mask_pad], dim=1)  # (B,H,N_max)
        logp, probs = _masked_log_softmax(logits, full_mask, dim=-1)
        return logp, probs


class TwinQHeads(nn.Module):
    def __init__(self, observation_space: spaces.Dict, action_space: spaces.MultiDiscrete, hid: int = 256):
        super().__init__()
        assert isinstance(action_space, spaces.MultiDiscrete)
        self.encoder = FlatObsBackbone(observation_space, hid=hid)

        # 与 Actor 一致的 head 尺寸设定
        self.head_sizes = [int(x) for x in (action_space.nvec.tolist()
                                            if hasattr(action_space.nvec, "tolist") else action_space.nvec)]
        self.num_heads = len(self.head_sizes)
        self.N_max     = int(max(self.head_sizes))

        device = next(self.parameters()).device
        self.q1 = nn.Linear(self.encoder.latent_dim, self.num_heads * self.N_max).to(device)
        self.q2 = nn.Linear(self.encoder.latent_dim, self.num_heads * self.N_max).to(device)

    def forward_tables(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        h, _ = self.encoder(obs)                                  # (B, hid)
        q1 = self.q1(h).view(-1, self.num_heads, self.N_max)      # (B, H, N_max)
        q2 = self.q2(h).view(-1, self.num_heads, self.N_max)      # (B, H, N_max)
        return q1, q2

# ---------- Config & Algo ----------

@dataclass
class CSACConfig:
    buffer_size: int = 200_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    alpha_init: float = 0.2
    target_entropy_per_head: float = -1.0  # 通常设为 -ln(num_devices)
    lambda_lr: float = 5e-3
    cost_limits: Dict[str, float] = None   # 例如 {"latency":0.1, "bandwidth":0.05}
    constraint_keys: List[str] = None      # 参与约束的 cost 名单
    warmup_steps: int = 10_000
    updates_per_step: int = 1
    max_grad_norm: float = 10.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ConstrainedDiscreteSAC:
    def __init__(self, env: gym.Env, cfg: CSACConfig):
        assert isinstance(env.action_space, spaces.MultiDiscrete)
        self.env = env
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # 网络
        self.policy = MultiHeadPolicySAC(env.observation_space, env.action_space).to(self.device)
        self.critic = TwinQHeads(env.observation_space, env.action_space).to(self.device)
        self.critic_targ = TwinQHeads(env.observation_space, env.action_space).to(self.device)
        self.critic_targ.load_state_dict(self.critic.state_dict())

        # 优化器
        self.opt_policy = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

        # 温度（熵系数）
        self.log_alpha = torch.tensor(np.log(cfg.alpha_init), dtype=torch.float32, device=self.device, requires_grad=True)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=cfg.lr)
        # 目标熵（按头累加）
        self.target_entropy = (cfg.target_entropy_per_head * len(env.action_space.nvec))

        # 拉格朗日乘子
        if cfg.constraint_keys is None:
            cfg.constraint_keys = ["latency", "bandwidth"]
        if cfg.cost_limits is None:
            cfg.cost_limits = {"latency": 0.20, "bandwidth": 0.20}
        self.lambdas = {k: torch.tensor(0.0, device=self.device) for k in cfg.constraint_keys}

        # 回放池
        self.replay = ReplayBuffer(cfg.buffer_size)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _compute_V(self, next_obs_t: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        离散SAC期望形式：
        V = Σ_h Σ_i π(i|s') * [ minQ(i) - α log π(i|s') ]
        返回形状 (B,)
        """
        with torch.no_grad():
            logp, probs = self.policy.dist_all(next_obs_t)           # (B,H,N)
            q1_t, q2_t = self.critic_targ.forward_tables(next_obs_t) # (B,H,N)

            def _pad_or_trunc_last(x: torch.Tensor, N: int) -> torch.Tensor:
                cur = x.shape[-1]
                if cur == N:
                    return x
                if cur < N:
                    return F.pad(x, (0, N - cur), value=0.0)
                return x[..., :N]

            with torch.no_grad():
                logp, probs = self.policy.dist_all(next_obs_t)  # (B,H,Np)
                q1_t, q2_t = self.critic_targ.forward_tables(next_obs_t)  # (B,H,Nq)
                N = probs.shape[-1]  # 以策略为准（应等于 N_max）

                logp = _pad_or_trunc_last(logp, N)
                q1_t = _pad_or_trunc_last(q1_t, N)
                q2_t = _pad_or_trunc_last(q2_t, N)

                qmin = torch.minimum(q1_t, q2_t)
                ent_term = (-self.alpha * logp)
                v = (probs * (qmin + ent_term)).sum(dim=-1).sum(dim=-1)
                return v


    def _pack_obs(self, obs_list):
        return _to_torch_obs_batch(obs_list, self.device)

    def update(self, batch):
        obs_b, act_b, rew_b, next_obs_b, done_b, costs_b = batch
        B = len(rew_b)
        obs_t      = self._pack_obs(obs_b)
        next_obs_t = self._pack_obs(next_obs_b)
        actions_t  = torch.as_tensor(np.stack(act_b), dtype=torch.long, device=self.device)   # (B,H)
        rewards_t  = torch.as_tensor(np.asarray(rew_b, dtype=np.float32), device=self.device) # (B,)
        dones_t    = torch.as_tensor(np.asarray(done_b, dtype=np.float32), device=self.device)

        # 约束成本的 batch 平均（用于更新拉格朗日乘子）
        avg_costs = {}
        for k in self.cfg.constraint_keys:
            vals = [float(c.get(k, 0.0)) for c in costs_b]
            avg_costs[k] = float(np.mean(vals)) if len(vals) > 0 else 0.0

        # Critic 目标：r - Σ λ_k g_k + γ V(s')
        with torch.no_grad():
            V_next = self._compute_V(next_obs_t)  # (B,)
            lagrange_term = torch.zeros_like(rewards_t)
            for k, lam in self.lambdas.items():
                gk = torch.as_tensor([float(c.get(k, 0.0)) for c in costs_b], device=self.device)
                lagrange_term = lagrange_term + lam.clamp(min=0.0) * gk
            target = rewards_t - lagrange_term + (1.0 - dones_t) * self.cfg.gamma * V_next  # (B,)

        q1, q2 = self.critic.forward_tables(obs_t)  # (B,H,N)
        # 取联合动作的 Q：各头相加
        gather_idx = actions_t.unsqueeze(-1)  # (B,H,1)
        q1_pick = q1.gather(dim=-1, index=gather_idx).squeeze(-1).sum(dim=-1)  # (B,)
        q2_pick = q2.gather(dim=-1, index=gather_idx).squeeze(-1).sum(dim=-1)  # (B,)
        critic_loss = F.mse_loss(q1_pick, target) + F.mse_loss(q2_pick, target)
        self.opt_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
        self.opt_critic.step()

        # Policy：期望形式 J_pi = E[ α log π - Q_min ]
        logp, probs = self.policy.dist_all(obs_t)         # (B,H,N)
        with torch.no_grad():
            q1_pi, q2_pi = self.critic.forward_tables(obs_t)
            qmin_pi = torch.minimum(q1_pi, q2_pi)
        policy_obj = (probs * (self.alpha * logp - qmin_pi)).sum(dim=-1).sum(dim=-1)  # (B,)
        policy_loss = policy_obj.mean()
        self.opt_policy.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
        self.opt_policy.step()

        # 温度 α
        entropy_per_head = -(probs * logp).sum(dim=-1)  # (B,H)
        entropy_total = entropy_per_head.sum(dim=-1)    # (B,)
        alpha_loss = -(self.log_alpha * (entropy_total.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()

        # 软更新 target critic
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_targ.parameters()):
                pt.data.lerp_(p.data, self.cfg.tau)

        # 拉格朗日乘子：λ_k ← [λ_k + η (avg_cost_k - limit_k)]_+
        for k in self.cfg.constraint_keys:
            lam = self.lambdas[k]
            limit = self.cfg.cost_limits.get(k, 0.0)
            lam.data = torch.clamp(lam.data + self.cfg.lambda_lr * (avg_costs[k] - limit), min=0.0)

        log_info = {
            "critic_loss": float(critic_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "alpha": float(self.alpha.item()),
        }
        for k in self.cfg.constraint_keys:
            log_info[f"lambda_{k}"] = float(self.lambdas[k].item())
            log_info[f"avg_{k}"] = float(avg_costs[k])
        return log_info
