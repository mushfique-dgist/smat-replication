# Stage 1-4 policy networks for SB3 PPO: SmatStage1Policy (Stage 1) and
# SmatStagePolicy with transfer_weights() (Stages 2-4), implemented directly
# rather than through MyoAssist's net_indexing_info mechanism (D-POLICY-01).
# The human and exo observation-width defaults are asserted against
# stage_config's derived constants.
#
# Architecture ("O p<n>" / "O Alg. 1 line <n>" cite page and algorithm-line
# numbers of the SMAT papers):
#  - Stage 1 (SmatStage1Policy): human actor MLP obs(65) -> 256 -> 128 ->
#    26 (O p2), tanh hidden + tanh-squashed mean; critic MLP obs(65) -> 256
#    -> 128 -> 1; diagonal Gaussian, one learned state-independent log-std
#    per muscle. The 2 exo action slots are always exactly 0, zero-padded
#    in _full_action(), and never enter the distribution/log-prob/entropy
#    (D-ACT-01: the env keeps the native 28-dim action space unchanged at
#    every stage, so the env cannot do this padding itself; the policy
#    must).
#  - Stages 2-4 (SmatStagePolicy): obs is [human 65 | exo 18] = 83
#    (D-OBS-01 / D-OBS-02). Exo actor MLP 18 -> 128 -> 64 -> 2 (O p2);
#    human actor MLP (65 [+2 at Stage 4]) -> 256 -> 128 -> 26 (O p2); Stage
#    4 expands the human actor's input with the exo's OWN current-step
#    command (O p4: exo acts first, human sees this step's command, not a
#    one-step-lagged value) -- this is where D-OBS-03's "+2 lives in the
#    policy, not the env" is actually implemented. One shared critic over
#    the full 83-dim state. Stage 3's frozen human actor keeps sampling
#    stochastically from its Stage-2 distribution unless
#    frozen_human_deterministic is set (D-POLICY-02: stochastic is the
#    decision; smat3/train.py never sets this flag).
#  - transfer_weights(): copies every matching tensor between stages; an
#    input layer that grew (the Stage-4 human actor) keeps its old columns
#    and gets new columns drawn N(0, new_input_std) -- O p4 "randomly
#    initialized" -- while a grown critic input column would get zeros
#    instead (function-preserving; in practice the critic's input width
#    never changes across stages here, only the human actor's does, at
#    Stage 4). This is smat3/train.py's only path between stages; 3b945c7's
#    own net_indexing_info / NetworkIndexHandler reinit system is never
#    used (D-POLICY-01).
#
# SB3 ActorCriticPolicy-style initialisation (orthogonal, gain sqrt(2)
# hidden / 0.01 actor output / 1.0 value output) and Adam (eps 1e-5) over
# the trainable parameters only -- a frozen sub-network's parameters are
# excluded from the optimizer via requires_grad_(False), not merely left
# ungraded, so no stale momentum/Adam state accumulates for a frozen actor
# across a long frozen stage.
from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import BasePolicy
from torch import nn

import stage_config as sc

ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}


def mlp(n_in: int, hidden: list[int], n_out: int, activation: type[nn.Module],
        out_activation: type[nn.Module] | None) -> nn.Sequential:
    layers, last = [], n_in
    for h in hidden:
        layers += [nn.Linear(last, h), activation()]
        last = h
    layers.append(nn.Linear(last, n_out))
    if out_activation is not None:
        layers.append(out_activation())
    return nn.Sequential(*layers)


def init_mlp(net: nn.Sequential, hidden_gain: float, out_gain: float) -> None:
    linears = [mod for mod in net if isinstance(mod, nn.Linear)]
    for i, lin in enumerate(linears):
        nn.init.orthogonal_(lin.weight, gain=out_gain if i == len(linears) - 1 else hidden_gain)
        nn.init.zeros_(lin.bias)


class SmatStage1Policy(BasePolicy):
    # Stage 1 only: human-only actor-critic on the 65-dim human observation,
    # always zero-padding the 2 exo action slots. See module header.

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        lr_schedule,
        use_sde: bool = False,
        human_net_arch: tuple = (256, 128),
        critic_net_arch: tuple = (256, 128),
        n_human_actions: int = 26,
        log_std_init: float = 0.0,
        activation: str = "tanh",
        mean_squash: str = "tanh",
        ortho_init: bool = True,
        adam_eps: float = 1e-5,
        **kwargs,
    ):
        assert not use_sde, "gSDE is not used in Stage 1"
        assert isinstance(action_space, spaces.Box) and action_space.shape[0] > n_human_actions
        super().__init__(observation_space, action_space, squash_output=False, **kwargs)
        self.human_net_arch = list(human_net_arch)
        self.critic_net_arch = list(critic_net_arch)
        self.n_human_actions = int(n_human_actions)
        self.n_exo_actions = int(action_space.shape[0]) - self.n_human_actions
        self.log_std_init = float(log_std_init)
        self.activation = activation
        self.mean_squash = mean_squash
        self.ortho_init = ortho_init
        self.adam_eps = float(adam_eps)

        n_obs = int(np.prod(observation_space.shape))
        act = ACTIVATIONS[activation]
        self.human_actor = mlp(n_obs, self.human_net_arch, self.n_human_actions, act,
                               nn.Tanh if mean_squash == "tanh" else None)
        self.critic = mlp(n_obs, self.critic_net_arch, 1, act, None)
        self.log_std = nn.Parameter(th.full((self.n_human_actions,), self.log_std_init), requires_grad=True)
        self.action_dist = DiagGaussianDistribution(self.n_human_actions)
        if ortho_init:
            init_mlp(self.human_actor, np.sqrt(2), 0.01)
            init_mlp(self.critic, np.sqrt(2), 1.0)
        self.optimizer = th.optim.Adam(self.parameters(), lr=lr_schedule(1), eps=self.adam_eps)

    def _get_constructor_parameters(self) -> dict:
        data = super()._get_constructor_parameters()
        data.update(human_net_arch=self.human_net_arch, critic_net_arch=self.critic_net_arch,
                    n_human_actions=self.n_human_actions, log_std_init=self.log_std_init,
                    activation=self.activation, mean_squash=self.mean_squash,
                    ortho_init=self.ortho_init, adam_eps=self.adam_eps)
        return data

    # ------------------------------------------------------------------ helpers
    def _features(self, obs: th.Tensor) -> th.Tensor:
        return obs.float().reshape(obs.shape[0], -1)

    def _full_action(self, human: th.Tensor) -> th.Tensor:
        return th.cat([human, human.new_zeros(human.shape[0], self.n_exo_actions)], dim=1)

    def get_distribution(self, obs: th.Tensor) -> DiagGaussianDistribution:
        """Distribution over the 26 muscle commands only."""
        return self.action_dist.proba_distribution(self.human_actor(self._features(obs)), self.log_std)

    # ------------------------------------------------------------------ SB3 interface
    def forward(self, obs: th.Tensor, deterministic: bool = False):
        x = self._features(obs)
        dist = self.action_dist.proba_distribution(self.human_actor(x), self.log_std)
        human = dist.get_actions(deterministic=deterministic)
        return self._full_action(human), self.critic(x), dist.log_prob(human)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        x = self._features(obs)
        dist = self.action_dist.proba_distribution(self.human_actor(x), self.log_std)
        human = actions[:, : self.n_human_actions]
        return self.critic(x), dist.log_prob(human), dist.entropy()

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        return self.critic(self._features(obs))

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._full_action(self.get_distribution(observation).get_actions(deterministic=deterministic))


class SmatStagePolicy(BasePolicy):
    # Stages 2-4: two actors (human, exo) and a shared critic. See module header
    # for the exact architecture and the Stage-4 human_sees_exo mechanism.

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        lr_schedule,
        use_sde: bool = False,
        n_human_obs: int = sc.HUMAN_OBS_DIM_BASE,
        n_exo_obs: int = sc.EXO_OBS_DIM,
        n_human_actions: int = 26,
        n_exo_actions: int = 2,
        human_net_arch: tuple = (256, 128),
        exo_net_arch: tuple = (128, 64),
        critic_net_arch: tuple = (256, 128),
        human_sees_exo: bool = False,
        train_human: bool = True,
        train_exo: bool = True,
        frozen_human_deterministic: bool = False,
        log_std_init: float = 0.0,
        exo_log_std_init: float = 0.0,
        activation: str = "tanh",
        mean_squash: str = "tanh",
        ortho_init: bool = True,
        adam_eps: float = 1e-5,
        **kwargs,
    ):
        assert not use_sde
        super().__init__(observation_space, action_space, squash_output=False, **kwargs)
        n_obs = int(np.prod(observation_space.shape))
        assert n_obs == n_human_obs + n_exo_obs, (n_obs, n_human_obs, n_exo_obs)
        assert action_space.shape[0] == n_human_actions + n_exo_actions
        # Defensive cross-check (new, not in the ported source): catches
        # stage_config.py drifting away from these defaults silently. Only
        # fires when the caller leaves n_human_obs/n_exo_obs at their
        # defaults; an explicit override (e.g. 67 at Stage 4, built by
        # smat3/train.py per D-OBS-03) is intentionally exempt.
        if n_human_obs == sc.HUMAN_OBS_DIM_BASE:
            assert n_exo_obs == sc.EXO_OBS_DIM, (n_exo_obs, sc.EXO_OBS_DIM)
        self.n_human_obs, self.n_exo_obs = int(n_human_obs), int(n_exo_obs)
        self.n_human_actions, self.n_exo_actions = int(n_human_actions), int(n_exo_actions)
        self.human_net_arch, self.exo_net_arch = list(human_net_arch), list(exo_net_arch)
        self.critic_net_arch = list(critic_net_arch)
        self.human_sees_exo = bool(human_sees_exo)
        self.train_human, self.train_exo = bool(train_human), bool(train_exo)
        self.frozen_human_deterministic = bool(frozen_human_deterministic)
        self.log_std_init, self.exo_log_std_init = float(log_std_init), float(exo_log_std_init)
        self.activation, self.mean_squash = activation, mean_squash
        self.ortho_init, self.adam_eps = ortho_init, float(adam_eps)
        assert self.train_human or self.train_exo

        act = ACTIVATIONS[activation]
        squash = nn.Tanh if mean_squash == "tanh" else None
        n_h_in = self.n_human_obs + (self.n_exo_actions if self.human_sees_exo else 0)
        self.human_actor = mlp(n_h_in, self.human_net_arch, self.n_human_actions, act, squash)
        self.exo_actor = mlp(self.n_exo_obs, self.exo_net_arch, self.n_exo_actions, act, squash)
        self.critic = mlp(n_obs, self.critic_net_arch, 1, act, None)
        self.log_std = nn.Parameter(th.full((self.n_human_actions,), self.log_std_init))
        self.exo_log_std = nn.Parameter(th.full((self.n_exo_actions,), self.exo_log_std_init))
        self.human_dist = DiagGaussianDistribution(self.n_human_actions)
        self.exo_dist = DiagGaussianDistribution(self.n_exo_actions)
        if ortho_init:
            init_mlp(self.human_actor, np.sqrt(2), 0.01)
            init_mlp(self.exo_actor, np.sqrt(2), 0.01)
            init_mlp(self.critic, np.sqrt(2), 1.0)
        for p in [*self.human_actor.parameters(), self.log_std]:
            p.requires_grad_(self.train_human)
        for p in [*self.exo_actor.parameters(), self.exo_log_std]:
            p.requires_grad_(self.train_exo)
        self.optimizer = th.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=lr_schedule(1),
                                       eps=self.adam_eps)

    def _get_constructor_parameters(self) -> dict:
        data = super()._get_constructor_parameters()
        data.update(n_human_obs=self.n_human_obs, n_exo_obs=self.n_exo_obs,
                    n_human_actions=self.n_human_actions, n_exo_actions=self.n_exo_actions,
                    human_net_arch=self.human_net_arch, exo_net_arch=self.exo_net_arch,
                    critic_net_arch=self.critic_net_arch, human_sees_exo=self.human_sees_exo,
                    train_human=self.train_human, train_exo=self.train_exo,
                    frozen_human_deterministic=self.frozen_human_deterministic,
                    log_std_init=self.log_std_init, exo_log_std_init=self.exo_log_std_init,
                    activation=self.activation, mean_squash=self.mean_squash,
                    ortho_init=self.ortho_init, adam_eps=self.adam_eps)
        return data

    # ------------------------------------------------------------------ helpers
    def _split(self, obs: th.Tensor):
        x = obs.float().reshape(obs.shape[0], -1)
        return x, x[:, : self.n_human_obs], x[:, self.n_human_obs: self.n_human_obs + self.n_exo_obs]

    def _human_input(self, oh: th.Tensor, u: th.Tensor) -> th.Tensor:
        return th.cat([oh, u.clamp(-1.0, 1.0)], dim=1) if self.human_sees_exo else oh

    def _act(self, obs: th.Tensor, deterministic: bool):
        x, oh, oe = self._split(obs)
        de = self.exo_dist.proba_distribution(self.exo_actor(oe), self.exo_log_std)
        u = de.get_actions(deterministic=deterministic)
        dh = self.human_dist.proba_distribution(self.human_actor(self._human_input(oh, u)), self.log_std)
        h_det = deterministic or (self.frozen_human_deterministic and not self.train_human)
        a_h = dh.get_actions(deterministic=h_det)
        log_prob = th.zeros(x.shape[0], device=x.device)
        if self.train_human:
            log_prob = log_prob + dh.log_prob(a_h)
        if self.train_exo:
            log_prob = log_prob + de.log_prob(u)
        return th.cat([a_h, u], dim=1), x, log_prob

    # ------------------------------------------------------------------ SB3 interface
    def forward(self, obs: th.Tensor, deterministic: bool = False):
        actions, x, log_prob = self._act(obs, deterministic)
        return actions, self.critic(x), log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        x, oh, oe = self._split(obs)
        a_h, u = actions[:, : self.n_human_actions], actions[:, self.n_human_actions:]
        log_prob = th.zeros(x.shape[0], device=x.device)
        entropy = th.zeros(x.shape[0], device=x.device)
        if self.train_exo:
            de = self.exo_dist.proba_distribution(self.exo_actor(oe), self.exo_log_std)
            log_prob = log_prob + de.log_prob(u)
            entropy = entropy + de.entropy()
        if self.train_human:
            dh = self.human_dist.proba_distribution(self.human_actor(self._human_input(oh, u)), self.log_std)
            log_prob = log_prob + dh.log_prob(a_h)
            entropy = entropy + dh.entropy()
        return self.critic(x), log_prob, entropy

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        return self.critic(self._split(obs)[0])

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._act(observation, deterministic)[0]


# ---------------------------------------------------------------------- weight transfer between stages
def transfer_weights(src_state: dict, dst: SmatStagePolicy, new_input_std: float = 0.01, seed: int = 0) -> dict:
    """Copy every source tensor into the matching destination tensor; an input layer that grew keeps the old
    columns and gets new columns (critic: zeros, function-preserving; human actor: N(0, new_input_std), O p4
    "randomly initialized"). Returns a record of what was copied, expanded and left at its initialisation."""
    g = th.Generator().manual_seed(seed)
    dst_state = dst.state_dict()
    record = {"copied": [], "expanded": [], "kept_init": []}
    with th.no_grad():
        for name, dst_t in dst_state.items():
            src_t = src_state.get(name)
            if src_t is None:
                record["kept_init"].append(name)
                continue
            if src_t.shape == dst_t.shape:
                dst_t.copy_(src_t)
                record["copied"].append(name)
            elif src_t.dim() == 2 and src_t.shape[0] == dst_t.shape[0] and src_t.shape[1] < dst_t.shape[1]:
                n_old = src_t.shape[1]
                dst_t[:, :n_old] = src_t
                if name.startswith("critic"):
                    dst_t[:, n_old:] = 0.0
                else:
                    dst_t[:, n_old:] = th.randn(dst_t.shape[0], dst_t.shape[1] - n_old, generator=g) * new_input_std
                record["expanded"].append(f"{name}: {tuple(src_t.shape)} -> {tuple(dst_t.shape)}")
            else:
                raise ValueError(f"cannot transfer {name}: {tuple(src_t.shape)} -> {tuple(dst_t.shape)}")
    dst.load_state_dict(dst_state)
    return record
