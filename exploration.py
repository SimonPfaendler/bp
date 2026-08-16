"""Exploration pieces adopted from FlashSAC (arXiv 2604.04539), adapted to
a CPU-bound simulator with tens rather than thousands of parallel envs.

Two mechanisms, both aimed at the same failure: alpha sits at its floor
(0.005) in nearly every run, so the policy is close to deterministic, and
what noise remains is drawn independently per timestep — i.e. jitter. A
pass is a multi-step manoeuvre (secure the ball, rotate, kick), and jitter
essentially never produces one by chance. There is currently no mechanism
that could discover a pass during play; the demos only teach the actor on
demo states, where the BC loss has long since converged.

  unified_target_entropy — parameterise the SAC entropy target by a target
      action standard deviation instead of -|A|. -|A| is not reachable for
      some policies, which is what let alpha ratchet to 114 in the scratch
      MASAC run; a sigma-based target is achievable by construction and
      keeps a defined amount of exploration alive.

  NoiseRepeatSAC — hold the sampled Gaussian noise vector for k steps
      (k ~ Zeta), so the stochastic part of the action is temporally
      correlated while the mean still tracks the state. Turns jitter into
      sustained, directed deviations, which is what makes a multi-step
      manoeuvre discoverable.
"""
import math

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.utils import obs_as_tensor


def unified_target_entropy(action_dim: int, target_std: float) -> float:
    """Entropy of a diagonal Gaussian with std `target_std`.

    H = 0.5 * |A| * log(2*pi*e*sigma^2). FlashSAC uses sigma = 0.15 across
    all embodiments. For |A|=6 that is about -2.87 (vs. -6 for "auto"), so
    the policy must keep noticeably more exploration alive.
    """
    return 0.5 * action_dim * math.log(2.0 * math.pi * math.e * target_std ** 2)


class NoiseRepeatSAC(SAC):
    """SAC whose rollout noise is held constant for k consecutive steps.

    Only action SELECTION changes; the update is stock SAC, so demo mixing,
    the BC callback and the LR split all keep working unchanged.
    """

    def __init__(self, *args, noise_repeat_s: float = 2.0,
                 noise_repeat_max: int = 16, **kwargs):
        # Set before super().__init__ so _setup_model can already see them.
        self.noise_repeat_s = float(noise_repeat_s)
        self.noise_repeat_max = int(noise_repeat_max)
        self._nr_eps = None          # held noise, (n_envs, act_dim)
        self._nr_left = None         # steps remaining per env
        self._nr_rng = np.random.default_rng(kwargs.get("seed") or 0)
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        # Live tensors / RNG state must not be cloudpickled into the zip.
        return super()._excluded_save_params() + [
            "_nr_eps", "_nr_left", "_nr_rng",
        ]

    def _refresh_noise(self, shape) -> th.Tensor:
        n_envs, act_dim = shape
        if (
            self._nr_eps is None
            or tuple(self._nr_eps.shape) != (n_envs, act_dim)
        ):
            self._nr_eps = th.randn(n_envs, act_dim, device=self.device)
            self._nr_left = np.zeros(n_envs, dtype=np.int64)
        for i in range(n_envs):
            if self._nr_left[i] <= 0:
                self._nr_eps[i] = th.randn(act_dim, device=self.device)
                # Zeta/zipf favours short holds but occasionally draws a
                # long, strongly correlated stretch.
                k = int(self._nr_rng.zipf(self.noise_repeat_s))
                self._nr_left[i] = min(k, self.noise_repeat_max)
            self._nr_left[i] -= 1
        return self._nr_eps

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        # Warmup phase: keep SB3's uniform-random sampling.
        if self.num_timesteps < learning_starts and not (
            self.use_sde and self.use_sde_at_warmup
        ):
            return super()._sample_action(learning_starts, action_noise, n_envs)

        assert self._last_obs is not None
        with th.no_grad():
            obs_tensor = obs_as_tensor(self._last_obs, self.device)
            mean_actions, log_std, _ = self.actor.get_action_dist_params(
                obs_tensor
            )
            eps = self._refresh_noise(tuple(mean_actions.shape))
            # Same squashed-Gaussian sample as SB3, but with a held eps.
            unscaled_action = th.tanh(
                mean_actions + log_std.exp() * eps
            ).cpu().numpy()

        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action
