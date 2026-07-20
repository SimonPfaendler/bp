"""Fixed-ratio demo mixing for the SB3 replay buffer.

The stock approach — dumping demo transitions into the FIFO replay buffer and
periodically re-injecting them — makes the demo share of each batch oscillate
(fresh after injection, diluted before the next one). That shows up as a
sawtooth in actor_loss and makes the demos' influence a function of buffer
fill state instead of a controlled hyperparameter.

DemoMixReplayBuffer keeps the demos in a separate buffer that is never
overwritten, and mixes a *constant* fraction into every sampled batch. Since
both SAC.train and MASAC.train sample via `self.replay_buffer.sample(...)`,
this works for both without touching the training loop.
"""

import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples


class DemoMixReplayBuffer(ReplayBuffer):
    """Replay buffer that blends a fixed fraction of demo transitions into
    every batch. `demo_ratio` is the share of each sampled batch drawn from
    the attached demo buffer (0 disables mixing → behaves like the base class).
    """

    def __init__(self, *args, demo_ratio: float = 0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.demo_ratio = float(demo_ratio)
        self._demo_buffer = None  # a plain ReplayBuffer, filled once

    def attach_demo_buffer(self, demo_buffer: ReplayBuffer) -> None:
        self._demo_buffer = demo_buffer

    def sample(self, batch_size: int, env=None) -> ReplayBufferSamples:
        if (
            self._demo_buffer is None
            or self._demo_buffer.size() == 0
            or self.demo_ratio <= 0.0
        ):
            return super().sample(batch_size, env=env)
        n_demo = int(round(batch_size * self.demo_ratio))
        n_demo = max(1, min(n_demo, batch_size - 1))
        n_online = batch_size - n_demo
        online = super().sample(n_online, env=env)
        demo = self._demo_buffer.sample(n_demo, env=env)
        # ReplayBufferSamples is a NamedTuple; zip pairs the fields in order,
        # so this stays correct even if SB3 adds fields. Some fields may be
        # None (e.g. discounts, only set for n-step buffers) — pass those
        # through instead of concatenating.
        merged = []
        for o, d in zip(online, demo):
            if o is None or d is None:
                merged.append(o if o is not None else d)
            else:
                merged.append(torch.cat([o, d], dim=0))
        return ReplayBufferSamples(*merged)


def build_demo_buffer(demos: dict, template: ReplayBuffer) -> ReplayBuffer:
    """Build a standalone ReplayBuffer holding exactly the demo transitions,
    sized to fit them all (no FIFO eviction). `template` supplies the obs/
    action spaces, device and n_envs so shapes match the main buffer."""
    n = demos["obs"].shape[0]
    n_envs = template.n_envs
    n_chunks = n // n_envs
    size = max(n_chunks * n_envs, n_envs)
    buf = ReplayBuffer(
        buffer_size=size,
        observation_space=template.observation_space,
        action_space=template.action_space,
        device=template.device,
        n_envs=n_envs,
        optimize_memory_usage=False,
    )
    infos = [{} for _ in range(n_envs)]
    for c in range(n_chunks):
        sl = slice(c * n_envs, (c + 1) * n_envs)
        buf.add(
            demos["obs"][sl], demos["next_obs"][sl], demos["actions"][sl],
            demos["rewards"][sl], demos["dones"][sl], infos,
        )
    print(f"Demo buffer: {buf.size() * n_envs} transitions (fixed, never evicted)")
    return buf


def load_buffer_into(model, buffer_path: str) -> bool:
    """Load a saved replay buffer's data INTO model.replay_buffer in place,
    preserving its class (so a DemoMixReplayBuffer keeps mixing). Returns True
    on success, False if shapes are incompatible (caller may fall back)."""
    from stable_baselines3.common.save_util import load_from_pkl

    loaded = load_from_pkl(buffer_path)
    rb = model.replay_buffer
    if loaded.buffer_size != rb.buffer_size or loaded.n_envs != rb.n_envs:
        return False
    for attr in (
        "observations", "next_observations", "actions",
        "rewards", "dones", "timeouts",
    ):
        if hasattr(loaded, attr) and hasattr(rb, attr):
            setattr(rb, attr, getattr(loaded, attr))
    rb.pos = loaded.pos
    rb.full = loaded.full
    return True
