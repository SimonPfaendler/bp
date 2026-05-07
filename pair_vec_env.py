"""PairVecEnv: expose N pair envs as 2*N agent slots to SB3.

Each underlying "pair env" runs one shared physics simulator and returns
a stacked (2, OBS) observation + (2,) reward per step. The wrapper
unstacks these into 2*N parallel slots, which is what SB3's SAC
training loop sees. Both slots from a pair share the same physics
tick, but their transitions are written to the replay buffer
independently — this is the parameter-sharing data-pooling effect.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Callable, List, Optional, Sequence

import cloudpickle
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
)


def _resolve_pair_indices(n_pairs: int, indices: VecEnvIndices) -> List[int]:
    """Map agent-slot indices -> unique pair indices."""
    if indices is None:
        return list(range(n_pairs))
    if isinstance(indices, int):
        indices = [indices]
    return sorted({int(i) // 2 for i in indices})


class DummyPairVecEnv(VecEnv):
    """Single-process pair vec env. Useful for debugging / smoke tests."""

    def __init__(self, env_fns: Sequence[Callable[[], Any]]):
        self.envs = [fn() for fn in env_fns]
        self.n_pairs = len(self.envs)
        sample = self.envs[0]
        super().__init__(
            num_envs=2 * self.n_pairs,
            observation_space=sample.single_observation_space,
            action_space=sample.single_action_space,
        )
        self._actions: Optional[np.ndarray] = None
        self._buf_obs = np.zeros(
            (self.num_envs,) + self.observation_space.shape,
            dtype=self.observation_space.dtype,
        )
        self._pending_seeds: List[Optional[int]] = [None] * self.n_pairs

    def reset(self) -> np.ndarray:
        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=self._pending_seeds[i])
            self._buf_obs[2 * i] = obs[0]
            self._buf_obs[2 * i + 1] = obs[1]
        self._pending_seeds = [None] * self.n_pairs
        return self._buf_obs.copy()

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self):
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: List[dict] = [{} for _ in range(self.num_envs)]
        for i, env in enumerate(self.envs):
            pair_action = np.stack(
                [self._actions[2 * i], self._actions[2 * i + 1]], axis=0
            )
            obs, r, done, truncated, info = env.step(pair_action)
            terminal = bool(done) or bool(truncated)
            if terminal:
                terminal_obs = obs.copy()
                obs, _ = env.reset()
                for k in (0, 1):
                    info_k = dict(info)
                    info_k["terminal_observation"] = terminal_obs[k]
                    info_k["TimeLimit.truncated"] = bool(
                        truncated and not done
                    )
                    infos[2 * i + k] = info_k
            else:
                infos[2 * i] = dict(info)
                infos[2 * i + 1] = dict(info)
            self._buf_obs[2 * i] = obs[0]
            self._buf_obs[2 * i + 1] = obs[1]
            rewards[2 * i] = r[0]
            rewards[2 * i + 1] = r[1]
            dones[2 * i] = terminal
            dones[2 * i + 1] = terminal
        return self._buf_obs.copy(), rewards, dones, infos

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs,
    ):
        pair_idx = _resolve_pair_indices(self.n_pairs, indices)
        return [
            getattr(self.envs[i], method_name)(*method_args, **method_kwargs)
            for i in pair_idx
        ]

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None):
        # Return one value per requested slot
        if indices is None:
            slot_indices = list(range(self.num_envs))
        elif isinstance(indices, int):
            slot_indices = [indices]
        else:
            slot_indices = list(indices)
        return [getattr(self.envs[int(i) // 2], attr_name) for i in slot_indices]

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        for i in _resolve_pair_indices(self.n_pairs, indices):
            setattr(self.envs[i], attr_name, value)

    def env_is_wrapped(self, wrapper_class, indices: VecEnvIndices = None):
        if indices is None:
            slot_indices = list(range(self.num_envs))
        elif isinstance(indices, int):
            slot_indices = [indices]
        else:
            slot_indices = list(indices)
        return [False for _ in slot_indices]

    def seed(self, seed: Optional[int] = None):
        if seed is None:
            self._pending_seeds = [None] * self.n_pairs
        else:
            self._pending_seeds = [seed + i for i in range(self.n_pairs)]
        return self._pending_seeds

    def get_images(self):
        return []




def _pair_worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.var()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                pair_action = data
                try:
                    obs, rewards, done, truncated, info = env.step(pair_action)
                    terminal = bool(done) or bool(truncated)
                    terminal_obs = None
                    if terminal:
                        terminal_obs = obs.copy()
                        obs, _ = env.reset()
                    remote.send(
                        (obs, rewards, bool(done), bool(truncated), info, terminal_obs)
                    )
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    try:
                        obs, _ = env.reset()
                    except Exception:
                        raise
                    zero_r = np.zeros(2, dtype=np.float32)
                    err_info = {"worker_error": repr(e)}
                    remote.send(
                            (obs, zero_r, True, False, err_info, obs.copy())
                    )
            elif cmd == "reset":
                seed = data
                obs, info = env.reset(seed=seed)
                remote.send((obs, info))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                attr, value = data
                setattr(env, attr, value)
                remote.send(None)
            elif cmd == "env_method":
                method, args, kwargs = data
                remote.send(getattr(env, method)(*args, **kwargs))
            elif cmd == "is_wrapped":
                remote.send(False)
            else:
                raise NotImplementedError(f"unknown cmd: {cmd}")
    except KeyboardInterrupt:
        pass
    finally:
        try:
            env.close()
        except Exception:
            pass


class SubprocPairVecEnv(VecEnv):
    """Pair vec env with one subprocess per pair (one physics simulator per process)."""

    def __init__(self, env_fns: Sequence[Callable[[], Any]], start_method: str = "spawn"):
        self.n_pairs = len(env_fns)
        self.waiting = False
        self.closed = False

        ctx = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(self.n_pairs)]
        )
        self.processes = []
        for work_remote, remote, fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(fn))
            p = ctx.Process(target=_pair_worker, args=args, daemon=True)
            p.start()
            self.processes.append(p)
            work_remote.close()

        # Probe the first env for spaces.
        self.remotes[0].send(("get_attr", "single_observation_space"))
        obs_space = self.remotes[0].recv()
        self.remotes[0].send(("get_attr", "single_action_space"))
        act_space = self.remotes[0].recv()

        super().__init__(
            num_envs=2 * self.n_pairs,
            observation_space=obs_space,
            action_space=act_space,
        )

        self._buf_obs = np.zeros(
            (self.num_envs,) + obs_space.shape, dtype=obs_space.dtype
        )
        self._pending_seeds: List[Optional[int]] = [None] * self.n_pairs

    def reset(self):
        for remote, seed in zip(self.remotes, self._pending_seeds):
            remote.send(("reset", seed))
        for i, remote in enumerate(self.remotes):
            obs, _ = remote.recv()
            self._buf_obs[2 * i] = obs[0]
            self._buf_obs[2 * i + 1] = obs[1]
        self._pending_seeds = [None] * self.n_pairs
        return self._buf_obs.copy()

    def step_async(self, actions: np.ndarray) -> None:
        for i, remote in enumerate(self.remotes):
            pair_action = np.stack(
                [actions[2 * i], actions[2 * i + 1]], axis=0
            )
            remote.send(("step", pair_action))
        self.waiting = True

    def step_wait(self):
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: List[dict] = [{} for _ in range(self.num_envs)]
        for i, remote in enumerate(self.remotes):
            obs, r, done, truncated, info, terminal_obs = remote.recv()
            terminal = done or truncated
            self._buf_obs[2 * i] = obs[0]
            self._buf_obs[2 * i + 1] = obs[1]
            rewards[2 * i] = r[0]
            rewards[2 * i + 1] = r[1]
            dones[2 * i] = terminal
            dones[2 * i + 1] = terminal
            if terminal:
                for k in (0, 1):
                    info_k = dict(info)
                    info_k["terminal_observation"] = terminal_obs[k]
                    info_k["TimeLimit.truncated"] = bool(truncated and not done)
                    infos[2 * i + k] = info_k
            else:
                infos[2 * i] = dict(info)
                infos[2 * i + 1] = dict(info)
        self.waiting = False
        return self._buf_obs.copy(), rewards, dones, infos

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.processes:
            p.join()
        self.closed = True

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs,
    ):
        pair_idx = _resolve_pair_indices(self.n_pairs, indices)
        for i in pair_idx:
            self.remotes[i].send(
                ("env_method", (method_name, method_args, method_kwargs))
            )
        return [self.remotes[i].recv() for i in pair_idx]

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None):
        if indices is None:
            slot_indices = list(range(self.num_envs))
        elif isinstance(indices, int):
            slot_indices = [indices]
        else:
            slot_indices = list(indices)
        # Group by pair to avoid duplicate IPC roundtrips.
        unique_pairs = sorted({int(i) // 2 for i in slot_indices})
        for i in unique_pairs:
            self.remotes[i].send(("get_attr", attr_name))
        results = {i: self.remotes[i].recv() for i in unique_pairs}
        return [results[int(i) // 2] for i in slot_indices]

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        for i in _resolve_pair_indices(self.n_pairs, indices):
            self.remotes[i].send(("set_attr", (attr_name, value)))
            self.remotes[i].recv()

    def env_is_wrapped(self, wrapper_class, indices: VecEnvIndices = None):
        if indices is None:
            slot_indices = list(range(self.num_envs))
        elif isinstance(indices, int):
            slot_indices = [indices]
        else:
            slot_indices = list(indices)
        return [False for _ in slot_indices]

    def seed(self, seed: Optional[int] = None):
        if seed is None:
            self._pending_seeds = [None] * self.n_pairs
        else:
            self._pending_seeds = [seed + i for i in range(self.n_pairs)]
        return self._pending_seeds

    def get_images(self):
        return []
