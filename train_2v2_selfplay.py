"""Train MASAC self-play in 2v2 SSL (CTDE — centralized critic).

Blue side controlled by a frozen SAC/MASAC checkpoint (no gradients). Yellow
side is the training side; same checkpoint is loaded as yellow initialization
for the first iteration so both teams start at parity.

MASAC keeps a parameter-shared decentralized actor (38 -> 6, transfers cleanly
from prior SAC checkpoints) but replaces the per-agent critic with a
centralized one that scores the joint (2x38) obs + (2x6) action — giving the
policy gradient team-level credit assignment.

For subsequent iterations, set --frozen_path to the previous run's _final.zip
to keep climbing the response ladder.
"""

import argparse
import datetime
import os
import time
from collections import deque

import numpy as np
import torch
import wandb
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)

from masac import MASAC
from masac_policy import MASACPolicy
from pair_vec_env import (
    DummyPairVecEnv,
    JointDummyPairVecEnv,
    JointSubprocPairVecEnv,
    SubprocPairVecEnv,
)
from ssl_rl_2v2_selfplay import SSL2v2SelfPlayEnv
from demo_buffer import DemoMixReplayBuffer, build_demo_buffer, load_buffer_into

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ.setdefault("WANDB__SERVICE_WAIT", "300")

slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
print(f"Detected CPUs: {slurm_cpus}")
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

MODEL_DIR = "models"
LOG_DIR = "logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def make_env_fn(reward_type, seed, frozen_path, pass_scenario_prob=0.0):
    def _init():
        env = SSL2v2SelfPlayEnv(
            reward_type=reward_type, frozen_path=frozen_path,
            pass_scenario_prob=pass_scenario_prob,
        )
        env.reset(seed=seed)
        return env
    return _init


def _load_any(path):
    """Load a checkpoint that may be a stock-SAC run or a MASAC iteration.

    SB3 stores the policy class in the zip, so SAC.load reconstructs whichever
    one it finds. MASAC.load is the fallback for any algorithm-level mismatch.
    The caller only reads `.policy.state_dict()`.
    """
    from stable_baselines3 import SAC

    try:
        return SAC.load(path, device="cpu")
    except Exception as e:
        print(f"  SAC.load failed ({e}); retrying with MASAC.load")
        return MASAC.load(path, device="cpu")


class CurriculumCallback(BaseCallback):
    """Rolling-window curriculum: start at Level 1 (easy scoring chance),
    flip to Level 5 (chaos) once rolling success_rate clears `threshold`.

    Works with both VecEnv variants — env_method dispatches to all envs.
    """

    def __init__(
        self, start_level=1, target_level=5,
        threshold=0.9, window=300, verbose=1,
    ):
        super().__init__(verbose)
        self.start_level = int(start_level)
        self.target_level = int(target_level)
        self.threshold = float(threshold)
        self.window = int(window)
        self.success_buffer = deque(maxlen=self.window)
        self.current_level = self.start_level

    def _on_training_start(self) -> None:
        self.training_env.env_method(
            "set_curriculum_level", self.start_level
        )
        if self.verbose:
            print(f"[Curriculum] starting at level={self.start_level}")

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for i, done in enumerate(dones):
            if done and "is_success" in infos[i]:
                self.success_buffer.append(float(infos[i]["is_success"]))
        if (
            self.current_level < self.target_level
            and len(self.success_buffer) >= self.window
        ):
            sr = float(np.mean(self.success_buffer))
            if sr >= self.threshold:
                self.current_level = self.target_level
                self.training_env.env_method(
                    "set_curriculum_level", self.current_level
                )
                self.success_buffer.clear()
                print(
                    f"[Curriculum] success_rate={sr:.2f} >= "
                    f"{self.threshold} → level={self.current_level}"
                )
        self.logger.record("curriculum/level", self.current_level)
        return True


class StatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.success_buffer = deque(maxlen=300)
        self.blue_goal_buffer = deque(maxlen=300)
        self.passes_buffer = deque(maxlen=300)
        self.scored_after_pass_buffer = deque(maxlen=300)
        # Per-scenario split: the thesis question is whether passing learned
        # in the staged scenario GENERALIZES to chaos spawns — without the
        # split, a passes increase would only measure the scenario share.
        self.pass_success = deque(maxlen=200)
        self.pass_passes = deque(maxlen=200)
        self.chaos_success = deque(maxlen=300)
        self.chaos_passes = deque(maxlen=300)
        self.chaos_sap = deque(maxlen=300)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for i, done in enumerate(dones):
            if not done:
                continue
            if "is_success" in infos[i]:
                self.success_buffer.append(float(infos[i]["is_success"]))
            if "blue_goal" in infos[i]:
                self.blue_goal_buffer.append(float(infos[i]["blue_goal"]))
            if "passes" in infos[i]:
                self.passes_buffer.append(float(infos[i]["passes"]))
            if "scored_after_pass" in infos[i]:
                self.scored_after_pass_buffer.append(
                    float(infos[i]["scored_after_pass"])
                )
            scen = infos[i].get("scenario")
            if scen == "pass":
                self.pass_success.append(float(infos[i].get("is_success", 0.0)))
                self.pass_passes.append(float(infos[i].get("passes", 0.0)))
            elif scen == "chaos":
                self.chaos_success.append(float(infos[i].get("is_success", 0.0)))
                self.chaos_passes.append(float(infos[i].get("passes", 0.0)))
                self.chaos_sap.append(
                    float(infos[i].get("scored_after_pass", 0.0))
                )
        if self.success_buffer:
            self.logger.record(
                "selfplay/live_success_rate",
                float(np.mean(self.success_buffer)),
            )
        if self.blue_goal_buffer:
            self.logger.record(
                "selfplay/blue_goal_rate",
                float(np.mean(self.blue_goal_buffer)),
            )
        if self.passes_buffer:
            self.logger.record(
                "rollout/passes_per_episode",
                float(np.mean(self.passes_buffer)),
            )
        if self.scored_after_pass_buffer:
            self.logger.record(
                "rollout/scored_after_pass_rate",
                float(np.mean(self.scored_after_pass_buffer)),
            )
        if self.pass_success:
            self.logger.record(
                "scenario_pass/success_rate", float(np.mean(self.pass_success))
            )
            self.logger.record(
                "scenario_pass/passes_per_episode",
                float(np.mean(self.pass_passes)),
            )
        if self.chaos_success:
            self.logger.record(
                "scenario_chaos/success_rate",
                float(np.mean(self.chaos_success)),
            )
            self.logger.record(
                "scenario_chaos/passes_per_episode",
                float(np.mean(self.chaos_passes)),
            )
            self.logger.record(
                "scenario_chaos/scored_after_pass_rate",
                float(np.mean(self.chaos_sap)),
            )
        return True


class DebugCallback(BaseCallback):
    """Log per-module grad norms, Q-value stats, and alpha every N steps.

    Without these it's impossible to tell whether policy collapse comes from
    Q-value explosion, actor grad spikes, or alpha collapsing to zero.
    """

    def __init__(self, log_every=500, q_sample_size=256, verbose=0):
        super().__init__(verbose)
        self.log_every = int(log_every)
        self.q_sample_size = int(q_sample_size)

    def _grad_norm(self, module):
        total = 0.0
        for p in module.parameters():
            if p.grad is not None:
                total += p.grad.detach().data.norm(2).item() ** 2
        return total ** 0.5

    def _on_step(self) -> bool:
        if self.n_calls % self.log_every != 0:
            return True

        # Grad norms per module (only meaningful right after backward).
        for name, module in [("actor", self.model.actor),
                             ("critic", self.model.critic)]:
            gn = self._grad_norm(module)
            self.logger.record(f"debug/{name}_grad_norm", gn)

        # Q-value distribution over a sampled batch.
        if self.model.replay_buffer is not None and self.model.replay_buffer.size() > 1000:
            try:
                batch = self.model.replay_buffer.sample(
                    self.q_sample_size, env=self.model._vec_normalize_env
                )
                with torch.no_grad():
                    q_values = self.model.critic(batch.observations, batch.actions)
                    q_tensor = q_values[0] if isinstance(q_values, tuple) else q_values
                    self.logger.record("debug/q_min", float(q_tensor.min().item()))
                    self.logger.record("debug/q_max", float(q_tensor.max().item()))
                    self.logger.record("debug/q_mean", float(q_tensor.mean().item()))
                    self.logger.record("debug/q_std", float(q_tensor.std().item()))
            except Exception as e:
                self.logger.record("debug/q_sample_error", 1.0)

        # Alpha (entropy coefficient).
        if hasattr(self.model, "log_ent_coef") and self.model.log_ent_coef is not None:
            alpha = float(self.model.log_ent_coef.exp().item())
            self.logger.record("debug/alpha", alpha)

        return True


class AlphaClampCallback(BaseCallback):
    """Enforce a lower bound on the SAC entropy coefficient (alpha).

    If alpha drops below `alpha_min`, the policy becomes too deterministic,
    which is a major cause of policy collapse in self-play. Clamp log_ent_coef
    each step so that alpha >= alpha_min.
    """

    def __init__(self, alpha_min=0.05, verbose=0):
        super().__init__(verbose)
        self.alpha_min = float(alpha_min)
        self._log_min = None

    def _on_training_start(self) -> None:
        self._log_min = float(np.log(self.alpha_min))

    def _on_step(self) -> bool:
        if hasattr(self.model, "log_ent_coef") and self.model.log_ent_coef is not None:
            with torch.no_grad():
                self.model.log_ent_coef.clamp_(min=self._log_min)
        return True


class BestSuccessCallback(BaseCallback):
    """Save a checkpoint whenever the rolling success rate hits a new best.

    Training oscillates; the final checkpoint may sit in a trough. This keeps
    the peak policy on disk (single file, overwritten on each new best).
    """

    def __init__(self, save_path, window=300, min_episodes=100, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.window = int(window)
        self.min_episodes = int(min_episodes)
        self.buffer = deque(maxlen=self.window)
        self.best = 0.0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for i, done in enumerate(dones):
            if done and "is_success" in infos[i]:
                self.buffer.append(float(infos[i]["is_success"]))
        if len(self.buffer) >= self.min_episodes:
            sr = float(np.mean(self.buffer))
            if sr > self.best:
                self.best = sr
                path = f"{self.save_path}_best"
                self.model.save(path)
                self.logger.record("selfplay/best_success_rate", self.best)
                if self.verbose:
                    print(f"[BestCkpt] success_rate={sr:.3f} -> saved {path}")
        return True


def load_demo_transitions(demo_dir, joint):
    """Load pass-demo .pkl episodes into flat transition arrays.

    joint=True (MASAC): keeps the (2, obs) joint layout, action flattened
    to (12,), reward = team mean — matching JointSubprocPairVecEnv.
    joint=False (SAC): unstacks each joint step into 2 per-agent
    transitions — matching the SubprocPairVecEnv slot layout.
    """
    import glob
    import pickle

    files = sorted(glob.glob(os.path.join(demo_dir, "*.pkl")))
    if not files:
        raise FileNotFoundError(f"No .pkl demos in {demo_dir}")
    obs_l, next_l, act_l, rew_l, done_l = [], [], [], [], []
    for fp in files:
        with open(fp, "rb") as f:
            rec = pickle.load(f)
        for s in rec["steps"]:
            o = np.asarray(s["obs"], dtype=np.float32)
            no = np.asarray(s["next_obs"], dtype=np.float32)
            a = np.asarray(s["action"], dtype=np.float32)
            r = np.asarray(s["reward"], dtype=np.float32)
            d = bool(s["done"])
            if joint:
                obs_l.append(o)
                next_l.append(no)
                act_l.append(a.reshape(-1))
                rew_l.append(float(r.mean()))
                done_l.append(d)
            else:
                for i in range(o.shape[0]):
                    obs_l.append(o[i])
                    next_l.append(no[i])
                    act_l.append(a[i])
                    rew_l.append(float(r[i]))
                    done_l.append(d)
    print(f"Loaded {len(files)} demo episodes -> {len(obs_l)} transitions "
          f"({'joint' if joint else 'per-agent'})")
    return {
        "obs": np.stack(obs_l),
        "next_obs": np.stack(next_l),
        "actions": np.stack(act_l),
        "rewards": np.asarray(rew_l, dtype=np.float32),
        "dones": np.asarray(done_l, dtype=np.float32),
    }


def inject_demos(replay_buffer, demos):
    """Write demo transitions into the SB3 buffer in n_envs-sized chunks
    (buffer.add expects one transition per env slot)."""
    n_envs = replay_buffer.n_envs
    n = demos["obs"].shape[0]
    n_chunks = n // n_envs
    infos = [{} for _ in range(n_envs)]
    for c in range(n_chunks):
        sl = slice(c * n_envs, (c + 1) * n_envs)
        replay_buffer.add(
            demos["obs"][sl], demos["next_obs"][sl], demos["actions"][sl],
            demos["rewards"][sl], demos["dones"][sl], infos,
        )
    print(f"Injected {n_chunks * n_envs}/{n} demo transitions "
          f"(buffer size now {replay_buffer.size()}/{replay_buffer.buffer_size})")


class DemoInjectionCallback(BaseCallback):
    """Periodically re-inject demo transitions (DQfD-light).

    The SB3 buffer is FIFO — without re-injection the demos are evicted
    after ~buffer_size collected steps and the critic forgets what a
    completed pass is worth.
    """

    def __init__(self, demos, every_steps=500_000, verbose=1):
        super().__init__(verbose)
        self.demos = demos
        self.every_steps = int(every_steps)
        self._last_inject = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_inject >= self.every_steps:
            inject_demos(self.model.replay_buffer, self.demos)
            self._last_inject = self.num_timesteps
        return True


class BCLossCallback(BaseCallback):
    """Behavior-cloning auxiliary update: pull the actor toward demo actions on
    demo states. Runs an extra actor gradient step every `every` env-steps
    (after learning_starts), sharing the actor optimizer but decoupled from the
    SAC actor loss — so no train() override is needed.

    Closes the gap the DemoMixReplayBuffer leaves open: demo mixing teaches the
    CRITIC what a completed pass is worth, but the actor only reaches those
    states if it already acts like the demo. BC couples the demos to the policy.
    SAC-only (MASAC's joint obs would need separate handling).
    """

    def __init__(self, demo_buffer, bc_coef=0.5, batch_size=256, every=1,
                 learning_starts=10000, verbose=0):
        super().__init__(verbose)
        self.demo_buffer = demo_buffer
        self.bc_coef = float(bc_coef)
        self.bc_batch = int(batch_size)
        self.every = int(every)
        self.learning_starts = int(learning_starts)

    def _on_step(self) -> bool:
        if (
            self.bc_coef <= 0.0
            or self.demo_buffer is None
            or self.demo_buffer.size() == 0
            or self.num_timesteps < self.learning_starts
            or self.n_calls % self.every != 0
        ):
            return True
        import torch.nn.functional as F
        data = self.demo_buffer.sample(self.bc_batch)
        self.model.policy.set_training_mode(True)
        pred = self.model.actor(data.observations, deterministic=True)
        bc_loss = F.mse_loss(pred, data.actions)
        self.model.actor.optimizer.zero_grad()
        (self.bc_coef * bc_loss).backward()
        self.model.actor.optimizer.step()
        self.logger.record("train/bc_loss", float(bc_loss.item()))
        return True


class PassScenarioScheduleCallback(BaseCallback):
    """Linearly anneal pass_scenario_prob from start to end over the run:
    start heavily staged so the policy learns to pass, end mostly chaos so it
    learns to apply passing in unstructured play. Bridges the staged->chaos
    generalization gap. Updates the env only when the prob shifts by >=0.02, to
    avoid per-step env_method overhead across subprocesses.
    """

    def __init__(self, start_prob, end_prob, total_steps, verbose=1):
        super().__init__(verbose)
        self.start_prob = float(start_prob)
        self.end_prob = float(end_prob)
        self.total_steps = max(1, int(total_steps))
        self._last = None

    def _on_step(self) -> bool:
        frac = min(1.0, self.num_timesteps / self.total_steps)
        prob = self.start_prob + frac * (self.end_prob - self.start_prob)
        if self._last is None or abs(prob - self._last) >= 0.02:
            self.training_env.env_method("set_pass_scenario_prob", prob)
            self._last = prob
        self.logger.record("curriculum/pass_scenario_prob", prob)
        return True


def build_vec_env(n_envs, reward_type, seed, frozen_path, use_subproc, algo,
                  pass_scenario_prob=0.0):
    fns = [
        make_env_fn(reward_type, seed + i, frozen_path, pass_scenario_prob)
        for i in range(n_envs)
    ]
    if algo == "masac":
        # Joint variants keep the 2-agent pairing intact so the replay buffer
        # stores joint transitions for the centralized critic.
        if use_subproc and n_envs > 1:
            return JointSubprocPairVecEnv(fns)
        return JointDummyPairVecEnv(fns)
    # Independent SAC: unstack each pair into 2N SB3 slots. Each slot is one
    # agent, sees its own (52,) obs, outputs its own (6,) action. Parameter
    # sharing happens automatically via the single shared policy.
    if use_subproc and n_envs > 1:
        return SubprocPairVecEnv(fns)
    return DummyPairVecEnv(fns)


def train(reward_type, seed, n_envs, frozen_path, init_path=None,
          total_steps=5_000_000, algo="masac", pass_scenario_prob=0.0,
          demo_dir=None, demo_reinject_every=500_000, demo_ratio=0.25,
          bc_coef=0.5, pass_scenario_prob_start=None):
    assert algo in ("masac", "sac"), algo
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    algo_tag = algo.upper()
    run_name = f"2v2_selfplay_{algo_tag}_{reward_type}_seed{seed}_{timestamp}"
    log_dir = os.path.join(LOG_DIR, run_name)

    wandb.init(
        project="ssl-rl-2v2-selfplay",
        name=run_name,
        sync_tensorboard=True,
        config={
            "algo": algo_tag,
            "reward_type": reward_type,
            "seed": seed,
            "n_envs": n_envs,
            "frozen_path": frozen_path,
            "init_path": init_path,
            "total_steps": total_steps,
            "pass_scenario_prob": pass_scenario_prob,
            "demo_dir": demo_dir,
            "demo_reinject_every": demo_reinject_every,
        },
    )

    # Optional pass-scenario schedule: start heavily staged, anneal to the
    # target prob over the run (bridge from staged passing to chaos).
    use_pass_schedule = pass_scenario_prob_start is not None
    initial_pass_prob = (
        pass_scenario_prob_start if use_pass_schedule else pass_scenario_prob
    )
    env = build_vec_env(
        n_envs=n_envs, reward_type=reward_type, seed=seed,
        frozen_path=frozen_path, use_subproc=True, algo=algo,
        pass_scenario_prob=initial_pass_prob,
    )
    print(
        f"2v2 {algo_tag} self-play | frozen={frozen_path} | seed={seed} | "
        f"envs={n_envs} | vec_slots={env.num_envs}"
    )

    init_load = init_path or frozen_path

    policy_kwargs = dict(net_arch=[512, 512, 512])
    # Demo mixing: swap in a DemoMixReplayBuffer that blends a fixed fraction
    # of demo transitions into every batch (constant share, no re-injection).
    use_demos = demo_dir is not None
    rb_class = DemoMixReplayBuffer if use_demos else None
    rb_kwargs = {"demo_ratio": demo_ratio} if use_demos else None
    if algo == "masac":
        model = MASAC(
            policy=MASACPolicy, env=env, verbose=1, device="cuda",
            tensorboard_log=log_dir, seed=seed,
            train_freq=48, gradient_steps=96, batch_size=2048,
            buffer_size=1_000_000, learning_rate=3e-4,
            learning_starts=20000, ent_coef="auto_0.05", target_entropy="auto",
            critic_warmup_grad_steps=5000, max_grad_norm=0.5,
            policy_kwargs=policy_kwargs, gamma=0.99,
            replay_buffer_class=rb_class, replay_buffer_kwargs=rb_kwargs,
        )
    else:
        # Independent SAC diagnostic: stock SB3 SAC over per-agent (52,)
        # slots — the warmup setup, but with the current world-frame obs
        # and current reward / kick logic. If this scores against static
        # Blue and MASAC does not, the issue is in the custom MASAC stack,
        # not in env/reward/kick.
        from stable_baselines3 import SAC
        model = SAC(
            policy="MlpPolicy", env=env, verbose=1, device="cuda",
            tensorboard_log=log_dir, seed=seed,
            train_freq=48, gradient_steps=96, batch_size=2048,
            buffer_size=1_000_000, learning_rate=3e-4,
            learning_starts=10000, ent_coef="auto_0.05", target_entropy="auto",
            policy_kwargs=policy_kwargs, gamma=0.99,
            replay_buffer_class=rb_class, replay_buffer_kwargs=rb_kwargs,
        )
    if init_load and os.path.exists(init_load):
        print(f"Transferring policy weights from {init_load}")
        old_model = _load_any(init_load)
        new_state = model.policy.state_dict()
        old_state = old_model.policy.state_dict()
        transferred, skipped = [], []
        # Shape-matching transfer: same-algo checkpoints (SAC->SAC generation
        # steps) warm-start actor AND critic/critic_target; cross-algo inits
        # (e.g. MASAC->SAC) skip the incompatible critic automatically.
        for k, v in old_state.items():
            if k in new_state and new_state[k].shape == v.shape:
                new_state[k] = v
                transferred.append(k)
            else:
                skipped.append(k)
        model.policy.load_state_dict(new_state)
        n_critic = sum(1 for k in transferred if k.startswith("critic"))
        print(
            f"Policy transfer: {len(transferred)} params copied "
            f"({n_critic} critic/target), {len(skipped)} skipped (mismatches)"
        )
        del old_model
    else:
        print(f"No init checkpoint at {init_load}; training from scratch")

    # Optional: load matching replay buffer to skip cold-start.
    if init_load and os.path.exists(init_load):
        buf_path = init_load.replace(".zip", "").replace("_steps", "") + "_replay_buffer"
        # Try both naming schemes (matches CheckpointCallback + final save).
        candidates = [
            init_load.replace(".zip", ".pkl").replace("_steps", "_replay_buffer_") + "steps.pkl",
            init_load.replace("_steps.zip", "").replace("_", "_", 1) + "_replay_buffer.pkl",
        ]
        # Simple explicit match for _NNNN_steps.zip -> _replay_buffer_NNNN_steps.pkl
        import re
        m = re.match(r"^(.*)_(\d+)_steps\.zip$", init_load)
        if m:
            base, steps = m.group(1), m.group(2)
            buffer_path = f"{base}_replay_buffer_{steps}_steps.pkl"
            if os.path.exists(buffer_path):
                # Peek at buffer's n_envs — SB3 refuses to add transitions if
                # the saved buffer's n_envs doesn't match the current VecEnv.
                import pickle
                with open(buffer_path, "rb") as _f:
                    _buf = pickle.load(_f)
                saved_n_envs = getattr(_buf, "n_envs", None)
                cur_n_envs = model.replay_buffer.n_envs
                if saved_n_envs == cur_n_envs:
                    if use_demos:
                        # Copy data in place — plain load_replay_buffer would
                        # replace the instance and drop the DemoMix behavior.
                        if not load_buffer_into(model, buffer_path):
                            model.load_replay_buffer(buffer_path)
                    else:
                        model.load_replay_buffer(buffer_path)
                    print(f"Loaded replay buffer: {buffer_path} ({model.replay_buffer.size()} transitions)")
                else:
                    print(
                        f"Skipping buffer load: saved n_envs={saved_n_envs} != current n_envs={cur_n_envs}. "
                        f"Run with matching n_pairs (each pair uses 1 env slot for MASAC / joint variant)."
                    )
                del _buf
            else:
                print(f"No matching replay buffer at {buffer_path}")

    # Separate Actor/Critic learning rates.
    # Actor 3x slower than critic prevents "actor-chase" collapse where the
    # policy follows a not-yet-stable critic into a bad local minimum.
    ACTOR_LR = 1e-4
    CRITIC_LR = 3e-4
    ENT_LR = 3e-4
    # Weight decay on critic bounds Q-value magnitude via L2 on network weights.
    # Prevents Q-explosion (observed: q_mean climbing from 0 to 200+ pre-collapse).
    # 1e-4 (not 1e-3): rescaled together with the 10x reward scaling — Q targets
    # are ~10x smaller now, so the old value would over-flatten the Q landscape.
    CRITIC_WEIGHT_DECAY = 1e-4
    for pg in model.critic.optimizer.param_groups:
        pg["weight_decay"] = CRITIC_WEIGHT_DECAY

    # SB3's train() calls _update_learning_rate every iteration, which resets
    # ALL optimizer param groups to lr_schedule(progress) — silently undoing
    # any manual LR split. Override it so the split LRs are re-applied on
    # every train() call instead.
    # NOTE: must be set on the CLASS, not the instance — instance attributes
    # land in __dict__ and model.save() would try to cloudpickle the closure
    # (which drags in the SubprocVecEnv and fails on AuthenticationString).
    actor_lr, critic_lr, ent_lr = ACTOR_LR, CRITIC_LR, ENT_LR

    def _split_lr_update(self, optimizers):
        for pg in self.actor.optimizer.param_groups:
            pg["lr"] = actor_lr
        for pg in self.critic.optimizer.param_groups:
            pg["lr"] = critic_lr
        if getattr(self, "ent_coef_optimizer", None) is not None:
            for pg in self.ent_coef_optimizer.param_groups:
                pg["lr"] = ent_lr
        self.logger.record("train/actor_lr", actor_lr)
        self.logger.record("train/critic_lr", critic_lr)

    type(model)._update_learning_rate = _split_lr_update
    print(f"Actor LR: {ACTOR_LR} | Critic LR: {CRITIC_LR} | Critic WD: {CRITIC_WEIGHT_DECAY} "
          f"(enforced via _update_learning_rate override)")

    # Demo mixing (DQfD-style): scripted pass->goal transitions live in a
    # separate buffer and a fixed fraction is blended into every batch, so the
    # critic keeps a constant view of what a completed pass is worth —
    # exploration alone never produces one (cooperative bootstrap problem).
    demos = None
    if demo_dir:
        demos = load_demo_transitions(demo_dir, joint=(algo == "masac"))
        demo_buf = build_demo_buffer(demos, model.replay_buffer)
        model.replay_buffer.attach_demo_buffer(demo_buf)
        print(f"Demo mixing enabled: {demo_ratio:.0%} of every batch drawn "
              f"from the demo buffer (constant share, no re-injection)")

    callback_list = [
        StatsCallback(),
        DebugCallback(log_every=500),
        AlphaClampCallback(alpha_min=0.005),
        BestSuccessCallback(save_path=f"{MODEL_DIR}/{run_name}"),
        # Gen 3: init/frozen is the Gen-2 champion (already solves L5), so
        # skip the L1 tap-in warmup and expose pass scenarios from step 0.
        CurriculumCallback(start_level=5, target_level=5, threshold=0.9),
        CheckpointCallback(
            save_freq=20000, save_path=MODEL_DIR,
            name_prefix=run_name, save_replay_buffer=True,
        ),
    ]
    # BC auxiliary loss: couple the actor to the demo actions (SAC only —
    # MASAC's joint obs / decentralized actor would need separate handling).
    if demos is not None and algo == "sac" and bc_coef > 0.0:
        callback_list.append(BCLossCallback(
            demo_buffer=demo_buf, bc_coef=bc_coef,
            learning_starts=model.learning_starts,
        ))
        print(f"BC loss enabled: bc_coef={bc_coef} (actor pulled toward demo "
              f"actions on demo states)")
    if use_pass_schedule:
        callback_list.append(PassScenarioScheduleCallback(
            start_prob=pass_scenario_prob_start, end_prob=pass_scenario_prob,
            total_steps=total_steps,
        ))
        print(f"Pass-scenario schedule: {pass_scenario_prob_start} -> "
              f"{pass_scenario_prob} over {total_steps} steps")
    callbacks = CallbackList(callback_list)

    model.learn(
        total_timesteps=total_steps,
        reset_num_timesteps=False,
        log_interval=10,
        callback=callbacks,
    )
    final = f"{MODEL_DIR}/{run_name}_final"
    model.save(final)
    model.save_replay_buffer(f"{final}_replay_buffer")
    print(f"Saved {final}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train 2v2 SAC self-play with frozen opponent."
    )
    parser.add_argument("--frozen_path", default=None,
                        help="Path to frozen SAC checkpoint for blue side. "
                             "Omit for stationary Blue (warmup phase).")
    parser.add_argument("--init_path", default=None,
                        help="Yellow init checkpoint. Defaults to frozen_path.")
    parser.add_argument("--reward_type", default="dense",
                        choices=["sparse", "dense"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_pairs", type=int,
                        default=max(1, slurm_cpus // 2))
    parser.add_argument("--total_steps", type=int, default=5_000_000)
    parser.add_argument("--algo", default="masac", choices=["masac", "sac"],
                        help="masac: custom CTDE (default). "
                             "sac: stock Independent SAC diagnostic.")
    parser.add_argument("--pass_scenario_prob", type=float, default=0.0,
                        help="Probability of the staged pass scenario "
                             "(vs. chaos spawn) at curriculum level 5.")
    parser.add_argument("--demo_dir", default=None,
                        help="Directory of pass-demo .pkl episodes to "
                             "prefill (and periodically re-inject into) "
                             "the replay buffer.")
    parser.add_argument("--demo_reinject_every", type=int, default=500_000,
                        help="Deprecated (demo mixing now uses a fixed ratio).")
    parser.add_argument("--demo_ratio", type=float, default=0.25,
                        help="Fixed fraction of every training batch drawn "
                             "from the demo buffer (0 disables mixing).")
    parser.add_argument("--bc_coef", type=float, default=0.5,
                        help="Behavior-cloning loss weight pulling the actor "
                             "toward demo actions (SAC only; 0 disables).")
    parser.add_argument("--pass_scenario_prob_start", type=float, default=None,
                        help="If set, linearly anneal pass_scenario_prob from "
                             "this start value to --pass_scenario_prob over the "
                             "run (staged->chaos curriculum). Omit for fixed.")
    args = parser.parse_args()

    train(
        reward_type=args.reward_type, seed=args.seed,
        n_envs=args.n_pairs, frozen_path=args.frozen_path,
        init_path=args.init_path, total_steps=args.total_steps,
        algo=args.algo, pass_scenario_prob=args.pass_scenario_prob,
        demo_dir=args.demo_dir, demo_reinject_every=args.demo_reinject_every,
        demo_ratio=args.demo_ratio, bc_coef=args.bc_coef,
        pass_scenario_prob_start=args.pass_scenario_prob_start,
    )
