"""On-policy logger for ssl_2v2 (MAPPO, HAPPO, ...).

HARL splits env-specific logging by pipeline: off-policy runners log inline
(that is the off_policy_base_runner hunk in patches/harl.patch), on-policy
runners hand every step and every eval episode to LOGGER_REGISTRY[env]. This
class is that hook — without it a MAPPO run trains fine but never writes a
single SSL metric.

Two metric families, named to line up with runs that already exist:

  * rollout-side rolling means under the SB3 StatsCallback names from
    train_2v2_selfplay.py (selfplay/live_success_rate, rollout/passes_per_episode,
    scenario_pass/*, scenario_chaos/*). The SAC numbers in the thesis are
    rollout-side, so this is the like-for-like comparison.
  * deterministic-eval rates under the names the HASAC patch writes
    (eval/success_rate, eval/passes_per_episode, ...) so MAPPO and HASAC
    curves overlay directly.

Both are written with add_scalar (flat tags) rather than BaseLogger's
add_scalars, which nests every tag under itself in TensorBoard.
"""
from collections import deque

import numpy as np

from harl.common.base_logger import BaseLogger

# Same windows as StatsCallback: 300 episodes overall / per chaos, 200 for the
# rarer staged-pass episodes.
ROLL_WINDOW = 300
SCEN_PASS_WINDOW = 200


class SSL2v2Logger(BaseLogger):
    def get_task_name(self):
        opp = self.env_args.get("frozen_path") or "static_blue"
        level = self.env_args.get("curriculum_level", "default")
        return f"ssl2v2_lvl{level}_vs_{opp.split('/')[-1] if isinstance(opp, str) else opp}"

    # ---------- training rollouts ----------

    def init(self, episodes):
        super().init(episodes)
        self.roll_success = deque(maxlen=ROLL_WINDOW)
        self.roll_blue_goal = deque(maxlen=ROLL_WINDOW)
        self.roll_passes = deque(maxlen=ROLL_WINDOW)
        self.roll_sap = deque(maxlen=ROLL_WINDOW)
        self.roll_level = deque(maxlen=ROLL_WINDOW)
        self.pass_success = deque(maxlen=SCEN_PASS_WINDOW)
        self.pass_passes = deque(maxlen=SCEN_PASS_WINDOW)
        self.chaos_success = deque(maxlen=ROLL_WINDOW)
        self.chaos_passes = deque(maxlen=ROLL_WINDOW)
        self.chaos_sap = deque(maxlen=ROLL_WINDOW)

    def per_step(self, data):
        super().per_step(data)
        dones, infos = data[3], data[4]
        # The env only fills is_success/passes/... on the terminal step, and
        # the subproc worker keeps that info dict while it auto-resets, so
        # infos[t][0] at a done step is the finished episode's summary.
        dones_env = np.all(dones, axis=1)
        for t in np.flatnonzero(dones_env):
            info = infos[t][0]
            if "is_success" not in info:
                continue
            self.roll_success.append(float(info["is_success"]))
            self.roll_blue_goal.append(float(info.get("blue_goal", 0.0)))
            self.roll_passes.append(float(info.get("passes", 0.0)))
            self.roll_sap.append(float(info.get("scored_after_pass", 0.0)))
            self.roll_level.append(float(info.get("curriculum_level", -1)))
            scen = info.get("scenario")
            if scen == "pass":
                self.pass_success.append(float(info["is_success"]))
                self.pass_passes.append(float(info.get("passes", 0.0)))
            elif scen == "chaos":
                self.chaos_success.append(float(info["is_success"]))
                self.chaos_passes.append(float(info.get("passes", 0.0)))
                self.chaos_sap.append(float(info.get("scored_after_pass", 0.0)))

    def episode_log(
        self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer
    ):
        super().episode_log(
            actor_train_infos, critic_train_info, actor_buffer, critic_buffer
        )
        step = self.total_num_steps
        w = self.writter
        if self.roll_success:
            w.add_scalar(
                "selfplay/live_success_rate", float(np.mean(self.roll_success)), step
            )
            w.add_scalar(
                "selfplay/blue_goal_rate", float(np.mean(self.roll_blue_goal)), step
            )
            w.add_scalar(
                "rollout/passes_per_episode", float(np.mean(self.roll_passes)), step
            )
            w.add_scalar(
                "rollout/scored_after_pass_rate", float(np.mean(self.roll_sap)), step
            )
            w.add_scalar("curriculum/level", float(np.mean(self.roll_level)), step)
        if self.pass_success:
            w.add_scalar(
                "scenario_pass/success_rate", float(np.mean(self.pass_success)), step
            )
            w.add_scalar(
                "scenario_pass/passes_per_episode",
                float(np.mean(self.pass_passes)), step,
            )
        if self.chaos_success:
            w.add_scalar(
                "scenario_chaos/success_rate", float(np.mean(self.chaos_success)), step
            )
            w.add_scalar(
                "scenario_chaos/passes_per_episode",
                float(np.mean(self.chaos_passes)), step,
            )
            w.add_scalar(
                "scenario_chaos/scored_after_pass_rate",
                float(np.mean(self.chaos_sap)), step,
            )

    # ---------- deterministic eval ----------

    def eval_init(self):
        super().eval_init()
        self.eval_episode_cnt = 0
        self.eval_yellow_goals = 0
        self.eval_blue_goals = 0
        self.eval_passes_sum = 0
        self.eval_scored_after_pass = 0
        self.eval_curriculum_levels = []
        self.eval_episode_lens = []

    def eval_thread_done(self, tid):
        # BaseLogger clears one_episode_rewards[tid] here, so take the length
        # before delegating.
        self.eval_episode_lens.append(len(self.one_episode_rewards[tid]))
        super().eval_thread_done(tid)
        self.eval_episode_cnt += 1
        info0 = self.eval_infos[tid][0]
        if info0.get("is_success", 0) > 0:
            self.eval_yellow_goals += 1
        if info0.get("blue_goal", 0) > 0:
            self.eval_blue_goals += 1
        self.eval_passes_sum += int(info0.get("passes", 0))
        if info0.get("scored_after_pass", 0) > 0:
            self.eval_scored_after_pass += 1
        self.eval_curriculum_levels.append(info0.get("curriculum_level", -1))

    def eval_log(self, eval_episode):
        self.eval_episode_rewards = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards if rewards]
        )
        eval_env_infos = {
            "eval_average_episode_rewards": self.eval_episode_rewards,
            "eval_max_episode_rewards": [np.max(self.eval_episode_rewards)],
        }
        self.log_env(eval_env_infos)

        n = max(1, self.eval_episode_cnt)
        eval_avg_rew = float(np.mean(self.eval_episode_rewards))
        eval_avg_len = float(np.mean(self.eval_episode_lens))
        eval_success_rate = self.eval_yellow_goals / n
        eval_blue_rate = self.eval_blue_goals / n
        eval_passes_per_ep = self.eval_passes_sum / n
        eval_scored_after_pass_rate = self.eval_scored_after_pass / n
        eval_curriculum_avg = (
            float(np.mean(self.eval_curriculum_levels))
            if self.eval_curriculum_levels else -1.0
        )

        step = self.total_num_steps
        self.writter.add_scalar("eval/success_rate", eval_success_rate, step)
        self.writter.add_scalar("eval/blue_goal_rate", eval_blue_rate, step)
        self.writter.add_scalar("eval/passes_per_episode", eval_passes_per_ep, step)
        self.writter.add_scalar(
            "eval/scored_after_pass_rate", eval_scored_after_pass_rate, step
        )
        self.writter.add_scalar("eval/curriculum_level", eval_curriculum_avg, step)

        print(
            f"Eval success_rate={eval_success_rate:.3f} "
            f"blue_goal_rate={eval_blue_rate:.3f} "
            f"passes/ep={eval_passes_per_ep:.2f} "
            f"avg_reward={eval_avg_rew:.2f} avg_len={eval_avg_len:.1f} "
            f"curriculum_level={eval_curriculum_avg:.1f}"
        )
        # Same column order as the off-policy progress.txt so one parser
        # reads both.
        self.log_file.write(
            ",".join(map(str, [
                step,
                eval_avg_rew,
                eval_avg_len,
                eval_success_rate,
                eval_blue_rate,
                eval_passes_per_ep,
                eval_scored_after_pass_rate,
                eval_curriculum_avg,
            ])) + "\n"
        )
        self.log_file.flush()
