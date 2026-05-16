from harl.common.base_logger import BaseLogger


class SSL2v2Logger(BaseLogger):
    def get_task_name(self):
        opp = self.env_args.get("frozen_path") or "static_blue"
        level = self.env_args.get("curriculum_level", "default")
        return f"ssl2v2_lvl{level}_vs_{opp.split('/')[-1] if isinstance(opp, str) else opp}"
