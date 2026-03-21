import csv
from pathlib import Path

from .runner import EpisodeStats


class CSVLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            ["episode", "steps", "return", "success", "terminated", "truncated", "walltime_s"]
        )

    def log_episode(self, stats: EpisodeStats) -> None:
        self.writer.writerow(
            [
                stats.episode,
                stats.steps,
                stats.ep_return,
                stats.success,
                stats.terminated,
                stats.truncated,
                stats.walltime_s,
            ]
        )
        self.file.flush()

    def close(self) -> None:
        self.file.close()
