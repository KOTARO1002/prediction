from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class Config:
    raw: Dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 42))

    def __getitem__(self, item: str) -> Any:
        return self.raw[item]


def load_config(path: str = "config.yaml") -> Config:
    content = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Config(raw=content)
