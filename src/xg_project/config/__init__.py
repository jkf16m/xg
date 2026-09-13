"""xg_project.config — shared configuration."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Global configuration."""

    session_path: Path | None = None
