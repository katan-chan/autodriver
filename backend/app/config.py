from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_GRAPHML = BASE_DIR / "data" / "hanoi_roads.graphml"
DEFAULT_EDGE_CSV = BASE_DIR / "data" / "hanoi_component_edges.csv"


class Settings(BaseModel):
    graphml_path: Path = DEFAULT_GRAPHML
    edge_csv_path: Path = DEFAULT_EDGE_CSV
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
