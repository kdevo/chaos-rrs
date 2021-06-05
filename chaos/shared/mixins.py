import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional


class VisualizableMixin(ABC):
    @abstractmethod
    def visualize(self, style: Optional[str] = None, **kwargs):
        pass


class PersistableMixin(ABC):
    def save(self, path: Path):
        with path.open('wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path) -> Any:
        with path.open('rb') as f:
            return pickle.load(f)
