from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import yaml

from chaos.shared.model import DataModel


class SpecLoader(ABC):
    @abstractmethod
    def load_integration_spec(self):
        pass


class YamlSpecLoader(SpecLoader):
    def __init__(self, interaction_spec_path: Path = Path("data/model/interactions.yml")):
        self.interaction_spec_path = interaction_spec_path

    def load_integration_spec(self):
        with self.interaction_spec_path.open() as ispec_file:
            return yaml.safe_load(ispec_file)


class Source(ABC):
    """Simple data repository that uses YAML for interaction and user specification"""

    def __init__(self, interaction_spec: Dict):
        self._data = DataModel(interaction_spec=interaction_spec)

    @abstractmethod
    def source_data(self) -> DataModel:
        pass

    @property
    def data(self) -> DataModel:
        return self._data

    @data.setter
    def data(self, d: DataModel):
        self._data = d
