from pathlib import Path

import pandas as pd

from chaos.shared.model import DataModel
from .source import Source, YamlSpecLoader


class CsvSource(Source):
    """Data repository for CSV example data"""
    RES_ROOT = Path('data/examples')

    def __init__(self, path=RES_ROOT, interaction_spec_path: Path = Path("data/model/interactions.yml")):
        super().__init__(YamlSpecLoader(interaction_spec_path).load_integration_spec())
        self.path = path

    def source_data(self) -> DataModel:
        users_df = pd.read_csv(self.path / 'users.csv', index_col='id', comment='#', )
        interactions = pd.read_csv(self.path / 'interactions.csv', comment='#').to_dict('records')
        for u in users_df.index.values:
            self._data.interaction_graph.add_node(u)
        for i in interactions:
            self._data.interaction_graph.add_interaction(i['from'], i['to'], i['event'])
        self._data.user_df = users_df
        return self._data
