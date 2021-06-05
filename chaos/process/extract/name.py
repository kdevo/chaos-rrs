from typing import Callable

import pandas as pd

from chaos.process.processor import DFExtractor
from chaos.shared.model import DataModel


def first_name(name: str) -> str:
    return name.split()[0].lower() if name else ''


class NameToGenderExtractor(DFExtractor):
    def __init__(self, source_col, target_col,
                 prename_selector: Callable[[str], str] = first_name,
                 name_to_gender_json='data/extractor/name2gender.json'):
        super().__init__(source_col, target_col, False)
        self._name_to_gender = pd.read_json(name_to_gender_json, orient='index', dtype='float16')
        self._name_to_gender.index = self._name_to_gender.index.str.lower()
        self._prename_selector = prename_selector
        self._binary_threshold = 0.6

    def execute(self, data: 'DataModel') -> 'DataModel':
        def extract(name) -> str:
            prename = self._prename_selector(name)
            try:
                row = self._name_to_gender.loc[prename]
                if row['female'] / 100 > self._binary_threshold:
                    return 'f'
                elif row['male'] / 100 > self._binary_threshold:
                    return 'm'
                else:
                    # Neutral name
                    return 'n'

            except KeyError:
                return 'na'

        data.user_df[self._target_col] = data.user_df[self._source_col].apply(extract)
        return data
