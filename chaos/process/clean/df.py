from typing import Optional, Any

from chaos.shared.model import DataModel
from ..processor import Processor

non_word_re = r"(?!(?:[^\W]|\s)+)"


class DFCleaner(Processor):
    def __init__(self, cols: list = None,
                 str_clean_regex: Optional[str] = None, str_clean_repl: str = '',
                 fill_na_val: Optional[Any] = ''):
        super().__init__()
        self._cols = cols
        self._clean_regex = str_clean_regex
        self._repl = str_clean_repl
        self._fill_with_val = fill_na_val
        # self._str_lower = str_lower

    def execute(self, dm: DataModel) -> DataModel:
        cols = self._cols if self._cols else dm.user_df.columns.values

        # if self._str_lower:
        #     dm.user_df[cols] = dm.user_df[cols].apply(test, axis=1)

        if self._clean_regex:
            dm.user_df[cols] = dm.user_df[cols].replace(self._clean_regex, self._repl, regex=True)

        if self._fill_with_val is not None:
            dm.user_df[cols] = dm.user_df[cols].fillna(self._fill_with_val)
        return dm
