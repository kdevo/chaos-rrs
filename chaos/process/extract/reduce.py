from collections import Counter
from typing import Any, Hashable, Callable, Optional

from chaos.process.processor import DFExtractor
from chaos.shared.model import DataModel


def default_key(k):
    return str(k).lower().strip()


class MostUsedExtractor(DFExtractor):
    def __init__(self, source_col: str, target_col: str,
                 top: Optional[int] = None, usage_threshold: int = 1,
                 key: Callable[[Any], Hashable] = default_key,
                 merge: bool = True, unique: bool = True):
        # TODO
        super().__init__(source_col=source_col, target_col=target_col)
        self._unique = unique
        self._selector = key
        self._top = top
        self._threshold = usage_threshold
        self._merge_existing = merge

    def execute(self, data: 'DataModel') -> 'DataModel':
        counter = Counter([self._selector(e) for words in data.user_df[self._source_col]
                           for e in self._collection(words)])
        most_used = {word for word, usage in counter.most_common(self._top) if usage >= self._threshold}

        def extract(ls):
            return set(filter(lambda it: it in most_used, map(self._selector, ls)))

        # TODO(kdevo): Refactor duplicated code to general DFExtractor class, introduce `extract` method
        if self._target_col in data.user_df and self._target_col != self._source_col:
            # If target column exists, try to append
            if self._merge_existing:
                data.user_df[self._target_col] = data.user_df[[self._source_col, self._target_col]].apply(
                    lambda s: self._merge_iterables(extract(s[self._source_col]), s[self._target_col]), axis=1
                )
            else:
                raise ValueError(f"Can not append to existing {self._target_col}, make sure to set merge = True")
        else:
            data.user_df[self._target_col] = data.user_df[self._source_col].apply(
                extract
            )

        return data
