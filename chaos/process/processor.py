import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import chaos.shared.model as model

logger = logging.getLogger(__name__)


class Processor(ABC):
    def __init__(self, name=None):
        pass

    @abstractmethod
    def execute(self, data: 'model.DataModel') -> 'model.DataModel':
        pass

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__repr__()


class Extractor(Processor, ABC):
    def __init__(self, target_col=None):
        """
        Generic extractor that puts a calculated result to a specified target_col.
        Args:
            name: Name of the extractor (useful for debugging).
            target_col: Name of the column to write the calculated result.
        """
        super().__init__()
        self._target_col = target_col

    @property
    def target_col(self):
        return self._target_col

    def __repr__(self):
        return f"{super().__repr__()} ⟶ {self._target_col}"


class SocialNetExtractor(Extractor, ABC):
    pass


class DFExtractor(Extractor, ABC):
    def __init__(self, source_col=None, target_col=None, merge: bool = True, unique: bool = True):
        """
        Extracts a feature from a data frame column `source_col` to another `target_col`.

        Args:
            source_col: Source column to extract data from
            target_col:
        """
        super().__init__()
        self._source_col = source_col
        self._target_col = target_col
        self._merge_existing = merge
        self._unique = unique

    @abstractmethod
    def execute(self, data: 'model.DataModel') -> 'model.DataModel':
        return data

    # @abstractmethod
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.combine.html
    # def extract(self, df: pd.DataFrame) -> pd.DataFrame:
    #     pass

    def _is_mergeable(self, to_check: Any):
        return type(to_check) == set or type(to_check) == list

    def _merge_iterables(self, it, val, unique=True):
        if self._merge_existing:
            if not self._is_mergeable(it):
                it = []
            if not self._is_mergeable(val):
                val = []
            return {*it, *val} if unique else [*it, *val]
        return val

    def _collection(self, data):
        if data is np.nan or data is None:
            return set() if self._unique else list()
        return set(data) if self._unique else list(data)

    def __repr__(self):
        return f"{self.__class__.__name__}({self._source_col}) ⟶ {self._target_col}"

