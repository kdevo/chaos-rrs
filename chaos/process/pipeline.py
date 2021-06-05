import logging
import multiprocessing as mp
import typing
from abc import ABC, abstractmethod
from time import time

from chaos.shared.model import DataModel
from .processor import Processor, DFExtractor
from ..shared.tools import timed

logger = logging.getLogger(__name__)


class DataPipeline(Processor, ABC):
    def __init__(self, processors: typing.List[Processor], name=None):
        super().__init__(name)
        self._processors = processors
        self._start_time = None

    def on_start(self):
        logger.info(f"Start {self.__repr__()}")

    def on_end(self):
        end_time = time()
        if self._start_time:
            logger.info(f"End {self.__repr__()}: Took {round(end_time - self._start_time, 5)}s.")

    def execute(self, data: DataModel) -> DataModel:
        self.on_start()
        data = self.process_all(data)
        self.on_end()
        return data

    @abstractmethod
    def process_all(self, data: DataModel) -> DataModel:
        pass

    def __repr__(self):
        return f"{super().__repr__()} with {len(self._processors)} sub-processors"

    @property
    def steps(self):
        return self._processors


class ParallelDataPipeline(DataPipeline):
    """ Naive (optimistic) implementation of a pipeline that executes processors in separate processes using multiprocessing.
    """

    def __init__(self, processors: typing.List[Processor], name=None, merge_method='left',
                 update_existing_columns=False):
        super().__init__(processors, name)
        self._pool = mp.Pool(mp.cpu_count())
        self._df_join = merge_method
        self._update_existing = update_existing_columns

    @timed(__name__)
    def process_all(self, data: DataModel) -> DataModel:
        logger.info(f"Executing {len(self._processors)} processors running in parallel: {self._processors}")
        results = [self._pool.apply_async(p.execute, [data]) for p in self._processors]
        for idx, res in enumerate(results):
            new_df = res.get().user_df
            if self._update_existing:
                logger.info(f"Update the following columns: {set(new_df) & set(data.user_df)}")
                data.user_df.update(res.get().user_df)

            added_cols = set(new_df.columns) - set(data.user_df.columns)
            if len(added_cols) > 0:
                logger.info(f"Detected the following new columns: {added_cols}. These will be merged.")
            if added_cols:
                data.user_df = data.user_df.merge(
                    res.get().user_df[added_cols], how=self._df_join, left_index=True, right_index=True, copy=False
                )
            # TODO(kdevo): Add proper type to detect graph manipulating:
            if not isinstance(self._processors[idx], DFExtractor):
                data.interaction_graph = data.interaction_graph.merge(
                    res.get().interaction_graph
                )
        return data


class SequentialDataPipeline(DataPipeline):
    def __init__(self, processors: typing.List[Processor], name=None,
                 callback_before: typing.Callable[[Processor], typing.Any] = None):
        super().__init__(processors, name)
        self._callback = callback_before

    @timed(__name__)
    def process_all(self, data: DataModel) -> DataModel:
        for i, p in enumerate(self._processors):
            logger.info(f"Processing step #{i + 1}: {p}")
            # logger.debug(f"Old user data: \n{data.user_df.head()}")
            if self._callback:
                self._callback(p)
            data = p.execute(data)
            # logger.debug(f"New user data: \n{data.user_df.head()}")
        return data


# TODO(kdevo): Provide sklearn DataPipeline access
class SkxPipeline(SequentialDataPipeline):
    def execute(self, data: DataModel) -> DataModel:
        pass
