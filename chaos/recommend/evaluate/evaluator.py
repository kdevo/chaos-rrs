import logging
import math
import multiprocessing
import random
from abc import abstractmethod, ABC
from collections import namedtuple
from contextlib import ContextDecorator
from dataclasses import dataclass
from typing import Dict, Callable, Iterable, Any, Optional, Tuple, List, Set, Union

import altair as alt
import numpy as np
import pandas as pd
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import precision_at_k, recall_at_k, reciprocal_rank, auc_score
from scipy.sparse import coo_matrix
from tqdm import tqdm

from chaos.recommend.candidates import StaticCandidateRepo
from chaos.recommend.predict.predictor import LFMPredictor, Predictor, ModelBasedPredictor
from chaos.shared.graph import InteractionGraph
from chaos.shared.mixins import VisualizableMixin
from chaos.shared.tools import timed
from grapresso import DiGraph
from grapresso.backends import InMemoryBackend

logger = logging.getLogger(__name__)


def f1_score(recall, precision):
    """
    The f1 score aggregates precision and recall by the harmonic mean.

    Args:
        recall: See recall function
        precision: See precision function

    Returns:
        Harmonic mean of recall and precision (commonly known as f1)

    """
    return (2 * recall * precision) / (recall + precision)


class disablelog(ContextDecorator):
    def __init__(self, until_level=logging.WARNING):
        self._level = until_level

    def __enter__(self):
        logging.disable(self._level)

    def __exit__(self, *exc):
        logging.disable(logging.NOTSET)


@dataclass(frozen=True, eq=True)
class Metric(ABC):
    name: str
    types: Tuple[str, str]
    k: int

    @property
    def descriptor(self):
        return [self.col_name, self.types]

    @property
    def is_test(self):
        return 'test' in self.types

    @property
    def is_train(self):
        return 'train' in self.types

    @property
    def col_name(self):
        return f"{self.name}@{self.k}" if self.k else f"{self.name}"


@dataclass(frozen=True, eq=True)
class SingleMetric(Metric):
    name: str
    types: Tuple[str, str] = ('train', 'test')
    k: Optional[int] = 5


@dataclass(frozen=True, eq=True)
class CompositeMetric(Metric):
    name: str
    types: Tuple[str, str]
    k: int

    func: Callable[..., float]
    metrics: Tuple[SingleMetric, ...]
    safe_divide: bool = True

    @classmethod
    def from_metrics(cls, name, func: Callable[..., float],
                     metrics: Tuple[SingleMetric, ...],
                     with_validation: bool = True):
        types = metrics[0].types
        k = metrics[0].k
        if with_validation:
            for m in metrics:
                if types != m.types or k != m.k:
                    # TODO(kdevo): Should we support different k (for e.g. mean)?
                    raise ValueError("Aggregated metrics must have same types and same k!")
        return CompositeMetric(name, types, k, func, metrics)

    def apply(self, *res: Tuple[float]):
        try:
            return self.func(*res)
        except ZeroDivisionError:
            if self.safe_divide:
                return 0.0
            else:
                raise

    def col_names(self) -> List[str]:
        return [m.col_name for m in self.metrics]


class Evaluator(ABC):
    """ Evaluation class to validate that the model is working correctly.
    Typical workflow:
    1. Split interactions to train/test
    2. Train predictor models with train set
    3. Test predictor models against test set and collect metrics
    4. Compare predictor models
    """
    AUC = SingleMetric('auc', k=None)
    PRECISION = SingleMetric('precision', k=5)
    RECALL = SingleMetric('recall', k=5)
    RECIPROCAL = SingleMetric('reciprocal-rank', k=5)
    F1 = CompositeMetric.from_metrics('f1', lambda p, r: 2 * (p * r) / (p + r), metrics=(PRECISION, RECALL))

    ALL = (AUC, PRECISION, RECALL, F1)

    BestResult = namedtuple("BestResult", ['predictor', 'epoch', 'value'], defaults=[0, 0])

    @abstractmethod
    def best_of_all(self, **kwargs) -> Any:
        pass

    @abstractmethod
    def run_all(self, metrics: Iterable[Metric], epochs: range = range(0, 10, 1), **kwargs):
        pass


class EpochBasedEvaluator(Evaluator, VisualizableMixin, ABC):
    def __init__(self, predictors: Dict[str, Predictor], test_split: float = 0.2, show_progress=True):
        self._predictors = predictors
        self.show_progress: bool = show_progress
        self.test_split = test_split

        self.results = {}
        self._last_run_metrics = []

    @property
    @abstractmethod
    def train_interactions(self) -> Any:
        pass

    @property
    @abstractmethod
    def test_interactions(self) -> Any:
        pass

    @property
    def last_run_metrics(self) -> List[Metric]:
        return self._last_run_metrics

    def run_all(self, metrics: Iterable[Metric] = None,
                epochs: range = range(0, 10, 1), **kwargs) -> Dict[str, pd.DataFrame]:
        # TODO: Consider to use set (or dict to preserve order) for faster membership testing
        unpacked_metrics = []
        composite_metrics = []
        if not metrics:
            metrics = self.supported_metrics
        for m in metrics:
            if not self.is_supported(m):
                raise ValueError(f"Metric {m} is not supported by this Evaluator!")
            if isinstance(m, CompositeMetric):
                for sub_m in m.metrics:
                    if sub_m in unpacked_metrics:
                        continue
                    unpacked_metrics.append(sub_m)
                if m not in composite_metrics:
                    composite_metrics.append(m)
            elif m not in unpacked_metrics:
                unpacked_metrics.append(m)

        self.results = {}
        for key in self._predictors.keys():
            self.results[key] = self.run(key, unpacked_metrics, composite_metrics, epochs)

        self._last_run_metrics = unpacked_metrics + composite_metrics
        return self.results

    @property
    @abstractmethod
    def supported_metrics(self) -> Set[Metric]:
        return set(self.ALL)

    def is_supported(self, metric: Metric):
        return any(map(lambda m: m.name == metric.name, self.supported_metrics))

    @disablelog()
    def run(self, predictor_key: str,
            single_metrics: Iterable[Metric],
            composite_metrics: Iterable[CompositeMetric], epochs) -> pd.DataFrame:

        trained_epochs = 0
        predictor = self._predictors[predictor_key]
        # Set train interactions
        predictor.translator.train_interactions = self.train_interactions

        all_metrics = list(single_metrics) + list(composite_metrics)
        multi_idx = pd.MultiIndex.from_product([epochs, [m.col_name for m in all_metrics]],
                                               names=('epoch', 'method'))
        eval_df = pd.DataFrame(columns=['train', 'test'], index=multi_idx)

        def calc_metrics(epoch):
            for m in single_metrics:
                train, test = self.evaluate(predictor, m)
                if train:
                    eval_df.loc[(epoch, m.col_name), 'train'] = train
                if test:
                    eval_df.loc[(epoch, m.col_name), 'test'] = test
            for m in composite_metrics:
                for t in m.types:
                    eval_df.loc[(epoch, m.col_name), t] = m.apply(
                        *[eval_df.loc[(epoch, subm.col_name), t] for subm in m.metrics]
                    )

        if isinstance(predictor, ModelBasedPredictor):
            progress = None
            if epochs.start < 0 or epochs.stop < 0:
                raise ValueError("Epochs must be in interval [0, n]")
            if epochs.start == 0:
                eval_df.loc[((0, m.col_name) for m in all_metrics), :] = 0.0
                epochs = epochs[1:]
                # Reset (if already trained):
                predictor.train(epochs=0, resume=False)
            if self.show_progress:
                progress = tqdm(total=len(epochs), desc=predictor_key, position=1, leave=False)
            for epoch in epochs:
                epoch_batch = epoch - trained_epochs
                if epoch_batch <= 0:
                    continue
                predictor.train(epochs=epoch_batch, resume=True)
                trained_epochs += epoch_batch
                calc_metrics(epoch)
                if progress:
                    progress.update()
            if progress:
                progress.close()
        else:
            calc_metrics(0)
            for epoch in epochs[1:]:
                eval_df.loc[(epoch,), :] = eval_df.loc[(0,), :].values
        return eval_df

    def create_report(self, format='overview', on=('train', 'test'), static_scale=True) -> alt.Chart:
        # FIXME(kdevo): Change "on" to metric
        if format == 'methods':
            complete_chart = alt.vconcat().configure(font="Comfortaa")
            for t in on:
                for d, df in self.results.items():
                    chart = alt.Chart(df.reset_index()).mark_bar().encode(
                        alt.X(f"mean({t}):Q", scale=alt.Scale(domain=(0, 1)) if static_scale else alt.Undefined),
                        y='method', color='method:N'
                    ).properties(width=300, height=100, title=d).interactive()
                    complete_chart &= chart

        elif format == 'overview':
            complete_chart = alt.vconcat().configure(font="Comfortaa")
            # Simple two column layout
            for t in on:
                row = None
                for idx, ri in enumerate(self.results.items()):
                    chart = alt.Chart(ri[1].reset_index()).mark_line().encode(
                        alt.Y(f"{t}:Q"),
                        x='epoch:Q',
                        color=alt.Color('method:N', scale=alt.Scale(scheme='category10'))
                    ).properties(width=200, height=200, title=ri[0]).interactive()
                    if row:
                        row = alt.hconcat(row, chart).resolve_scale(y='shared')
                    else:
                        row = chart
                complete_chart &= row

        else:
            raise ValueError(f"Format '{format} not supported!")

        return complete_chart

    def visualize(self, style: Optional[str] = None, **kwargs):
        self.create_report(style).show()

    @abstractmethod
    def evaluate(self, predictor: Predictor, metric: Metric) -> Tuple[Optional[float], Optional[float]]:
        pass

    def best_of_all(self, best_metric: Union[str, Metric] = None, on='test') -> Evaluator.BestResult:
        # TODO(kdevo): For now, best_metric must be mutually exclusive regarding its (train, test) properties
        best_metric_name = best_metric.name if isinstance(best_metric, Metric) else best_metric
        if best_metric_name:
            mn, mt = next(filter(lambda m: m.name == best_metric_name, self.last_run_metrics)).descriptor
        else:
            mn, mt = self.last_run_metrics[0].descriptor
        if on not in mt:
            raise ValueError(
                f"Can't get best {best_metric_name} on type {on} because it has not been evaluated during the run!"
            )

        best = Evaluator.BestResult(None, None, 0)
        for k, result_df in self.results.items():
            maximum_idx = pd.to_numeric(result_df[on].loc[:, mn]).idxmax()
            maximum_val = result_df[on].loc[maximum_idx, mn]
            if maximum_val > best.value:
                best = Evaluator.BestResult(k, maximum_idx, maximum_val)

        return best

    def __getitem__(self, predictor_key: str) -> Predictor:
        return self._predictors[predictor_key]


class LFMEvaluator(EpochBasedEvaluator):
    """ Setting this to True will not exclude re-recommendations, i.e. items that have already been trained
    and therefore (possibly) already recommended to the user.
    """
    _predictors: Dict[str, LFMPredictor]

    def __init__(self, predictors: Dict[str, LFMPredictor], interactions: coo_matrix, test_split: float = 0.2,
                 show_progress=True, random_state: np.random.RandomState = None):
        super().__init__(predictors, test_split, show_progress=show_progress)
        self.allow_rerecommendations: bool = True
        self._train_it, self._test_it = random_train_test_split(interactions,
                                                                test_percentage=self.test_split,
                                                                random_state=random_state)
        self._threads = multiprocessing.cpu_count()
        logging.debug(f"Maximum average precision@5: {self._test_it.nnz / 5}")

    def evaluate(self, predictor: LFMPredictor, metric: Metric) -> Tuple[float, float]:
        # Boost performance by not checking intersections between train/test (re-enable if in doubt)
        additional_args = {'check_intersections': False}
        if 'precision' in metric.name:
            method = precision_at_k
            additional_args['k'] = metric.k
        elif 'recall' in metric.name:
            method = recall_at_k
            additional_args['k'] = metric.k
        elif 'reciprocal' in metric.name:
            method = reciprocal_rank
        elif 'auc' in metric.name:
            method = auc_score
        else:
            raise ValueError(f"Unsupported evaluation method: {metric}!")

        train = None
        if metric.is_train:
            train = method(model=predictor.model,
                           test_interactions=self._train_it,
                           user_features=predictor.translator.user_matrix,
                           item_features=predictor.translator.user_matrix,
                           num_threads=self._threads, **additional_args)
            train = train.mean()
        test = None
        if metric.is_test:
            test = method(model=predictor.model,
                          test_interactions=self._test_it,
                          train_interactions=None if self.allow_rerecommendations else self._train_it,
                          user_features=predictor.translator.user_matrix,
                          item_features=predictor.translator.user_matrix,
                          num_threads=self._threads, **additional_args)
            test = test.mean()
        return float(train), float(test)

    @property
    def train_interactions(self) -> coo_matrix:
        return self._train_it

    @property
    def test_interactions(self) -> coo_matrix:
        return self._test_it

    @property
    def supported_metrics(self) -> Set[Metric]:
        return super().supported_metrics

    @property
    def predictors(self) -> Dict[str, LFMPredictor]:
        return self._predictors

    def __getitem__(self, predictor_key: str) -> LFMPredictor:
        return self.predictors[predictor_key]


# TODO(kdevo): This implementation is really slow
class PredictionGraphEvaluator(EpochBasedEvaluator):
    TRUE_POSITIVES = SingleMetric('true_positives', ('train', 'test'))
    FALSE_NEGATIVES = SingleMetric('false_negatives', ('train', 'test'))

    # RECALL = CompositeMetric('recall', ('train', 'test'), metrics=(TRUE_POSITIVES, FALSE_NEGATIVES))
    # PRECISION = CompositeMetric('recall', ('train', 'test'), metrics=(TRUE_POSITIVES, FALSE_NEGATIVES))

    def __init__(self, predictors: Dict[str, Predictor], interaction_graph: InteractionGraph,
                 test_split: float = 0.2,
                 reciprocal: bool = True,
                 show_progress: bool = True):
        super().__init__(predictors, test_split, show_progress)
        self._cg = StaticCandidateRepo(set(interaction_graph.node_names()))
        self._reciprocal = reciprocal

        # TODO(kdevo): Add warnings for limits, e.g. if too many test edges etc.
        rand_edges = list(interaction_graph.edges())
        random.shuffle(rand_edges)
        if self._reciprocal:
            test_edges = []
            seen_pairs = set()
            for e in rand_edges:
                if (e.v, e.u) in seen_pairs:
                    continue
                if r_e := interaction_graph.edge(e.v, e.u):
                    seen_pairs.add((e.u, e.v))
                    test_edges += [e, r_e]
                    if len(test_edges) >= (len(rand_edges) * self.test_split) / 2:
                        break
        else:
            test_edges = rand_edges[:math.ceil(len(rand_edges) * self.test_split)]

        self._test_graph = InteractionGraph.from_edges(test_edges, backend=InMemoryBackend())
        self._train_graph = interaction_graph.without_edges(test_edges, backend=InMemoryBackend())

    @timed(__name__, logging.INFO)
    def evaluate(self, predictor: Predictor, metric: Metric) -> Tuple[Optional[float], Optional[float]]:
        def calc(metric: Metric, prediction_graph: DiGraph, test_graph: DiGraph, train_graph: DiGraph = None):
            # TODO(kdevo): Add train parameter
            test_users = [e.u for e in test_graph.edges()]
            for u_test in test_users:
                tp = 0
                for v_pred in prediction_graph[u_test.name].neighbours:
                    if u_test.edge(v_pred.name) or (train_graph and train_graph[u_test.name].edge(v_pred.name)):
                        # If reciprocal, v needs to see u "from the other side", too:
                        if self._reciprocal and v_pred.edge(u_test.name):
                            tp += 1
                        # Otherwise, it does not matter if v is also recommended to u:
                        elif not self._reciprocal:
                            tp += 1

                u_test.data['precision'] = tp / metric.k
                u_test.data['recall'] = tp / len(u_test.edges)
            return sum([u.data[metric.name] for u in test_users]) / len(test_users)

        train, test = None, None
        prediction_graph = predictor.build_prediction_graph(k=metric.k, override_cg=self._cg)
        if metric.is_train:
            train = calc(metric, prediction_graph, self._train_graph)
        if metric.is_test:
            test = calc(metric, prediction_graph, self._test_graph)
        return train, test

    # TODO: Cache if k is the same for all
    # def run(self, predictor_key: str,
    #         single_metrics: Iterable[Metric],
    #         composite_metrics: Iterable[CompositeMetric], epochs) -> pd.DataFrame:
    #     prediction_net = self[predictor_key].build_predict_net(users=self._test_net.node_names(),
    #                                                  k=metric.k, reciprocal=self._reciprocal)

    @property
    def train_interactions(self) -> InteractionGraph:
        return self._train_graph

    @property
    def test_interactions(self) -> InteractionGraph:
        return self._test_graph

    @property
    def supported_metrics(self) -> Set[Metric]:
        return {self.PRECISION, self.RECALL, self.F1}

    def calc_stats(self):
        cold_start_nodes = set(filter(lambda n: len(n.edges) == 0, self._train_graph))
        # recommended_k = round(n_test_edges / len(interaction_graph))
        # logger.info(f"Recommended choice of k: {recommended_k}")
        # if recommended_k == 0:
        #     logger.warning(f"Only using {n_test_edges} interactions in the test set. "
        #                    f"This is less than the number of nodes ({len(interaction_graph)}), evaluation might be skewed.")
        df = pd.DataFrame({
            'cold_start': [len(cold_start_nodes), sum([n in cold_start_nodes for n in self._test_graph])],
            # TODO:
            'recommended_k': [None, None]
        }, index=['train', 'test'])
        return df
