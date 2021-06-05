from abc import abstractmethod, ABC
from typing import Optional, Dict, Callable, List

import numpy as np

from chaos.recommend.candidates import CandidateRepo
from chaos.recommend.predict.predictions import PredictionGraph
from chaos.recommend.predict.predictor import Predictor, ModelBasedPredictor, MemoryBasedPredictor
from chaos.recommend.translator import GraphTranslator
from chaos.shared.model import DataModel, UserType
from chaos.shared.user import user_id
from grapresso import DiGraph
from grapresso.components.node import Node


class PreferenceAggregationStrategy(ABC):
    @abstractmethod
    def fuse(self, u, v, u_preferences: Dict[str, float], v_preferences: Dict[str, float]) -> float:
        """
        Fuses two preference scores together and puts them in both, u's and v's preference lists.

        Args:
            u: Key of u
            v: Key of v
            u_preferences: Preference dict of u
            v_preferences: Preference dict of v

        Returns:
            aggregation(u, v)

        """
        pass

    @abstractmethod
    def score(self, u2v: float, v2u: float) -> float:
        pass


class ArithmeticStrategy(PreferenceAggregationStrategy):
    """
    An arithmetic strategy takes two scores as input and returns one aggregated score.

    Some means/functions are more effective than others in satisfying both parties interests,
    e.g. the harmonic mean has been proven as useful in REs (among others).
    """
    Function = Callable[[float, float], float]

    # Functions are ordered descending (max > qm > am > gm > hm > un >= min)
    maximum: Function = lambda u, v: max(u, v)
    quadratic_mean: Function = lambda u, v: np.sqrt(((u ** 2 + v ** 2) / 2))

    # The three Pythagorean means ordered descending
    arithmetic_mean: Function = lambda u, v: (u + v) / 2
    geometric_mean: Function = lambda u, v: np.sqrt(u * v)
    harmonic_mean: Function = lambda u, v: (2 * u * v) / (u + v)

    # Cross-ratio uninorm as first used in a RRS in DOI: 10.1109/SMC.2019.8914362
    uninorm: Function = lambda u, v: (u * v) / ((u * v) + (1 - u) * (1 - v))

    minimum: Function = lambda u, v: min(u, v)

    def __init__(self, func: Function, uw: float = 1.0, vw: float = 1.0):
        self._function = func
        self._uw = uw
        self._vw = vw

    def fuse(self, u, v, u_preferences: Dict[str, float], v_preferences: Dict[str, float]) -> float:
        score = self.score(u_preferences.get(v, 0.0), v_preferences.get(u, 0.0))
        u_preferences[v] = score
        v_preferences[u] = score
        return score

    def score(self, u: float, v: float) -> float:
        # TODO(kdevo): Fix proper limit setting (e.g. uninorm)
        #   For instance, this can be accomplished by adding a numpy limit constructor param
        score = round(self._function(max(u, 1e-5) * self._uw, max(v, 1e-5) * self._vw), 4)
        return score


class Strategies:
    HARMONIC = ArithmeticStrategy(ArithmeticStrategy.harmonic_mean)
    UNINORM = ArithmeticStrategy(ArithmeticStrategy.uninorm)


class ReciprocalWrapper(ModelBasedPredictor):
    def __init__(self, u2u_predictor: Predictor,
                 aggregation_strategy: PreferenceAggregationStrategy = Strategies.HARMONIC,
                 enable_cache=False, enable_stats=True,
                 ku_factor: float = 4, kv_factor: float = 2):
        """
        Wrapper for non-reciprocal predictors to retrieve recommendation based on score aggregation
            (see preference aggregation strategies).

        Args:
            u2u_predictor: Non-reciprocal user-to-user recommender. For RRS, the wrapper should not change any recommendation results.
            aggregation_strategy: Selectable aggregation strategy
        """
        super().__init__(u2u_predictor.translator)
        self._predictor = u2u_predictor

        # Cache:
        self._prediction_net = PredictionGraph()
        self._with_cache = enable_cache

        self._aggregation = aggregation_strategy
        self._dm = self.translator.dm
        self._stats_enabled = enable_stats
        self._ku_factor = ku_factor
        self._kv_factor = kv_factor
        self._stats = {'rank_violations': {}}

    def predict(self, u: UserType, k: Optional[int] = 5) -> Dict[str, float]:
        """
        Reciprocal predictions using this wrapper have a O(...) runtime

        Approach:
          1. Predict k objects for subject user
          2. Predict for each object the score for user u
          3. Aggregate preferences in a specified way

        Args:
            u: User to predict for.
            k: Predict top k*k_lookahead objects, then predict for each object.

        Returns:
            Dict of user to score.
        """
        if not (u := user_id(u)):
            # TODO(kdevo): One way of handling unknown users would either be:
            #   - Find a similar user to use instead
            #   - Only use the one-way recommendations

            raise ValueError("Unknown users are not yet supported!")

        ku = min(round(k * self._ku_factor), len(self._dm.interaction_graph))
        kv = min(round(k * self._kv_factor), len(self._dm.interaction_graph))

        u_pref = self._predictor.predict(u, ku)
        for v in u_pref:
            # Cache predictions:
            if (v_pref := self._from_cache(v, kv)) is None:
                v_pref = self._predictor.predict(v, kv)
                self._add_to_cache(v, v_pref)
            self._aggregation.fuse(u, v, u_pref, v_pref)

        if self._stats_enabled:
            # Calculate number of "rank violations"
            #  Good to measure how the RRS approach "performs better": the more violations,
            #    the more the RRS wrapper is needed to make truly reciprocal recommendations.
            items = list(u_pref.values())
            self._stats['rank_violations'][u] = sum([items[i] < items[i + 1] for i in range(len(items) - 1)])
        return {v: s for v, s in sorted(u_pref.items(), key=lambda vs: -vs[1])[:k]}

    def _from_cache(self, user: str, for_k: int):
        if not self._with_cache:
            return None
        try:
            user_node = self._prediction_net[user]
            # k needs to be smaller than cached edges, otherwise consider as cache miss
            if len(user_node) >= for_k:
                return {e.v: e.strength for e in user_node.edges}
            else:
                return None
        except KeyError:
            return None

    def _add_to_cache(self, user: str, scores):
        self._prediction_net.add_predictions(user, scores)

    @property
    def is_trained(self):
        return self._predictor.is_trained if isinstance(self._predictor, ModelBasedPredictor) else True

    def train(self, epochs: int, resume: bool = False, **kwargs):
        if isinstance(self._predictor, ModelBasedPredictor):
            self.invalidate_cache()
            return self._predictor.train(epochs, resume, **kwargs)

    def invalidate_cache(self):
        # Invalidate cache
        if len(self._prediction_net) > 0:
            self._prediction_net = PredictionGraph()

    @property
    def stats(self):
        return self._stats


class RCFPredictor(MemoryBasedPredictor):
    def __init__(self, translator: GraphTranslator, candidate_generator: CandidateRepo,
                 aggregation_strategy: PreferenceAggregationStrategy = Strategies.HARMONIC,
                 neighbour_direction: str = 'in',
                 similarity_measure: str = 'interest'):
        """
        Partial RCF algorithm implementation, see https://arxiv.org/pdf/1501.06247.pdf page 6 and
          DOI 10.1109/SMC.2019.8914362 for comparison of aggregation functions in this context.

        RCF is a memory-based RRS that works by either using attractiveness similarity or interest similarity
        to estimate similar users that can be recommended to each other.

        Args:
            dm: DataModel to get interaction graph from.
            candidate_generator: Retrieve candidates for recommendation user.
            aggregation_strategy: Strategy to choose to aggregate preferences (u, v)
        """
        super().__init__(translator, candidate_generator)
        self._translator = translator
        self._agg = aggregation_strategy
        self._neighbours: Callable[[Node], List[Node]] = self.in_neighbours if neighbour_direction == 'in' else self.out_neighbours
        self._similarity: Callable[[Node, Node], float] = self.interest_similarity if similarity_measure == 'interest' else self.attraction_similarity

    @staticmethod
    def jaccard(s1, s2):
        return len(s1 & s2) / len(s1 | s2)

    def in_neighbours(self, u):
        return [e.u for e in self.graph.edges(None, u)]

    def out_neighbours(self, u):
        return [e.v for e in self.graph.edges(u, None)]

    def interest_similarity(self, u: Node, v: Node):
        return self.jaccard(set(self.out_neighbours(u)), set(self.out_neighbours(v)))

    def attraction_similarity(self, u: Node, v: Node):
        return self.jaccard(set(self.in_neighbours(u)), set(self.in_neighbours(v)))

    def weighted_interest_similarity(self, u: Node, v: Node):
        # TODO(kdevo): Test and validate, adjust normalization
        u_adj = self.graph[u].adj
        v_adj = self.graph[v].adj
        u_set = set(e.v for e in u_adj.values())
        v_set = set(e.v for e in v_adj.values())
        sim_nodes = u_set & v_set
        return sum([(u_adj[n].strength + v_adj[n].strength) / 2 for n in sim_nodes])

    def predict(self, u: UserType, k: int = 5) -> Dict[str, float]:
        if not (u := user_id(u)):
            # TODO: Find via content-based similarity
            raise ValueError("RCF algorithm is not able to handle cold start/unknown users!")
        u = self.dm.interaction_graph[u]

        neighbours = self._neighbours
        similarity = self._similarity

        scores = {}
        for v in [self.graph[c] for c in self._candidate_generator.retrieve_candidates(u)]:
            score_uv = 0.0
            score_vu = 0.0
            v_neighbours = neighbours(v)
            for vn in v_neighbours:
                score_uv += similarity(u, vn)

            u_neighbours = neighbours(u)
            for un in u_neighbours:
                score_vu += similarity(v, un)

            if len(v_neighbours) > 0: score_uv /= len(v_neighbours)
            if len(u_neighbours) > 0: score_vu /= len(u_neighbours)
            scores[v] = self._agg.score(score_uv, score_vu)

        return {v.name: s for v, s in sorted(scores.items(), key=lambda vs: vs[1], reverse=True)[:k]}

    @property
    def dm(self) -> DataModel:
        return self._translator.dm

    @property
    def graph(self) -> DiGraph:
        return self._translator.train_interactions
