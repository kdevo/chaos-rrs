import itertools
import logging
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, List, Dict, Union, Iterable, Collection, Hashable

import networkx
import numpy as np
import pandas as pd
from lightfm.data import Dataset
from scipy import sparse
from scipy.sparse import coo_matrix

from chaos.shared.graph import InteractionGraph
from chaos.shared.model import DataModel
from chaos.shared.user import User, UserType, user_id
from grapresso import DiGraph
from grapresso.components.node import Node

logger = logging.getLogger(__name__)

Features = Union[List[Union[str, tuple]], Dict[str, Union[str, Iterable[str]]]]


@dataclass()
class Translator(ABC):
    """Translator acts as the bridge between our generic data Model class and the Predictor classes' bare-bone model.

    Implementations should provide the highly model-specific translation logic that is often necessary.
    See LFMTranslator for a LightFM-specific predictor that provides an interaction matrix and a user matrix which
        can be directly fed into a LightFM model.
    No assumptions are made on the Predictor classes internal model, i.e. the type is set to 'Any' below.
    """

    dm: DataModel
    features: Union[List[str], Dict[str, float]] = None
    predictor_dm: Any = field(init=False)
    dynamic_re: re.Pattern = field(default=re.compile(r'^.*_?(tags|tokens)$', flags=re.I))

    _train_interactions: Any = field(init=False)

    @abstractmethod
    def to_predictor_user(self, user: User) -> Any:
        pass

    @abstractmethod
    def to_model_user(self, user: Any) -> User:
        pass

    def collect_dynamic_features(self):
        return {col for col in self.dm.user_df.columns.values if col in self.features and self.dynamic_re.match(col)}

    @abstractmethod
    def translate_interactions(self, graph: InteractionGraph) -> Any:
        pass

    @property
    @abstractmethod
    def train_interactions(self) -> Any:
        pass

    @train_interactions.setter
    @abstractmethod
    def train_interactions(self, interactions: Union[InteractionGraph, Any]):
        pass


@dataclass
class GraphTranslator(Translator):
    _train_interactions: DiGraph = field(init=False)

    def __post_init__(self):
        self._train_interactions = self.dm.interaction_graph

    def to_predictor_user(self, user: User) -> Node:
        return user.node

    def to_model_user(self, user: Node) -> User:
        return self.dm.get_user(user.name)

    def translate_interactions(self, graph: InteractionGraph) -> DiGraph:
        return graph

    @property
    def train_interactions(self) -> DiGraph:
        return self._train_interactions

    @train_interactions.setter
    def train_interactions(self, interactions: Union[InteractionGraph, Any]):
        self._train_interactions = interactions


@dataclass()
class LFMTranslator(Translator):
    """ LFMTranslator is an implementation of Translator for the default predictor

    Only interactions that involve users who are present in dm.user_df will be considered.
    """
    use_indicator: bool = True
    """ Setting this during init will override the translated interaction matrix. """
    interaction_matrix: coo_matrix = None

    interaction_weight_key: str = 'strength'

    """ Tuple of the form (interaction_matrix, user_matrix)"""
    predictor_dm: tuple = field(init=False, default_factory=tuple)
    """ User matrix in shape: (n_users + n_feat, n_users + n_feat) if use_indicator = True
    otherwise, if use_indicator = False: (n_feat, n_feat), no indicator features used then
    """

    normalize_feat_weights: bool = True
    user_matrix: coo_matrix = field(init=False)
    user_id_mapping: pd.Series = field(init=False)

    _train_interactions: coo_matrix = field(init=False)

    def __post_init__(self):
        if type(self.features) == list:
            self.features = {f: 1.0 for f in self.features}
        logger.info(f"Translating with features: {self.features}")
        self.predictor_dm = self._translate(
            self.features,
            self.use_indicator
        )
        if self.interaction_matrix is None:
            self.interaction_matrix = self.predictor_dm[0]
        self.user_matrix = self.predictor_dm[1]
        self._train_interactions = self.interaction_matrix

        if self.is_using_features and self.use_indicator:
            logger.info(f"Model is in HYBRID mode (content-collaborative-filtering) with features: {self.features}")
        elif not self.is_using_features:
            logger.warning("Model is in COLLABORATIVE mode with indicator features only. "
                           "ATTENTION: Degraded cold-start performance!)")
        elif self.is_using_features:
            logger.warning(
                "Only relying on user profile_data. No indicator features in use. "
                "ATTENTION: Model is likely to perform worse if features are limited as per-user information is limited.")
        else:
            raise ValueError("Model is neither using indicator features for collaborative filtering "
                             "nor using profile_data for content-based filtering.")

    def _translate(self, feat_to_weight: Dict[str, float] = None, identity_features=True):
        ds = Dataset(user_identity_features=identity_features, item_identity_features=identity_features)

        # Metadata - Collect selected feature values:
        if self.is_using_features:
            user_dict = self.dm.user_df[feat_to_weight.keys()].to_dict('index')
            feat_label_counter = Counter()
            user_records = []
            dynamic_feat = self.collect_dynamic_features()
            logger.info(f"Dynamic-length features: {dynamic_feat}")
            # We iterate to built a LFM-Dataset compatible class
            # Not all feat are known in advance, i.e. some may need to be expanded (e.g. if cell is iterable/bag-of-words)
            for u, features in user_dict.items():
                ufeat_label_counter = Counter()
                for feat, val in features.items():
                    if val is None:
                        continue
                    if feat in dynamic_feat:
                        if type(val) == str:
                            val = [val]
                        if not isinstance(val, Iterable):
                            raise ValueError(
                                f"Dynamic feature {feat} needs to be iterable (value is '{val}' for user '{u}')")
                        logger.debug(f"Dynamic-length feature: {feat}")
                        for v in val:
                            ufeat_label_counter[(feat, v)] += 1
                    else:
                        if not isinstance(val, Hashable):
                            raise ValueError('Ensure to name features appropriately if they are dynamic-length! '
                                             f'Naming pattern: {self.dynamic_re} '
                                             f'The feature {feat} with value {val} can not be handled (not hashable).')
                        ufeat_label_counter[(feat, val)] += 1

                ufeat_counter = {
                    g: len(set(c)) for g, c in itertools.groupby(ufeat_label_counter.keys(), key=lambda k: k[0])
                }
                user_records.append((u, {
                    self.build_feat_label(f, v): 1 / ufeat_counter[f] * feat_to_weight[f]
                    for f, v in ufeat_label_counter.keys()
                }))
                feat_label_counter += ufeat_label_counter

            all_labels = [self.build_feat_label(f, v) for f, v in feat_label_counter.keys()]
            ds.fit(user_dict.keys(), user_dict.keys(), all_labels, all_labels)
            # Building item features vs. building user features should not make a difference for RRS
            user_matrix = ds.build_user_features(user_records, self.normalize_feat_weights)
            logger.info(
                f"Built a total of {len(feat_label_counter)} unique profile_data features with {feat_to_weight}"
            )
            # TODO: Actually make use of the following mapping df in LFMPredictor class etc.:
            self.user_id_mapping = pd.Series(ds.mapping()[0])
            feat_mapping = ds.mapping()[1]
            index = pd.MultiIndex.from_tuples(list(feat_label_counter.keys()), names=('feature', 'label'))
            if identity_features:
                index = pd.MultiIndex.from_tuples((('#', u) for u in self.user_id_mapping.index),
                                                  names=('feature', 'label')).append(index)
            count = [1] * self.use_indicator * len(self.user_id_mapping) + list(feat_label_counter.values())
            weight = [1.] * self.use_indicator * len(self.user_id_mapping) + [feat_to_weight[f] for f, _ in
                                                                              feat_label_counter.keys()]
            self.feat_mapping = pd.DataFrame(
                {'count': count,
                 'weight': weight,
                 'index': feat_mapping.values(),
                 'is_indicator': [self.use_indicator and f in self.user_id_mapping.keys()
                                  for f in feat_mapping.keys()]
                 }, index=index)
            self.feat_mapping.sort_index(level=0, inplace=True)
        else:
            ds.fit(self.dm.user_df.index.values, self.dm.user_df.index.values)
            user_matrix = ds.build_user_features([])
            logger.info("No feature keys provided: Did not built profile features.")
            self.user_id_mapping = pd.Series(ds.mapping()[0])

        # Interactions - Format to a LightFM compatible format (u, v, weight):
        # TODO(kdevo): Check that nodes are also present in df
        interaction_records = [(edge.from_node.name, edge.to_node.name, edge[self.interaction_weight_key])
                               for edge in self.dm.interaction_graph.edges()]

        # We build an interaction matrix in shape of (num_users, num_users), with weights as values
        (_, interaction_matrix) = ds.build_interactions(interaction_records)

        model_dimensions = ds.model_dimensions()
        assert model_dimensions[0] == model_dimensions[1]
        logger.info(f"Predictor model dimensions: {model_dimensions[0]} × {model_dimensions[1]}")
        logger.info(f"Predictor interactions shape: {interaction_matrix.shape[1]} × {interaction_matrix.shape[-1]}")
        return interaction_matrix, user_matrix

    def translate_interactions(self, graph: InteractionGraph) -> coo_matrix:
        # Partly adapted from networkx.convert_matrix.to_scipy_sparse_matrix
        nlen = len(graph)
        index = dict(zip(iter(graph), range(nlen)))
        coefficients = zip(*(
            (index[e.u], index[e.v], e.strength)
            for e in graph.edges()
        ))
        row, col, data = coefficients
        coo = sparse.coo_matrix((data, (row, col)), shape=(nlen, nlen))
        return coo

    def to_predictor_user(self, user: UserType) -> int:
        num_id = self.user_id_mapping.loc[user_id(user)]
        return num_id

    def predictor_users(self, users: Collection[UserType]) -> np.array:
        other_user_ids = np.array([self.to_predictor_user(u) for u in users])
        return other_user_ids

    def feat_matrix(self, user: UserType, to_coo_matrix=True) -> Union[sparse.coo_matrix, np.array]:
        if uid := user_id(user):
            try:
                return self.user_matrix.getrow(self.user_id_mapping.loc[uid]).tocoo()
            except KeyError:
                raise ValueError(f"User ID {user.id} is untrained!")

        feat_column_indices = self.feat_indices(user.profile_data)

        # For building a normalized inverse feat frequency weight vector:
        index_df = self.feat_mapping.reset_index().set_index('index').sort_index()
        ifq = 1 / index_df.loc[feat_column_indices].groupby(['feature'])['label'].count()
        ifq_vec = np.zeros((1, len(self.feat_mapping)))
        ifq_vec.put(feat_column_indices, [ifq[index_df['feature'].loc[idx]] for idx in feat_column_indices])
        weight_vec = np.zeros((1, len(self.feat_mapping)))
        weight_vec.put(feat_column_indices, index_df['weight'].loc[feat_column_indices].values)
        weight_vec = ifq_vec * (weight_vec / len(ifq))

        return coo_matrix(weight_vec) if to_coo_matrix else weight_vec

    def feat_indices(self, features: Features) -> List[int]:
        try:
            if type(features) == dict:
                feat_column_indices = self.feat_mapping['index'].loc[
                    [(f, i) for f, l in features.items() for i in itertools.chain([l] if type(l) == str else l)]
                ]
            else:
                feat_column_indices = self.feat_mapping['index'].loc[
                    [self.split_feat_label(fl) if type(fl) == str else fl for fl in features]
                ]
            return feat_column_indices.values
        except KeyError:
            raise ValueError(f"One of the features has not been fitted during training or dataset construction.")

    @staticmethod
    def build_feat_label(key, attribute):
        return f"{key}:{attribute}"

    @staticmethod
    def split_feat_label(feat_label: str):
        first_idx = feat_label.find(':')
        return feat_label if first_idx == -1 else (feat_label[:first_idx], feat_label[first_idx + 1:])

    def to_model_user(self, user: Union[str, int, User]) -> User:
        if isinstance(user, User):
            return user
        elif type(user) == str:
            return self.dm.get_user(user)
        else:
            # TODO(kdevo): Check explicit dtype
            return self.dm.get_user(self.user_id_mapping[self.user_id_mapping == user].index[0])

    @property
    def is_using_features(self):
        return self.features is not None and len(self.features) >= 1

    def __str__(self):
        return f"Hybrid model: {'with indicator ' if self.use_indicator else 'without indicator '}{', feat: ' + str(self.features)}"

    @property
    def train_interactions(self) -> coo_matrix:
        return self._train_interactions

    @train_interactions.setter
    def train_interactions(self, interactions: Union[InteractionGraph, coo_matrix]):
        if isinstance(interactions, InteractionGraph):
            self._train_interactions = self.translate_interactions(interactions)
        else:
            self._train_interactions = interactions
