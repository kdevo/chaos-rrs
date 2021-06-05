import logging
from typing import Callable, Any

import networkx as nx
import pandas as pd
from sklearn.preprocessing import minmax_scale

from chaos.process.processor import Processor, SocialNetExtractor
from chaos.shared.model import DataModel
from grapresso.components.edge import Edge
from grapresso.components.node import Node

logger = logging.getLogger(__name__)


class GraphEdgeMapper(Processor):
    def __init__(self, name: str = None, **kwargs: Callable[[Edge], Any]):
        super().__init__(name)
        self._attrib_to_fn = kwargs

    def execute(self, data: DataModel) -> DataModel:
        for edge in data.interaction_graph.edges():
            for attrib, fn in self._attrib_to_fn.items():
                val = fn(edge)
                edge[attrib] = val
        return data


class GraphNodeMapper(Processor):
    def __init__(self, name=None, **kwargs: Callable[[Node], Any]):
        super().__init__(name)
        self._attrib_to_fn = kwargs

    def execute(self, data: DataModel) -> DataModel:
        for n in data.interaction_graph.backend:
            for attrib, fn in self._attrib_to_fn.items():
                n.data[attrib] = fn(n)
        return data


class GraphNodeExtractor(Processor):
    def __init__(self, metrics, name=None):
        super().__init__(name)
        self._metrics = metrics

    def execute(self, data: DataModel) -> DataModel:
        if 'reciprocity' in self._metrics:
            reciprocity = nx.algorithms.reciprocity(data.interaction_graph.nxg)
            data = data.apply(GraphNodeMapper(reciprocity=lambda n: reciprocity[n]))
        return data


class GraphPopularityExtractor(SocialNetExtractor):
    """ Has direct dependency to networkx"""

    def __init__(self, target_col, metrics=('eigenvector',), quantiles=None, labels=None,
                 add_as_node_attrib=False, scale: bool = True):
        super().__init__(target_col=target_col)
        self._metrics = metrics
        self._labels = labels
        self._quantiles = quantiles
        self._metrics = metrics
        self._add_as_node_attrib = add_as_node_attrib
        self._scale = scale

    def execute(self, data: DataModel) -> DataModel:
        # TODO(kdevo): Optimize
        nxg = data.interaction_graph.nxg

        popularity = pd.Series(data=0, index=data.user_df.index)
        if 'eigenvector' in self._metrics:
            eigenvector = nx.eigenvector_centrality_numpy(nxg, weight='cost')
            logger.debug(f"Eigenvector centrality: {eigenvector}")
            popularity += minmax_scale(popularity.index.map(eigenvector)) if self._scale else popularity.index.map(
                eigenvector)
            if self._add_as_node_attrib:
                data = data.apply(GraphNodeMapper(eigenvector=lambda n: eigenvector[n]))

        if 'betweenness' in self._metrics:
            betweenness = nx.betweenness_centrality(nxg, weight='cost', normalized=True)
            logger.debug(f"Betweenness centrality: {popularity}")
            popularity += minmax_scale(popularity.index.map(betweenness)) if self._scale else popularity.index.map(
                betweenness)
            if self._add_as_node_attrib:
                data = data.apply(GraphNodeMapper(betweenness=lambda n: betweenness[n]))

        if 'degree' in self._metrics:
            degree = nx.degree_centrality(nxg)
            logger.debug(f"Degree centrality: {degree}")
            popularity += minmax_scale(popularity.index.map(degree)) if self._scale else popularity.index.map(degree)
            if self._add_as_node_attrib:
                data = data.apply(GraphNodeMapper(degree=lambda n: degree[n]))

        data.user_df[self.target_col] = minmax_scale(popularity)

        assert not (self._labels and not self._quantiles)
        if self._quantiles:
            data.user_df[self.target_col] = pd.qcut(data.user_df[self.target_col], self._quantiles,
                                                    labels=self._labels, retbins=self._labels is None)

        return data
