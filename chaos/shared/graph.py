import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, Hashable, Iterable

import matplotlib as mpl
import matplotlib.collections
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx import density
from sklearn.preprocessing import minmax_scale

from chaos.shared.mixins import VisualizableMixin
from chaos.shared.tools import timed
from grapresso import DiGraph
from grapresso.backends import NetworkXBackend
from grapresso.backends.api import DataBackend
from grapresso.components.edge import Edge

mpl.rcParams['font.sans-serif'] = ['Comfortaa', 'Verdana', 'DejaVu Sans']

logger = logging.getLogger(__name__)


@dataclass(frozen=True, repr=False)
class Interaction:
    event: str
    strength: float

    def __repr__(self):
        return f"{self.event}"

    def __str__(self):
        return self.__repr__()


class InteractionGraph(DiGraph, VisualizableMixin):
    # FIXME(kdevo): Intentional side-effect to recycle node layout throughout multiple InteractionGraph objects
    seed_to_layout = {}

    def __init__(self, interactions_spec: Dict = None, data_backend: DataBackend = None, only_store_strength=False):
        """
        Construct new social network to save interactions between users.

        Args:
            interactions_spec: Mapping, e.g. {'view': {'strength': 1.0}}
            data_backend: Graph backend for storing the nodes and edges
            only_store_strength: Set to True if you do not want full interaction records and only the cumulated strength.
              For instance, if user A viewed the profile of B two times (a, b, [view, view]), then only store
              (a, b, 2) instead.
        """
        super().__init__(data_backend)
        if interactions_spec:
            self._interactions_spec = interactions_spec
        else:
            self._interactions_spec = defaultdict(default_factory=defaultdict(float=0.0))

        self._only_strength = only_store_strength

    @property
    def nxg(self) -> nx.DiGraph:
        """
        Get a view over the NetworkX backend.

        Returns:
            Networkx DiGraph view
        """
        nxg_copy = self.copy_to(InteractionGraph(self._interactions_spec, NetworkXBackend(), self._only_strength))
        return nxg_copy.backend.nx_graph

    def add_interaction(self, from_user: Hashable, to_user: Hashable, event: str):
        edge = self.edge(from_user, to_user)
        if edge is None:
            self.add_edge(from_user, to_user, interactions=None if self._only_strength else [], strength=0)
            edge = self[from_user][to_user]

        interaction_strength = self._interactions_spec[event]['strength']
        if not self._only_strength:
            edge.interactions.append(Interaction(event, interaction_strength))
        edge['strength'] += interaction_strength

    @timed(__name__)
    def draw(self, title=None, layout_id=1, with_edge_labels=False, strength_label='strength',
             node_size_attribute=None):
        plt.figure(dpi=150)
        if layout_id in InteractionGraph.seed_to_layout and len(self) == len(
                InteractionGraph.seed_to_layout[layout_id]):
            # If there already is a layout available, reuse it:
            logger.debug(f"Re-use existing layout with seed {layout_id} for drawing.")
        else:
            # Create new layout:
            InteractionGraph.seed_to_layout[layout_id] = nx.layout.spring_layout(
                self.nxg,
                k=4.0 / math.sqrt(len(self.nxg)),
                weight=strength_label,
                seed=layout_id
            )
        node_sizes = self._draw_nodes(layout_id, size_attribute=node_size_attribute)
        self._draw_edges(layout_id, with_labels=with_edge_labels, strength_label=strength_label, node_size=node_sizes)

        ax = plt.gca()
        ax.set_title(title if title else self.__class__.__name__, loc='left')
        ax.set_axis_off()
        plt.show()

    def _draw_nodes(self, layout_id, color='#0097a7', with_labels=True, size_attribute=None):
        # color = "#04a6b8"
        base_size = 300
        if len(self.nxg.nodes) > 50:
            base_size = 50
        if size_attribute:
            size = minmax_scale([d.get(size_attribute, 0) for n, d in self.nxg.nodes.data()],
                                (base_size / 2, base_size * 4))
        else:
            size = base_size
        nx.draw_networkx_nodes(self.nxg, InteractionGraph.seed_to_layout[layout_id],
                               node_shape='o', node_size=size,
                               node_color=color, alpha=0.8)
        if len(self.nxg.nodes) <= 50 and with_labels:
            nx.draw_networkx_labels(self.nxg, InteractionGraph.seed_to_layout[layout_id], font_size=8,
                                    font_weight='bold',
                                    labels={n: f"{str(n)[0:2]}." for n in self.nxg.nodes.keys()})
        return size

    def _draw_edges(self, layout_id, colormap='viridis', with_labels=False, strength_label='strength', node_size=None):
        width = 1.4
        if len(self.nxg.nodes) > 50:
            width = 0.8
        edge_strengths = [e[2][strength_label] for e in self.nxg.edges.data()]
        additional_params = {}
        if colormap:
            additional_params = {'edge_color': edge_strengths,
                                 'edge_cmap': plt.cm.get_cmap(colormap)}
        edges = nx.draw_networkx_edges(self.nxg, InteractionGraph.seed_to_layout[layout_id],
                                       width=width,
                                       # edge_vmin=0.0,
                                       # edge_vmax=1.0,

                                       connectionstyle='arc3,rad=0.1',
                                       arrowstyle='->' if width > 1.0 else '-',
                                       # min_source_margin=12, min_target_margin=10,
                                       node_shape='o',
                                       node_size=node_size,
                                       alpha=0.9, **additional_params)

        if with_labels:
            edge_labels = {(e[0], e[1]): round(e[2][strength_label], 2) for e in self.nxg.edges.data()}
            nx.draw_networkx_edge_labels(self.nxg, InteractionGraph.seed_to_layout[layout_id],
                                         label_pos=0.6, rotate=True,
                                         font_size=8, alpha=0.5,
                                         edge_labels=edge_labels)
        if colormap:
            pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.get_cmap(colormap))
            pc.set_array(np.asarray(edge_strengths))
            plt.colorbar(pc)

    def merge(self, other_graph: 'InteractionGraph') -> 'InteractionGraph':
        return self.union(other_graph)

    def density(self):
        return density(self.nxg)

    def visualize(self, **kwargs):
        self.draw(**kwargs)

    def without_edges(self, edges: Iterable[Edge], backend: DataBackend = None):
        graph = self.copy_to(InteractionGraph(self._interactions_spec, backend, self._only_strength))
        for e in edges:
            graph.backend.remove_edge(e.u.name, e.v.name)
        return graph

    def __setstate__(self, state):
        self._interactions_spec, self._only_strength, self._nodes_data = state

    def __getstate__(self):
        return self._interactions_spec, self._only_strength, self._nodes_data
