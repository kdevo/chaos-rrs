import logging
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Dict, Set, Hashable

import pandas as pd
from networkx import reciprocity

from chaos.process.processor import Processor
from chaos.shared.graph import InteractionGraph
from chaos.shared.mixins import PersistableMixin
from chaos.shared.tools import timed
from chaos.shared.user import User, UserType, user_id

logger = logging.getLogger(__name__)


class UserRepository(ABC):
    @abstractmethod
    def get_user(self, uid: Hashable) -> User:
        pass

    @abstractmethod
    def get_user_preferences(self, uid: Hashable) -> str:
        pass


@dataclass
class DataModel(PersistableMixin, UserRepository):
    interaction_spec: Dict

    user_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame({'preference_filter': ''},
                                                                       index=pd.Index(data=[], name=DataModel.id_col)))
    interaction_graph: InteractionGraph = field(init=False)

    id_col: str = 'id'
    preference_filter_col = 'preference_filter'

    def __post_init__(self):
        self.interaction_graph = InteractionGraph(self.interaction_spec)

    @property
    def user_ids(self) -> Set[str]:
        return set(self.user_df.index.values)

    def get_user(self, user: UserType) -> User:
        uid = user_id(user)
        row = self.user_df.loc[uid, self.user_df.columns != self.preference_filter_col]
        # Unpack preference filter:
        preference_filter = None
        if self.preference_filter_col in self.user_df:
            preference_filter = self.get_user_preferences(uid)
        return User(uid, row.to_dict(), preference_filter, self.interaction_graph[uid])

    def __len__(self):
        return len(self.user_df)

    def describe(self):
        print("Network density:", self.interaction_graph.density())
        print("Reciprocity:", reciprocity(self.interaction_graph.nxg))
        print("Cold start nodes:", {u for u in self.interaction_graph if len(u.edges) == 0} & self.user_ids)
        print(self.user_df.describe())

    def apply(self, proc: Processor):
        return proc.execute(self)

    @timed(__name__, logging.INFO)
    def sync_graph(self, inplace=False) -> InteractionGraph:
        """
        Synchronizes graph with df. Nodes not in the df will be dropped.

        Args:
            inplace: Perform action in-place

        Returns:
            None graph if inplace else new graph

        """
        ignore = set(self.interaction_graph.nxg.nodes) - set(self.user_ids)
        graph = InteractionGraph(self.interaction_spec)
        for e in self.interaction_graph.edges():
            if e.u not in ignore and e.v not in ignore:
                graph.add_edge(e.u.name, e.v.name, **e.data)
        if inplace:
            self.interaction_graph = graph
        else:
            return graph

    def get_user_preferences(self, user: UserType) -> str:
        uid = user_id(user)
        preferences = self.user_df.loc[uid, self.preference_filter_col]
        return None if preferences == '' else preferences

    def get_node(self, uid: UserType):
        return self.interaction_graph[user_id(uid)]
