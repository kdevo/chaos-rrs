import logging
from collections import deque
from typing import Dict, Any

import pandas as pd
from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport
from graphql.language.ast import Document
from tqdm import tqdm

from chaos.fetch.source import Source
from chaos.shared.graph import InteractionGraph
from chaos.shared.model import DataModel
from chaos.shared.tools import timed

logger = logging.getLogger(__name__)


class GQLSource(Source):
    def __init__(self, gql_spec: Dict[str, Any],
                 endpoint: str,
                 headers: Dict[str, Any],
                 start_user: str,
                 profile_key: str,
                 neighour_node_key: str,
                 breadth: int = 8,
                 max_nodes: int = 5000):
        super().__init__(gql_spec['interactions'])
        self._endpoint = endpoint
        self._breadth = breadth
        self._max_nodes = max_nodes
        self._neighbour_key = neighour_node_key
        self._spec = gql_spec
        self._start_node = start_user
        self._profile_key = profile_key
        self._fragments = {it_key: {} for it_key in self._spec['interactions']}
        for it_key, it_val in self._spec['interactions'].items():
            for key, fragment_name in it_val.items():
                if 'fragment' in key:
                    fragment_type = key.replace('fragment', '').replace('_', '')
                    self._fragments[it_key]['bi' if fragment_type == '' else fragment_type] = fragment_name

        http_transport = RequestsHTTPTransport(self._endpoint, headers=headers)
        schema = None
        # if schema_path:
        #     pass
        # TODO(kdevo): Fix when gql reaches 3.0.0
        # with open(schema_path, 'r') as schema_file:
        #     schema = graphql.build_ast_schema(graphql.parse(schema_file.read()))
        self._client = Client(transport=http_transport, schema=schema)

    def source_data(self) -> DataModel:
        def fetch_user(u):
            query = gql(self._spec['fragments'] + self._spec['query'])
            data = self._client.execute(query, variable_values={self._neighbour_key: u})
            exclude_from_metadata = set()
            for it_key, fragment in self._fragments.items():
                for fragment_type, fragment_name in fragment.items():
                    fragment_location = self._locate_fragment(query, fragment_name)
                    for v in self.find_users(data, fragment_location) - {u}:
                        if fragment_type in ('out', 'bi'):
                            self.data.interaction_graph.add_interaction(u, v, it_key)
                        if fragment_type in ('in', 'bi'):
                            self.data.interaction_graph.add_interaction(v, u, it_key)
                    exclude_from_metadata.add(fragment_location)
            return {k: v for k, v in data[self._profile_key].items() if k not in exclude_from_metadata}
            # return data.get('rateLimit', default={'remaining': max_visited_nodes})['remaining']

        graph = InteractionGraph()
        metadata = {}
        # BFS-like interaction graph building with reciprocal constraint/lookahead
        seen_nodes = {self._start_node}
        to_visit = deque(maxlen=self._max_nodes)
        to_visit.append(self._start_node)
        for _ in tqdm(range(self._max_nodes), desc="⬇️ Retrieving user profiles via reciprocal BFS"):
            if len(to_visit) == 0:
                logger.warning(f"Could not fulfill maximum of depth {self._max_nodes}")
            u = to_visit.popleft()
            metadata[u] = fetch_user(u)
            # Only reciprocal connections are considered for iteration:
            best_neighbors = [e.v.name for e in sorted(
                filter(lambda e: e.v not in seen_nodes, self.data.interaction_graph.symmetric_edges(from_node=u)),
                key=lambda e: e.strength, reverse=True
            )[:self._breadth]]
            seen_nodes.update(best_neighbors)
            to_visit.extend(best_neighbors)

        # Only the nodes within the df can be trained, but the others are kept for network analysis purposes:
        self.data.user_df = pd.DataFrame.from_dict(metadata, orient='index')

        return self.data

    def _locate_fragment(self, query: Document, fragment_name):
        fragment = next(
            d for d in query.definitions if d.name.value == fragment_name
        )
        # Indicates where to find the fragment in the resulting data:
        return fragment.selection_set.selections[0].name.value

    @timed(__name__, logging.DEBUG)
    def find_users(self, res, interaction_location: str):
        """
        Helper method to (inefficiently and recursively) fetch users from fragment dict
        TODO(kdevo): Possibly easier to iterate through the schema - couldn't figure out a way to do it with gql 2.0 though
        """

        # TODO(kdevo): Getting a dict 'path.to.key' and then using this path for other entries in list should be faster
        def dig(parent_element, key):
            if e := parent_element.get(key):
                yield e
            for e in parent_element.values():
                if type(e) == dict:
                    yield from dig(e, key)
                elif type(e) == list:
                    for v in e:
                        if type(v) == dict:
                            yield from dig(v, key)

        return {user for element in dig(res, interaction_location) for user in dig(element, self._neighbour_key)}
