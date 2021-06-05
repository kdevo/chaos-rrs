from chaos.shared.graph import InteractionGraph
from grapresso.backends import NetworkXBackend
from grapresso.backends.api import DataBackend


class PredictionGraph(InteractionGraph):
    """
    Even for RRS, the prediction network is modeled as a directed graph.
    This is because if user a user b in the top k recommendation list,
      it does not necessarily mean that user b has user a in the top k recommendation list.
    """

    def __init__(self, data_backend: DataBackend = None):
        super().__init__(None, data_backend=data_backend)

    def add_predictions(self, user, prediction_scores):
        for n, s in prediction_scores.items():
            self.add_edge(user, n, strength=s)

    def k(self, user):
        return len(self.backend[user])

    def add_prediction(self, u, v, score):
        self.add_edge(u, v, strength=score)

    def only_reciprocal(self) -> 'PredictionGraph':
        net = PredictionGraph()
        # Add all nodes (not only the ones with symmetric edges, see below):
        for n in self:
            net.add_node(n.name, **n.data)
        # Add symmetric edges only:
        for e in self.symmetric_edges():
            net.add_prediction(e.u.name, e.v.name, e.strength)

        return net

    def to_reciprocal(self) -> 'ReciprocalPredictionGraph':
        net = ReciprocalPredictionGraph()
        for n in self:
            net.add_node(n.name, **n.data)
        for e in self.symmetric_edges():
            # TODO: Add aggregation function to apply
            net.add_prediction(e.u.name, e.v.name, e.strength)
        return net


class ReciprocalPredictionGraph(PredictionGraph):
    def __init__(self):
        super().__init__(NetworkXBackend(directed=False))
