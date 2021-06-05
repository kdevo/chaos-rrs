import math
import random

import networkx
import pandas as pd

from chaos.process.extract.graph import GraphEdgeMapper
from chaos.recommend.candidates import DMCandidateRepo
from chaos.recommend.evaluate.evaluator import PredictionGraphEvaluator, LFMEvaluator, Evaluator, SingleMetric, \
    CompositeMetric, f1_score
from chaos.recommend.predict.predictor import LFMPredictor
from chaos.recommend.predict.reciprocal import RCFPredictor, ReciprocalWrapper, Strategies
from chaos.recommend.translator import LFMTranslator, GraphTranslator
from chaos.shared.model import DataModel


# TODO(kdevo): Refactor to actual unit tests instead of "runbook-like"
class TestEvaluator:
    def test_socialnet_eval(self, default_dm: DataModel):
        cg = DMCandidateRepo(default_dm, True)
        translator = LFMTranslator(default_dm, features=['course'])
        hybrid = LFMPredictor(translator, candidate_generator=cg)
        predictors = {'lfm hybrid': hybrid}

        metrics = [SingleMetric('precision', k=5), SingleMetric('recall', k=5),
                   CompositeMetric.from_metrics('f1', func=f1_score, metrics=(SingleMetric('precision', k=5), SingleMetric('recall', k=5)))]

        lfm_eval = LFMEvaluator(predictors, translator.interaction_matrix)
        r1 = lfm_eval.run_all(metrics, epochs=range(0, 10, 1))
        # lfm_eval.create_report().show()

        predictors['lfm wrapped'] = ReciprocalWrapper(LFMPredictor(translator, cg), enable_cache=True)
        predictors['lfm cf'] = LFMPredictor(LFMTranslator(default_dm, []), cg)
        predictors['rcf1 n+ attract harmonic'] = RCFPredictor(GraphTranslator(default_dm), cg, similarity_measure='attract', neighbour_direction='out')
        predictors['rcf1 n+ attract uninorm'] = RCFPredictor(GraphTranslator(default_dm), cg, aggregation_strategy=Strategies.UNINORM, similarity_measure='attract', neighbour_direction='out')
        predictors['rcf2 n- interest'] = RCFPredictor(GraphTranslator(default_dm), cg)
        eval = PredictionGraphEvaluator(predictors, default_dm.interaction_graph, reciprocal=True)
        # r2 = eval.run_all(metrics, epochs=range(0, 15, 1))
        # eval.create_report('overview', on=('train', 'test',)).show()
        # default_dm.interaction_graph.draw("Complete")
        # eval.train_interactions.draw("Train")
        # eval.test_interactions.draw("Test")
        # eval['rcf'].build_predict_net(k=2).draw("RCF")
        # eval['lfm'].build_predict_net(k=2).draw("LFM")

        # TODO: add assertions

    def test_bipartite_indicator(self):
        BI = True
        D = 0.2
        N = 20
        U = list(range(N))
        V = list(range(len(U), len(U) + N))
        all_nodes = len(U) + len(V)

        dm = DataModel({'view': {'strength': 2.0}})
        dm.user_df = pd.DataFrame(columns=['gender', 'p1', 'p2'])
        # Features of the two classes are asymmetric:
        for u in U:
            dm.user_df.loc[u] = {'gender': 'f', 'p1': 0, 'p2': 1}
        for v in V:
            dm.user_df.loc[v] = {'gender': 'm', 'p1': 1, 'p2': 0}

        def rand_u(exclude: int = None):
            while (r := random.randint(0, len(U) - 1)) == exclude:
                pass
            return r

        def rand_v(exclude: int = None):
            while (r := random.randint(len(U), len(U) + len(V) - 1)) == exclude:
                pass
            return r

        for e in range(round(all_nodes ** 2 * D)):
            dm.interaction_graph.add_interaction(rand_u(), rand_v(), 'view')
            dm.interaction_graph.add_interaction(rand_v(), rand_u(), 'view')
            if not BI:
                dm.interaction_graph.add_interaction(rand_u(), rand_u(), 'view')
                dm.interaction_graph.add_interaction(rand_v(), rand_v(), 'view')

        assert networkx.algorithms.bipartite.is_bipartite(dm.interaction_graph.nxg)
        no_prof = LFMPredictor(
            LFMTranslator(dm, features=[], use_indicator=True),
            no_components=N,
        )
        with_prof = LFMPredictor(
            LFMTranslator(dm, features=['gender'], use_indicator=True),
            no_components=N,
        )
        complete_prof = LFMPredictor(
            LFMTranslator(dm, features=['gender', 'p1', 'p2'], use_indicator=True),
            no_components=N,
        )

        eval = LFMEvaluator({'+profile': with_prof, '-profile': no_prof, '++profile': complete_prof},
                            no_prof.translator.interaction_matrix)
        eval.run_all(epochs=range(0, 20))
        # eval.create_chart().show()
        # TODO: Add assertion that +profile and ++profile are both better than -profile (they have to, as feat are asymmetric)
