from chaos.recommend.candidates import DMCandidateRepo
from chaos.recommend.predict.predictor import StubPredictor, LFMPredictor
from chaos.recommend.predict.reciprocal import RCFPredictor, ArithmeticStrategy, ReciprocalWrapper
from chaos.recommend.translator import GraphTranslator, LFMTranslator
from chaos.shared.model import DataModel


class TestReciprocal:
    def test_rcf(self, default_dm: DataModel):
        cg = DMCandidateRepo(default_dm)
        translator = GraphTranslator(default_dm)
        rcf = RCFPredictor(translator, cg, ArithmeticStrategy(ArithmeticStrategy.harmonic_mean))
        recs_u = rcf.predict('Kai', 10)
        recs_v = rcf.predict('Yannick', 10)
        assert list(recs_u.keys())[0] == 'Yannick'
        assert recs_v['Kai'] == recs_u['Yannick']
        print(recs_u)
        print(recs_v)

        translator.dm.interaction_graph.remove_edge('Yannick', 'Kai')
        translator.dm.interaction_graph.remove_edge('Kai', 'Yannick')
        recs = rcf.predict('Kai', 10)
        # Yannick is not the first in the recommendation list anymore because we removed the reciprocal link before:
        assert list(recs.keys())[0] != 'Yannick'
        print(recs)

    def test_wrapper(self, default_dm: DataModel):
        user = default_dm.get_user('Monica')
        cg = DMCandidateRepo(default_dm)

        wrapped_rs = ReciprocalWrapper(StubPredictor(GraphTranslator(default_dm), cg),
                                       ArithmeticStrategy(ArithmeticStrategy.harmonic_mean))
        wrapped_rs2 = ReciprocalWrapper(LFMPredictor(LFMTranslator(default_dm), cg),
                                        ArithmeticStrategy(ArithmeticStrategy.harmonic_mean))

        wrapped_rs2.train(10)
        wrapped_rs2.build_prediction_graph(k=2).only_reciprocal().draw()

        true_rrs = RCFPredictor(GraphTranslator(default_dm), cg, ArithmeticStrategy(ArithmeticStrategy.harmonic_mean))
        wrapped_true_rrs = ReciprocalWrapper(
            RCFPredictor(GraphTranslator(default_dm), cg, ArithmeticStrategy(ArithmeticStrategy.harmonic_mean)),
            ArithmeticStrategy(ArithmeticStrategy.harmonic_mean)
        )

        # For already reciprocal RS, it does not make any difference (besides bad performance)
        #   if they are wrapped in ReciprocalWrapper or not
        assert true_rrs.predict(user, 5) == wrapped_true_rrs.predict(user, 5)
        assert wrapped_true_rrs.stats['rank_violations']['Monica'] == 0
        print(wrapped_true_rrs.stats)

        assert wrapped_rs.predict(user, 10) != true_rrs.predict(user, 10)
