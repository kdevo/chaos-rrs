from networkx.algorithms import bipartite

from chaos.fetch.local import CsvSource
from chaos.recommend.predict.predictor import LFMPredictor
from chaos.recommend.translator import LFMTranslator
from chaos.shared.model import DataModel


class TestPredictor:
    def test_predictions(self):
        dm = CsvSource(CsvSource.RES_ROOT / 'bipartite').source_data()
        dm.interaction_graph.draw("Interaction Graph")
        predictor = LFMPredictor(
            LFMTranslator(dm, features=['gender', 'favorite-drink'], use_indicator=False),
            no_components=10
        )
        # dm.interaction_graph.draw()
        # Intentional overfit for test stability
        predictor.train(20)

        for u in dm.user_ids:
            user = dm.get_user(u)
            predictions = predictor.predict(user)
            print(f"Prediction for {user.id}", predictions)
            if user.profile_data['gender'] == 'male':
                assert dm.get_user(next(iter(predictions))).profile_data['gender'] == 'female'
            elif user.profile_data['gender'] == 'female':
                assert dm.get_user(next(iter(predictions))).profile_data['gender'] == 'male'

        # TODO(kdevo): Explore and check similar users
        # similar = predictor.similar_users(dm.user('Kai'), k=3)

        # Nested form features:
        sim1 = predictor.similar_features({'gender': ['male', 'female'], 'favorite-drink': ['coffee', 'beer']})
        # "Flat" form features
        sim2 = predictor.similar_features(
            ['gender:male', ('gender', 'female'), 'favorite-drink:coffee', ('favorite-drink', 'beer')])
        assert sim1 == sim2
        print(sim1)

        # Input graph is bipartite, so output graph projection should be, too
        net = predictor.build_prediction_graph(k=2)
        net.draw()
        assert bipartite.is_bipartite(net.nxg)

    def test_embeddings(self, default_dm: DataModel):
        predictor = LFMPredictor(
            LFMTranslator(default_dm, features=['course'])
        )
        predictor.train(10)
        # predictor.visualize()
        predictor.dump_user_embeddings()

        default_dm.user_df['some_tags'] = [['ai', 'ml']] * len(default_dm.user_df)
        default_dm.user_df.at['Kai', 'some_tags'] = ['rs', 'ai', 'ml']
        predictor = LFMPredictor(
            LFMTranslator(default_dm, features=['course', 'some_tags'])
        )
        predictor.train(10)
        # predictor.visualize()
        predictor.dump_user_embeddings()
        # TODO: Add test
