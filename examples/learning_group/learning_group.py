import logging.config
import math
from pathlib import Path

import pandas as pd
import yaml

from chaos.fetch.local import CsvSource
from chaos.process.extract.graph import GraphEdgeMapper, GraphPopularityExtractor
from chaos.process.pipeline import SequentialDataPipeline
from chaos.recommend.candidates import CandidateGeneratorBuilder
from chaos.recommend.evaluate.evaluator import LFMEvaluator
from chaos.recommend.predict.predictor import LFMPredictor
from chaos.recommend.predict.reciprocal import ReciprocalWrapper, ArithmeticStrategy
from chaos.recommend.translator import LFMTranslator
from chaos.shared.user import User

pd.set_option('display.max_columns', None)
with open('data/logging.yml') as logging_cfg_file:
    logging_cfg = yaml.safe_load(logging_cfg_file)
    logging.config.dictConfig(logging_cfg)

data = CsvSource(CsvSource.RES_ROOT / 'learning-group', Path("data/model/interactions.yml")).source_data()

pipeline = SequentialDataPipeline([
    # Smooth interactions:
    GraphEdgeMapper(strength=lambda e: math.log(1 + e.strength, 2)),
    GraphEdgeMapper(
        capacity=lambda e: e.strength,
        cost=lambda e: 1 / e.strength
    ),
    # Map other useful attributes for usage in algorithms:
    GraphPopularityExtractor('popularity', quantiles=3, metrics=('eigenvector', 'degree'),
                             labels=['low', 'medium', 'high'], add_as_node_attrib=True),
    GraphEdgeMapper(
        strength=lambda e: e.strength - (0.5 * e.strength * e.v.data['eigenvector'])
    ),
])

dm = pipeline.execute(data)
dm.describe()
print(f"DataPipeline result: \n {dm.user_df.head(10)}")

dm.interaction_graph.draw("Interaction Graph")

cg = CandidateGeneratorBuilder.build_reciprocal_default(dm)
translator = LFMTranslator(dm, [], use_indicator=True)
hp = {'learning_rate': 0.003, 'no_components': 32}
eval = LFMEvaluator(
    {
        'Hybrid, course with ID': LFMPredictor(LFMTranslator(dm, ['course'], use_indicator=True), **hp),
        'Hybrid, course + popularity with ID': LFMPredictor(
            LFMTranslator(dm, ['course', 'popularity'], use_indicator=True), **hp
        ),
        'Collaborative Filtering only': LFMPredictor(translator, **hp),
    }, translator.interaction_matrix, test_split=0.3
)
eval.run_all(epochs=range(0, 144, 1))
eval.create_report().show()

import networkx.algorithms.bipartite
bipartite = networkx.algorithms.bipartite.is_bipartite(dm.interaction_graph.nxg)
print(f"The interaction graph is {'' if bipartite else 'NOT'} bipartite.")

res = eval.best_of_all('f1')
print(f"Best predictor for F1: {res.predictor} @ epoch {res.epoch} with {res.value}")

hybrid = eval[res.predictor]

print(hybrid.predict(User.from_data({'course': 'MCD'})))
print(hybrid.predict('Ivan'))
print(hybrid.predict('Stefan'))

eval.predictors[res.predictor].similar_users(dm.get_user('Kai'))
eval.predictors[res.predictor].build_prediction_graph(k=2).draw()

rrs_hybrid = ReciprocalWrapper(eval.predictors[res.predictor], ArithmeticStrategy(ArithmeticStrategy.uninorm),
                               ku_factor=2, kv_factor=2)
rrs_hybrid.build_prediction_graph(k=2).visualize()
print(hybrid.predict('Ivan'))
print(hybrid.predict('Stefan'))
