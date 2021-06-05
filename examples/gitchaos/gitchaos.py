import logging.config
from pathlib import Path

import yaml

from chaos.process.clean.df import DFCleaner
from chaos.process.clean.humanize import TextConverter, ColumnFormatType
from chaos.process.extract.graph import GraphEdgeMapper, GraphPopularityExtractor
from chaos.process.extract.name import NameToGenderExtractor
from chaos.process.extract.nlp import NLPEntityExtractor, NLPTokenExtractor
from chaos.process.extract.reduce import MostUsedExtractor
from chaos.process.pipeline import SequentialDataPipeline
from chaos.recommend.candidates import InteractionCG, DMCandidateRepo, StrategicCG
from chaos.recommend.evaluate.evaluator import LFMEvaluator, Evaluator
from chaos.recommend.predict.predictor import LFMPredictor
from chaos.recommend.predict.reciprocal import ReciprocalWrapper
from chaos.recommend.translator import LFMTranslator
from chaos.shared.model import DataModel
from examples.gitchaos.extract import GitHubPreprocessor
from examples.gitchaos.fetch import GitHubSource

with open('./res/logging.yml') as logging_cfg_file:
    logging_cfg = yaml.safe_load(logging_cfg_file)
    logging.config.dictConfig(logging_cfg)
logger = logging.getLogger(__name__)

# Put your token here:
YOUR_TOKEN = open('./res/github-token').read()
START_NODE = 'kdevo'
NODES = 5000

BREADTH = 7
VISUALIZE_WITH_AVATARS = True
FILENAME = f"gh-{START_NODE}@{NODES}"

logger.info(f"Start username: {START_NODE} / Breadth: {BREADTH} / Nodes: {NODES}")
src = GitHubSource(gql_spec=yaml.safe_load(open('./res/gql-spec.yml')),
                   token=YOUR_TOKEN,
                   start_user=START_NODE,
                   breadth=BREADTH, max_nodes=NODES)
checkpoint_path = Path(f'res/{FILENAME}')
data = None
if checkpoint_path.exists():
    data = DataModel.load(checkpoint_path)
    src.data = data
else:
    data = src.source_data()
    data.save(checkpoint_path)

pipeline = SequentialDataPipeline([
    SequentialDataPipeline(name='Metadata Preparation Pipeline', processors=[
        GitHubPreprocessor(skills_per_user=25, programming_languages_per_user=6),
        NameToGenderExtractor('name', 'assumed_gender', name_to_gender_json='../../data/extractor/name2gender.json'),
        DFCleaner(['bio'], fill_na_val=''),
        # Make it a bit easier for tokenization
        DFCleaner(['company', 'location'], str_clean_regex=r'[-.,;+]', fill_na_val=''),
        TextConverter('bio', ColumnFormatType.MARKDOWN),
        NLPEntityExtractor('bio', {'GPE': 'location_tags', 'LOC': 'location_tags', 'LANGUAGE': 'location_tags',
                                   'ORG': 'org_tags', 'PRODUCT': 'org_tags', 'NORP': 'org_tags'}),
    ]),
    SequentialDataPipeline(name='User Metadata', processors=[
        SequentialDataPipeline(name='Bio', processors=[
            NLPTokenExtractor('bio', 'bio_tags'),
            MostUsedExtractor('bio_tags', 'bio_tags', usage_threshold=2),
        ]),
        SequentialDataPipeline(name='Organizations', processors=[
            NLPTokenExtractor('company', 'org_tags'),
            MostUsedExtractor('org_tags', 'org_tags', usage_threshold=2),
        ]),
        SequentialDataPipeline(name='Location', processors=[
            NLPEntityExtractor('company', {'GPE': 'location_tags'}),
            NLPTokenExtractor('location', 'location_tags'),
            MostUsedExtractor('location_tags', 'location_tags', usage_threshold=2)
        ]),
        SequentialDataPipeline(name='Process skills', processors=[
            NLPEntityExtractor('descriptions', {'%': 'skill_tags'}),
            MostUsedExtractor('skills', 'skill_tags', usage_threshold=2),
            MostUsedExtractor('programmingLanguages', 'skill_tags', top=40, usage_threshold=2),
            MostUsedExtractor('skill_tags', 'skill_tags', top=1000, usage_threshold=2)
        ]),
    ]),
])
# Add profile URLs
data.user_df['url'] = data.user_df.index.map(lambda u: f'https://github.com/{u}')

interaction_pipeline = SequentialDataPipeline(name='Graph Manipulations', processors=[
    GraphEdgeMapper(cost=lambda e: 1 / e.strength, capacity=lambda e: e.strength),
    GraphPopularityExtractor(target_col='popularity', metrics=('eigenvector', 'degree'),
                             labels=['unknown', 'less-known', 'normal', 'well-known', 'popular', 'prominent'],
                             quantiles=[0.0, 0.1, 0.4, 0.6, 0.8, 0.99, 1.0], add_as_node_attrib=True)
])

checkpoint_path = Path(f'res/{FILENAME}-processed')
if checkpoint_path.exists():
    logger.info(f"Loading existing processed model {checkpoint_path}")
    data = DataModel.load(checkpoint_path)
else:
    data.sync_graph(True)
    data = interaction_pipeline.execute(pipeline.execute(data))
    data.save(checkpoint_path)

translator = LFMTranslator(data)
cg = StrategicCG(
    InteractionCG(DMCandidateRepo(data), interaction_pattern='follow', include=False),
    on_unknown_user=DMCandidateRepo(data)
)

# TODO(kdevo): With many nodes, this takes very long:
# gev = PredictionGraphEvaluator(
#     {
#         'hybrid': LFMPredictor(
#             LFMTranslator(
#                 data, features={'bio_tags': 0.5, 'location_tags': 0.3, 'skill_tags': 0.3, 'org_tags': 0.3}
#             ), cg, **hp
#         ),
#         'RCF': RCFPredictor(GraphTranslator(data), cg)
#     }, data.reduced_graph
# )
# gev.run_all()
# gev.create_chart().show()

hp = {'no_components': 48, 'learning_rate': 0.04}
# Found by using LFMHyperparameterOptimizer with 250 trials on github-xs, f1 metric, typically beats the above hps with less components!
hp_opt = {'no_components': 42, 'learning_rate': 0.0418, 'user_alpha': 1.7007e-05, 'item_alpha': 1.5008e-05}

evaluator = LFMEvaluator(
    interactions=translator.interaction_matrix,
    predictors={
        'Hybrid all': LFMPredictor(
            LFMTranslator(
                data, features={'bio_tags': 0.4, 'location_tags': 0.2, 'skill_tags': 0.2, 'org_tags': 0.2}
            ), cg, **hp
        ),
        'Hybrid all tuned': LFMPredictor(
            LFMTranslator(
                data, features={'bio_tags': 0.4, 'location_tags': 0.2, 'skill_tags': 0.2, 'org_tags': 0.2}
            ), cg, **hp_opt
        ),
        'Hybrid orgs + bio': LFMPredictor(
            LFMTranslator(
                data, features=['org_tags', 'bio_tags']
            ), cg, **hp
        ),
        'Collaborative Filtering only': LFMPredictor(translator, **hp_opt)
    }
)
evaluator.run_all(epochs=range(0, 72, 2), metrics=(Evaluator.PRECISION, Evaluator.RECALL, Evaluator.F1))
evaluator.create_report().show()

res = evaluator.best_of_all(Evaluator.PRECISION)
print(f"Best predictor for precision: {res.predictor} @ epoch {res.epoch} with {res.value}")
best_predictor = hybrid = evaluator[evaluator.best_of_all('precision').predictor]
hybrid.train(epochs=60 + round(NODES * 0.01))

if VISUALIZE_WITH_AVATARS:
    logger.info("Download avatar images...")
    src.dl_avatars(base_dir=Path('temp/avatars/'))
    logger.info("Create sprite with all avatar images...")
    dim = src.create_avatar_sprite(Path('temp/avatars/'), Path(f'temp/avatars/#{FILENAME}.jpeg'))
    hybrid.visualize('users', Path(f'temp/avatars/#{FILENAME}.jpeg'), sprite_single_img_dim=dim,
                     extra_cols={'url'})
else:
    hybrid.visualize('users')

reciprocal_hybrid = ReciprocalWrapper(hybrid)
