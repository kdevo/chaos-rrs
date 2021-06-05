import logging.config
import math

import pytest
import yaml

from chaos.fetch.local import CsvSource
from chaos.process.extract.graph import GraphEdgeMapper, GraphPopularityExtractor
from chaos.process.pipeline import SequentialDataPipeline

with open('tests/logging.yml') as logging_cfg_file:
    logging_cfg = yaml.safe_load(logging_cfg_file)
    logging.config.dictConfig(logging_cfg)


# @pytest.fixture()
# def default_dm():
#     def gen_dm(name='learning-group'):
#         return CsvSource(CsvSource.RES_ROOT / name).source_data()
#
#     return gen_dm


@pytest.fixture()
def default_dm():
    pipeline = SequentialDataPipeline([
        # Smooth interactions:
        GraphEdgeMapper(strength=lambda e: math.log(1 + e.strength, 2)),
        # Inverse propensity weight:
        # GraphEdgeMapper(strength=lambda e: e.strength / min(len(e.v.edges), 1)),
        # Map other useful attributes for usage in algorithms:
        GraphEdgeMapper(
            capacity=lambda e: e.strength,
            cost=lambda e: 1 / e.strength
        ),
        GraphPopularityExtractor('popularity', quantiles=3,
                                 labels=['low', 'medium', 'high']),
    ])
    data = pipeline.execute(CsvSource(CsvSource.RES_ROOT / 'learning-group').source_data())
    return data
