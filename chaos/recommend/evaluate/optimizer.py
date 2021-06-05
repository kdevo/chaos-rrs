from typing import Callable, Dict, Iterable

import pandas as pd
import optuna
import logging

from optuna import Trial
from optuna.trial import BaseTrial

from chaos.recommend.evaluate.evaluator import Evaluator, Metric, LFMEvaluator, disablelog
from chaos.recommend.predict.predictor import LFMPredictor

logger = logging.getLogger(__name__)


class LFMHyperparameterOptimizer(Evaluator):
    def __init__(self, predictor_factory: Callable[[Dict], LFMPredictor]):
        optuna.logging.set_verbosity(optuna.logging.WARN)
        self._predictor_factory = predictor_factory
        self._hp_study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler()
        )

    # TODO: Refactor
    def run_all(self, metrics: Iterable[Metric], trials=2000, epochs=range(1, 101),
                best_of_metric: Metric = None) -> pd.DataFrame:
        if not best_of_metric:
            best_of_metric = next(metrics)
        logger.info(f"Started optimization process for metric '{best_of_metric}'. This might take a while.")
        self._hp_study.optimize(self._objective(metrics, best_of_metric, epochs),
                                n_jobs=1, show_progress_bar=True,
                                n_trials=trials)
        logger.info(
            f"Found best value for {best_of_metric} = {self._hp_study.best_value}: \n"
            f"{self._hp_study.best_params}"
        )
        return self._hp_study.trials_dataframe()

    def _objective(self, all_metrics: Iterable[Metric], best_of_metric: Metric, epochs: range):
        # Partly adapted from https://www.eigentheories.com/blog/lightfm-vs-hybridsvd/
        @disablelog()
        def obj(trial: Trial):
            # TODO(kdevo): Extract to parameter for easier customizability:
            hp = {
                'no_components': trial.suggest_int('no_components', 1, 100),
                'item_alpha': trial.suggest_loguniform('item_alpha', 1e-10, 1e-0),
                'user_alpha': trial.suggest_loguniform('user_alpha', 1e-10, 1e-0),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.25)
            }
            predictor = self._predictor_factory(hp)
            eval = LFMEvaluator({'model': predictor},
                                predictor.translator.interaction_matrix,
                                show_progress=False)
            eval.run_all(epochs=epochs, metrics=all_metrics)
            best = eval.best_of_all(best_of_metric.name)
            trial.set_user_attr('full_result', eval.results['model'])
            return best.value

        return obj

    def best_of_all(self) -> BaseTrial:
        # TODO(kdevo): Adjust for param metric
        return self._hp_study.best_trial

