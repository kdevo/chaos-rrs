import logging
import multiprocessing as mp
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Set, Union, Iterable

import numpy as np
import tensorflow as tf
from lightfm import LightFM
from lightfm.evaluation import auc_score
from scipy.sparse import coo_matrix
from sklearn.preprocessing import minmax_scale
from tensorboard.plugins import projector

from chaos.recommend.candidates import CandidateRepo, CandidateGeneratorBuilder
from chaos.recommend.predict.predictions import PredictionGraph
from chaos.recommend.translator import User, LFMTranslator, Features, Translator, GraphTranslator
from chaos.shared.graph import InteractionGraph
from chaos.shared.mixins import VisualizableMixin, PersistableMixin
from chaos.shared.model import UserType, user_id
from chaos.shared.tools import timed

logger = logging.getLogger(__name__)


class Predictor(PersistableMixin, ABC):
    def __init__(self, translator: Translator, candidate_generator: Optional[CandidateRepo] = None):
        self._translator = translator
        self._candidate_generator = candidate_generator if candidate_generator else CandidateGeneratorBuilder.build_default(
            self._translator.dm
        )

    @abstractmethod
    def predict(self, user: UserType, k: int = 5) -> Dict[str, float]:
        pass

    @property
    def translator(self) -> Translator:
        return self._translator

    def build_prediction_graph(self, users: Optional[Iterable[str]] = None, k: int = 5,
                               reciprocal: bool = False, override_cg: CandidateRepo = None) -> PredictionGraph:
        # TODO(kdevo): Use reciprocal parameter
        old_cg = None
        if override_cg:
            old_cg = self._candidate_generator
            self._candidate_generator = override_cg
        if not users:
            users = self._translator.dm.user_ids
        net = PredictionGraph()
        for u in users:
            net.add_predictions(u, self.predict(u, k))
        if override_cg:
            self._candidate_generator = old_cg
        return net


class ModelBasedPredictor(Predictor, ABC):
    @property
    @abstractmethod
    def is_trained(self) -> bool:
        pass

    @abstractmethod
    def train(self, epochs: int, resume: bool = False, **kwargs):
        pass


class MemoryBasedPredictor(Predictor, ABC):
    pass


@dataclass
class LFMPredictor(ModelBasedPredictor, VisualizableMixin):
    def __init__(self, translator: LFMTranslator, candidate_generator: CandidateRepo = None, **hyper_parameters):
        super().__init__(translator, candidate_generator)
        hyper_parameters_defaults = {
            'no_components': 18,
            'random_state': np.random.RandomState(0xc0ffee),
            'loss': 'warp'
        }
        hyper_parameters = {**hyper_parameters_defaults, **hyper_parameters}
        logger.info(f"Hyperparameters: {hyper_parameters}")

        self._lightfm: LightFM = LightFM(**hyper_parameters)
        self._threads = mp.cpu_count()
        logger.debug(f"Using {self._threads} threads (= cpu cores of your machine) for training the model.")

        self._trained = False
        self._translator = translator
        self._interaction_matrix, self._user_matrix = translator.predictor_dm
        self._predict_net = PredictionGraph()

    @timed(__name__)
    def train(self, epochs: int = 3, resume=False,
              interactions: Union[coo_matrix, InteractionGraph] = None,
              check_sanity=True):
        """
        Train the model, by default, use provided interactions from the translator.
        Override interactions to train on by providing train_it.

        Args:
            epochs: Choosing a too high value here will likely result in overfitting
            interactions:
            resume: Resume training if possible and do not start all over again.
                    For example, if True and train is called n=3 times with epochs=3, n*epochs=9 epochs will be trained.
            check_sanity: Calculates AUC score based on train_interactions after fitting

        Returns:

        """
        logger.info(f"Begin training LightFM model with {epochs} epochs...")

        # Destroy predict net cache (not valid anymore)
        self._predict_net = PredictionGraph()
        train_interactions = self._translator.train_interactions

        # pg = click.progressbar(range(epochs))
        # LFMProgressCapture(on_epoch=pg.update)
        if not resume or not self._trained:
            self._lightfm.fit(interactions=train_interactions, sample_weight=train_interactions,
                              item_features=self._user_matrix,
                              user_features=self._user_matrix,
                              num_threads=self._threads,
                              epochs=epochs, verbose=False)
            self._trained = True
        else:
            self._lightfm.fit_partial(interactions=train_interactions, sample_weight=train_interactions,
                                      item_features=self._user_matrix,
                                      user_features=self._user_matrix,
                                      num_threads=self._threads,
                                      epochs=epochs, verbose=False)

        # Sanity check using AUC score
        if check_sanity:
            auc = auc_score(self._lightfm, train_interactions,
                            user_features=self._user_matrix,
                            item_features=self._user_matrix,
                            num_threads=self._threads)
            auc = auc.mean()
            if round(auc, 1) <= 0.5:
                logger.warning(f"The AUC score of {auc} is "
                               f"{'worse than random' if auc < 0.5 else 'approximately equal to random'}. "
                               "Please inspect your model and/or hyperparameters carefully.")
            else:
                logger.info(f"AUC score: {auc}")
        return self

    @timed(__name__, logging.DEBUG)
    def predict(self, user: UserType, k: int = 5) -> Dict[str, float]:
        """
        Predict a recommendation for a single user.

        Args:
            user: The user who is predicted for
            k: Limit predictions on the top k users
        Returns:
            Dictionary with username as a key and score from 0 to 1 for score (sorted descending)

        """

        if uid := user_id(user):
            num_uid = self._translator.to_predictor_user(uid)
            user_feat = self._user_matrix
            logger.debug(f"Prediction for user '{uid}' with internal id {num_uid}")
        else:
            # In the following, we tell LFM to use the first row (user_id = 0) in user_feat.
            # This matrix contains an approximation of the given user and is built on-demand.
            num_uid = 0
            user_feat = self._translator.feat_matrix(user)
            logger.debug(f"Prediction for unknown user: {user}")

        other_users = self._candidate_generator.retrieve_candidates(user)
        other_num_uids = self._translator.predictor_users(other_users)

        scores = self._lightfm.predict(num_uid, other_num_uids,
                                       user_features=user_feat, item_features=self._user_matrix,
                                       num_threads=self._threads)

        scores = minmax_scale(scores)

        sorted_scores = sorted(zip(scores, other_users), reverse=True)[:k]
        user_to_score = {v: s for s, v in sorted_scores}
        return user_to_score

    @property
    def model(self):
        return self._lightfm

    @property
    def translator(self) -> LFMTranslator:
        return self._translator

    def similar_features(self, features: Features, k=5, include_indicator_feat=False, result_as_df=False):
        # Define similarity as the cosine of the angle between the tag latent vectors
        # Partly adapted from https://making.lyst.com/lightfm/docs/examples/hybrid_crossvalidated.html

        tag_indices = self._translator.feat_indices(features)
        idx_df = self._translator.feat_mapping.reset_index().set_index('index')[
            ['feature', 'label', 'is_indicator', 'count']
        ]

        def sim(embeddings):
            # Normalize the vectors to unit length (unit vec)
            tag_embeddings = (embeddings.T / np.linalg.norm(embeddings, axis=1)).T
            feat_to_similarity = {}
            for idx in tag_indices:
                query_embedding = tag_embeddings[idx]
                similarity = np.dot(tag_embeddings, query_embedding)
                # Sort features, exclude first feature which is the feature itself
                sorted_indices = [idx for idx in np.argsort(-similarity)[1:]]
                if not include_indicator_feat:
                    sorted_indices = list(filter(lambda idx: not idx_df.loc[idx]['is_indicator'], sorted_indices))
                sorted_indices = sorted_indices[:k]
                if result_as_df:
                    key = self._translator.build_feat_label(idx_df['feature'].loc[idx], idx_df['label'].loc[idx])
                    feat_to_similarity[key] = idx_df.loc[sorted_indices][['feature', 'label', 'count']]
                else:
                    feat_to_similarity.setdefault(idx_df['feature'].loc[idx], {})[idx_df['label'].loc[idx]] = list(
                        idx_df.loc[sorted_indices][['feature', 'label']].itertuples(index=False, name=None)
                    )

            return feat_to_similarity

        # TODO(kdevo): Verify if the following holds true:
        # Similarly interest user tags:
        u_tags = sim(self.model.user_embeddings)
        # Similarly attract user tags:
        # v_tags = sim(self.model.item_embeddings)

        return u_tags

    @timed(__name__)
    def similar_users(self, user: UserType, k=5):
        """
        EXPERIMENTAL: Scale-invariant cosine-similarity approach to find/recommend similar users.

        As cosine similarity neither depends on biases nor other popularity-influenced factors, this method is a good
        approach to recommend completely new users to less popular users, whereas the usual recommend() is not appropriate.

        Args:
            user: User (can be unknown during training) to find similar other users
            k: Top k users

        Returns:
            Top k users that are similar
        """
        feat_matrix = self._translator.feat_matrix(user)
        idx_df = self._translator.feat_mapping.reset_index().set_index('index')[['feature', 'label']]

        def sim(reps, rep, k):
            # _bias, rep = self.model.get_item_representations(feat_matrix)
            # Partly adapted from https://github.com/lyst/lightfm/issues/359#issuecomment-412245108
            norm_reps = np.linalg.norm(reps, axis=1)
            norm_rep = np.linalg.norm(rep)
            # Cosine similarity ∈ [-1, 1]
            scores = (reps @ rep) / (norm_reps * norm_rep)

            # Slightly slower:
            # scores = reps.dot(rep[0]) / norm_reps / norm_rep
            # Slowest:
            # from sklearn.metrics.pairwise import cosine_similarity
            # scores = cosine_similarity(reps, rep)
            scores = minmax_scale(scores)

            # TODO(kdevo): The following will return faulty values if no indicator features are used:
            return {idx_df.loc[idx]['label']: scores[idx] for idx in np.argsort(-scores)[0:k]}

        # Ignore bias / axis section as it has no influence on vector direction that is important for cosine similarity
        # Similar users
        u_scores = sim(self.model.get_user_representations(self._translator.user_matrix)[1],
                       self.model.get_user_representations(feat_matrix)[1][0], k)
        # Similarly liked users
        # v_scores = sim(self.model.get_item_representations(self._translator.user_matrix)[1],
        #                self.model.get_item_representations(feat_matrix)[1][0], k)

        return u_scores

    def dump_user_embeddings(self, label_path: Optional[Path] = Path('./temp/labels.tsv'),
                             embedding_path: Optional[Path] = Path('./temp/embeddings.tsv'),
                             extra_cols: Set[str] = frozenset()):
        if label_path:
            with open(label_path, 'w') as f:
                feats = [*self._translator.features.keys(), *extra_cols] if self._translator.is_using_features else []
                f.write("user\t{}\n".format('\t'.join(feats)))
                for user in self._translator.user_id_mapping.keys():
                    # Write row with user profile_data labels:
                    user_features = self._translator.dm.user_df[feats].loc[user]
                    f.write("{}\t{}\n".format(user, '\t'.join(str(uf) for uf in user_features.values)))

        user_repr: np.array = self.model.get_user_representations(self._translator.user_matrix)[1]
        if embedding_path:
            with open(embedding_path, 'w') as f:
                for user in user_repr:
                    # Write every embedding value of the user:
                    f.write('{}\n'.format('\t'.join(str(v) for v in user)))
        return user_repr

    def dump_feat_embeddings(self, label_path: Optional[Path] = Path('./temp/labels.tsv'),
                             embedding_path: Optional[Path] = Path('./temp/embeddings.tsv'),
                             include_indicator_feat=False):
        v_embeddings = False

        feat_mapping = self._translator.feat_mapping
        if not include_indicator_feat:
            feat_mapping = feat_mapping.loc[~feat_mapping['is_indicator']]
        if label_path:
            with open(label_path, 'w') as f:
                f.write("id\tfeature\tlabel\n")
                for feat, label in feat_mapping.index.values:
                    f.write("{}\t{}\t{}\n".format(self._translator.build_feat_label(feat, label), feat, label))

        model_embeddings = self._lightfm.item_embeddings if v_embeddings else self._lightfm.user_embeddings
        embeddings: np.array = model_embeddings[
                               0 if include_indicator_feat else len(self._translator.user_id_mapping):
                               len(self._translator.feat_mapping)
                               ]
        # TODO(kdevo): Check if normalization is not necessary:
        # embeddings = (embeddings.T / np.linalg.norm(embeddings, axis=1)).T
        if embedding_path:
            with open(embedding_path, 'w') as f:
                for feat in embeddings:
                    # Write every embedding value of the feature:
                    f.write('{}\n'.format('\t'.join(str(v) for v in feat)))
        return embeddings

    def visualize(self, style='user', sprite_path: Path = None,
                  sprite_single_img_dim: tuple = (96, 96),
                  logdir: Path = Path('./temp'),
                  extra_cols: Set[str] = frozenset()):
        # TODO(kdevo): Add feat only option
        # TODO(kdevo): Extract assertions to tests
        labels_fn = f'{style}-labels.tsv'
        logdir.mkdir(exist_ok=True)
        if style == 'users':
            embeddings = self.dump_user_embeddings(logdir / labels_fn, None, extra_cols)
            assert embeddings.shape[0] == len(self._translator.user_id_mapping)
            non_indicator_feat = len(self._translator.feat_mapping.loc[~self._translator.feat_mapping['is_indicator']])
            assert embeddings.shape == (self.model.user_embeddings.shape[0] - non_indicator_feat,
                                        self.model.user_embeddings.shape[1])
        elif style == 'features':
            embeddings = self.dump_feat_embeddings(logdir / labels_fn, None)
        else:
            raise

        # A lot of trial-and-error was needed to find out that `name` needs to be consistent
        # The official TF Projector documentation regarding this is a mess,
        #  see also https://github.com/tensorflow/tensorboard/issues/2471 for other workarounds/info
        name = f"{style}-embeddings"
        features = tf.Variable(embeddings, name=name)
        # Unpacking ensures that checkpoint is named accordingly:
        checkpoint = tf.train.Checkpoint(**{name: features})
        checkpoint.save(logdir / f'{name}.ckpt')

        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = f"{name}/.ATTRIBUTES/VARIABLE_VALUE"
        embedding.metadata_path = labels_fn
        if sprite_path:
            embedding.sprite.image_path = str(sprite_path.absolute())
            embedding.sprite.single_image_dim.extend(list(sprite_single_img_dim))
        projector.visualize_embeddings(logdir, config)
        logger.info(
            f"Open TensorBoard with the following command: `tensorboard --logdir {logdir.absolute()}`, then go to 'Projector'"
        )
        # TODO(kdevo): In a later release, TF >=2.4 can be opened automatically.
        #   Currently, this is not possible due to the following issue: https://github.com/tensorflow/tensorboard/issues/3683
        # tb = program.TensorBoard()
        # tb.configure(logdir='./temp/', bind_all=True)
        # url = tb.launch()
        # while 1:
        #     sleep(1)

    @property
    def is_trained(self):
        return self._trained


class StubPredictor(Predictor):
    """
    Only for testing purposes for now, e.g. to compare more "intelligent" RS with random guessing.
    """

    def __init__(self, translator: GraphTranslator, candidate_generator: CandidateRepo, mode='random'):
        super().__init__(translator, candidate_generator)
        self._mode = mode

    @property
    def is_trained(self) -> bool:
        return True

    def train(self, epochs: int, resume: bool = False, **kwargs):
        pass  # NOOP, not ML-based

    def predict(self, user: User, k: int = 5) -> Dict[str, float]:
        candidates = list(self._candidate_generator.retrieve_candidates(user))
        if self._mode == 'random':
            scores = {v: sum([e.strength for e in self.translator.dm.interaction_graph.edges(None, v)])
                      for v in random.choices(candidates, k=k)}
            return {v: s for v, s in sorted(scores.items(), key=lambda vs: vs[1], reverse=True)[:k]}
        elif self._mode == 'loop':
            # TODO(kdevo): Only random mode is implemented for now, also support own "finding loop" alg similar to
            #     trust networks, but by finding circle:
            #   Circle over n>2 edges: Implicit reciprocal
            #   Circle over n=2 edges: Direct reciprocal
            #   Prefer edges with max strength
            raise NotImplementedError("loop mode not implemented")
        else:
            raise NotImplementedError(f"{self._mode} not implemented")
