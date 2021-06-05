import logging
import re
import numpy as np
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Set, Tuple, Collection, Callable, Union

from chaos.recommend.translator import User
from chaos.shared.model import DataModel, UserType
from chaos.shared.user import user_id
from grapresso.components.edge import Edge

logger = logging.getLogger(__name__)


class CandidateRepo(ABC):
    @abstractmethod
    def retrieve_candidates(self, user: UserType) -> Collection[str]:
        pass


@dataclass
class StaticCandidateRepo(CandidateRepo):
    usernames: Set[str] = field(default_factory=set)

    def retrieve_candidates(self, _: UserType) -> Collection[str]:
        return self.usernames


class DMCandidateRepo(CandidateRepo):
    def __init__(self, dm: DataModel, include_own=False):
        self._dm = dm
        self._include_own = include_own

    def retrieve_candidates(self, user: UserType) -> Collection[str]:
        users = set(self._dm.user_ids)
        return users if self._include_own else users - {user_id(user)}

    def retrieve_candidate_users(self, user: str) -> Collection[User]:
        return [self.dm.get_user(uid) for uid in self.retrieve_candidates(user)]

    @property
    def dm(self) -> DataModel:
        return self._dm


class CandidateGenerator(CandidateRepo):
    def __init__(self, generator: CandidateRepo):
        self._generator = generator

    @abstractmethod
    def retrieve_candidates(self, user: UserType) -> Collection[str]:
        candidates = self._generator.retrieve_candidates(user)
        logger.debug(
            f"{self.__class__.__name__}: Got {len(candidates)} candidates from {self._generator.__class__.__name__}")
        return candidates

    @property
    def generator(self):
        return self._generator


class CandidateGeneratorBuilder:
    def __init__(self, base_repo):
        self._cgs = [base_repo]

    def build(self) -> CandidateRepo:
        current = self._cgs[0]
        # Build in reverse order (execution order), skip first:
        for chain in self._cgs[1:]:
            current = chain(current)
        return current

    def filter(self, cg: Callable[..., CandidateRepo], **kwargs):
        self._cgs.append(lambda r: cg(r, **kwargs))
        return self

    def cache(self, **kwargs):
        self._cgs.append(lambda r: CacheCG(r))
        return self

    def only_reciprocal(self, **kwargs):
        self._cgs.append(lambda r: ReciprocalCG(r))
        return self

    @staticmethod
    def build_reciprocal_default(dm: DataModel) -> 'CandidateRepo':
        return StrategicCG(
            ReciprocalCG(CacheCG(PreferenceCG(DMCandidateRepo(dm)))),
            on_unknown_user=StaticCandidateRepo(dm.user_ids)
        )

    @staticmethod
    def build_default(dm: DataModel) -> 'CandidateRepo':
        return StrategicCG(
            DMCandidateRepo(dm),
            on_unknown_user=StaticCandidateRepo(dm.user_ids)
        )

    def __ror__(self, other: Callable[..., CandidateRepo]):
        self._cgs.append(lambda r: other(r))
        return self

    def __or__(self, other):
        return self.__ror__(other)


class CacheCG(CandidateGenerator):
    def __init__(self, generator):
        super().__init__(generator)
        self._cache = {}
        self._stats = Counter()

    def retrieve_candidates(self, user: UserType) -> Collection[str]:
        user = user_id(user)
        if cached_result := self._cache.get(user):
            self._stats['hit'] += 1
        else:
            self._stats['miss'] += 1
            cached_result = super().retrieve_candidates(user)
            self._cache[user] = cached_result
        return cached_result

    @property
    def stats(self):
        return self._stats


class StrategicCG(CandidateGenerator):
    def __init__(self, generator: CandidateRepo,
                 on_unknown_user: CandidateRepo = None,
                 on_error: Tuple[Exception, CandidateRepo] = None):
        super().__init__(generator)
        self._on_unknown_user = on_unknown_user
        self._on_error = on_error

    def retrieve_candidates(self, for_user: UserType) -> Collection[str]:
        try:
            if self._on_unknown_user and isinstance(for_user, User) and for_user.is_unknown:
                return self._on_unknown_user.retrieve_candidates(for_user)
            else:
                return super().retrieve_candidates(for_user)
        except Exception as ex:
            if self._on_error and isinstance(ex, self._on_error[0]):
                return self._on_error[1].retrieve_candidates(for_user)
            else:
                raise ex


class DMCandidateGenerator(DMCandidateRepo):
    """ Class generates candidates based on the data model.
    TODO(kdevo): Consider CandidateGenerator as a mixin to remove duplicative functionality. Con: Diamond inheritance
    """

    def __init__(self, generator: Union['DMCandidateRepo', 'CandidateGenerator'], dm: DataModel = None):
        self._generator = generator
        if not dm:
            g = self
            while (isinstance(g, CandidateGenerator) or isinstance(g, DMCandidateGenerator)) and (g := g.generator):
                if isinstance(g, DMCandidateRepo) and g.dm:
                    dm = g.dm
                    break
            else:
                raise ValueError("Could not retrieve `dm` from underlying generator. "
                                 "Please provide one manually through constructor!")
        super().__init__(dm)

    def retrieve_candidate_users(self, user: UserType) -> Collection[User]:
        return [self.dm.get_user(uid) for uid in self.retrieve_candidates(user)]

    def retrieve_candidates(self, user: UserType) -> Collection[str]:
        return self._generator.retrieve_candidates(user)

    @property
    def dm(self):
        return self._dm

    @property
    def generator(self):
        return self._generator


class PreferenceCG(DMCandidateGenerator):
    def retrieve_candidates(self, user: UserType) -> Collection[str]:
        candidates = super().retrieve_candidates(user)

        candidates = self.dm.user_df.loc[candidates]

        prefs = user.preferences_filter if isinstance(user, User) else self.dm.get_user_preferences(user)
        if prefs is not None and prefs is not np.nan:
            candidates = candidates.query(prefs)

        return set() if candidates is None else set(candidates.index.values)


class InteractionCG(DMCandidateGenerator):
    """
    Warnings:
        only_store_strength MUST be False, otherwise can not correctly examine interactions for pattern matching!
    """

    def __init__(self, generator: Union['DMCandidateRepo', 'CandidateGenerator'],
                 interaction_pattern: str = r".*",
                 include: bool = False,
                 include_new: bool = True):
        super().__init__(generator)
        self._interaction_pattern = re.compile(interaction_pattern)
        self._include_matches = include
        self._include_new = include_new

    def retrieve_candidates(self, user: UserType) -> Collection[str]:
        candidates = super().retrieve_candidates(user)

        def match(edge: Edge):
            if edge is None:
                return self._include_new
            for it in edge.interactions:
                if self._interaction_pattern.match(str(it)):
                    return self._include_matches
            else:
                return not self._include_matches

        user_node = self.dm.get_node(user_id(user))
        return {c for c in candidates if match(user_node.edge(c))}


class ReciprocalCG(CandidateGenerator):
    """ A candidate is reciprocal if u is compatible to v and v is compatible to u.
    This class is a special generator that does not perform any action on its own, but instead filters candidates
    that fulfill the above condition of reciprocal compatibility.

    It first retrieves candidates v for user u `(u → v)`, then uses these candidates
    to perform a reverse match by delegating to all underlying generator-decorators to check compatibility to u `(v → u)`.
    """

    def retrieve_candidates(self, user: UserType) -> Collection[str]:
        u_candidates = set(super().retrieve_candidates(user))

        v_candidates = set()
        for v in u_candidates:
            if user_id(user) in super().retrieve_candidates(v):
                v_candidates.add(v)
        return u_candidates & v_candidates
