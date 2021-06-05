from random import random

from chaos.recommend.candidates import PreferenceCG, ReciprocalCG, InteractionCG, DMCandidateRepo, \
    CandidateGeneratorBuilder
from chaos.shared.model import DataModel


class TestCandidateGenerator:
    def test_filtering(self, default_dm: DataModel):
        # Ensure that column exists:
        default_dm.user_df[default_dm.preference_filter_col] = ''
        default_dm.user_df[default_dm.preference_filter_col] = default_dm.user_df[
            default_dm.preference_filter_col].apply(
            lambda _: 'course == "MCD"' if random() > 0.5 else 'course == "ISE"'
        )
        # By adding two users with compatible preferences, there will be at least one pair that is reciprocally compatible:
        default_dm.user_df.loc['Yannick', default_dm.preference_filter_col] = 'course == "ISE"'
        default_dm.user_df.loc['Kai', default_dm.preference_filter_col] = 'course == "ISE"'
        # The following is a quite unusual filtering that only generates candidates that are reciprocally compatible and
        # already interacted before with each other:
        candidates = ReciprocalCG(InteractionCG(
            PreferenceCG(DMCandidateRepo(default_dm)), include=True, include_new=False
        )).retrieve_candidates(default_dm.get_user('Kai'))
        assert len(list(candidates)) > 0
        assert any(map(lambda c: c == 'Yannick', candidates))
        print(list(candidates))
        cg1 = InteractionCG(
            ReciprocalCG(PreferenceCG(DMCandidateRepo(default_dm))),
            include=True, include_new=False
        )
        cg1_candidates = cg1.retrieve_candidates(default_dm.get_user('Kai'))
        print(list(cg1_candidates))
        assert len(list(candidates)) <= len(list(cg1_candidates))

        # Another possibility to construct a candidate generator is by using a Builder pattern:
        cg2 = (CandidateGeneratorBuilder(DMCandidateRepo(default_dm))
               .filter(PreferenceCG).cache().only_reciprocal()
               .filter(InteractionCG, include=True, include_new=False)
               .build())
        cg2_candidates = cg2.retrieve_candidates('Kai')

        # The candidate generator is the same as above WITH cache and should deliver same result:
        assert set(cg1_candidates) == set(cg2_candidates)
        cache_cg = cg2.generator.generator
        miss = cache_cg.stats['miss']
        # Cache is completely unpopulated and every request to the chain is "new":
        assert cache_cg.stats['hit'] == 0
        # Let's do another round
        kcgb_candidates = cg2.retrieve_candidates('Kai')
        # Hits should be previous misses now:
        assert cache_cg.stats['hit'] == miss
        # Again, candidates should be the same:
        assert candidates == set(kcgb_candidates)
        # This will be faster than before, because it uses the cache
        hcgb_candidates = cg2.retrieve_candidates(default_dm.get_user('Horst'))
        print(cache_cg.stats)
        # TODO(kdevo): Add test
