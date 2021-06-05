import re

from chaos.recommend.translator import LFMTranslator
from chaos.shared.user import User


class TestTranslator:
    def test_feat_indices(self, default_dm):
        lfmt = LFMTranslator(default_dm, ['gender', 'course'], use_indicator=False)
        idc = lfmt.feat_indices({'gender': 'male', 'course': 'MCD'})
        idc_ls = lfmt.feat_indices(['gender:male', 'course:MCD'])
        print(idc)
        assert idc[0] == 0 and idc[1] == 3
        default_dm.user_df['dynamic_skills'] = ['engineering'] * len(default_dm)
        default_dm.user_df.at['Kai', 'dynamic_skills'] = ['engineering', 'rs']
        lfmt = LFMTranslator(default_dm, {'gender': 0.2, 'dynamic_skills': 0.8}, use_indicator=False,
                             dynamic_re=re.compile('dynamic'))
        idc1 = lfmt.feat_indices({'dynamic_skills': 'engineering'})
        assert idc1[0] == 1

    def test_to_model_translation(self, default_dm):
        # Translator without identity features:
        # Translator with identity features:
        lfmt_id = LFMTranslator(default_dm, ['gender', 'course'], use_indicator=True)
        unknown_user = lfmt_id.feat_matrix(User.from_data({'gender': 'male', 'course': 'MCD'}))
        known_user = lfmt_id.feat_matrix(default_dm.get_user('Stefan'))
        # Known user has an additional identity feature, but shape is normalized
        assert unknown_user.shape == known_user.shape
        # Additional feature causes non-zero entries:
        assert (unknown_user - known_user).nnz > 0
        assert unknown_user.sum() == 1 and known_user.sum() == 1

        lfmt_id_weighted = LFMTranslator(default_dm, {'gender': 0.9, 'course': 0.3}, use_indicator=True)
        unknown_user = lfmt_id_weighted.feat_matrix(User.from_data({'gender': 'male', 'course': ['MCD', 'ISE']}))
        known_user = lfmt_id.feat_matrix(default_dm.get_user('Stefan'))
        # TODO(kdevo): Add assertions
        # assert unknown_user.sum() == 1 and known_user.sum() == 1
