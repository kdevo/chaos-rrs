from collections import Counter
from typing import Dict, Optional

import numpy as np
import pandas as pd

from chaos.process.processor import Processor
from chaos.shared.model import DataModel


class GitHubPreprocessor(Processor):
    """Mainly preprocesses repos to extract user programming languages and skills (from repo topics)
    """

    def __init__(self, skills_per_user, programming_languages_per_user,
                 drop_cols=('repositories', 'email', 'aboutRepo',
                            'isBountyHunter', 'isCampusExpert', 'isDeveloperProgramMember', 'isSiteAdmin')):
        super().__init__()
        self._top_tags_per_repo = 5
        self._top_tags = skills_per_user
        self._top_lang = programming_languages_per_user
        self._drop_cols = drop_cols

    @staticmethod
    def safe_get(d, *ks, default=None):
        try:
            for k in ks:
                d = d[k]
            return d
        except (KeyError, TypeError):
            return default

    def execute(self, data: 'DataModel') -> DataModel:
        udf = data.user_df

        def get_repo_text(d: Optional[Dict]):
            try:
                return d['object']['text']
            except (KeyError, TypeError):
                return ''

        def fill_empty(df):
            return df.replace(r'^\s*$', np.nan, regex=True)

        # Complement user bio:
        udf['aboutRepo'] = fill_empty(udf['aboutRepo'])
        udf['bio'] = fill_empty(udf['bio'] + '\n' + udf['aboutRepo'].apply(get_repo_text))

        # TODO(kdevo): Maybe filter by https://en.wikipedia.org/wiki/List_of_legal_entity_types_by_country
        udf['company'] = udf['company'].str.replace('@', '')

        udf['email'] = fill_empty(udf['email'])

        udf = self.process_repos(udf, self._top_tags_per_repo, self._top_tags, self._top_lang)

        udf['publicityScore'] = 0
        for col in ('bio', 'email', 'name', 'company', 'location', 'twitterUsername', 'websiteUrl'):
            udf['publicityScore'] += udf[col].notna().astype(int) * (1 / udf[col].count())
        for col in ('isHireable', 'isBountyHunter', 'isCampusExpert', 'isDeveloperProgramMember', 'isSiteAdmin'):
            udf['publicityScore'] += udf[col].astype(int) * (1 / udf[col].count())
        # The more repositories the author showcases, the higher the publicity:
        udf['publicityScore'] += udf['publicityScore'] * 0.1 * (udf['pinnedItemsRemaining'] / 6) + \
                                 udf['publicityScore'] * 0.1 * (len(udf['repositories']) / 10)

        # We explored and processed the repos, they can now be dropped:
        udf.drop(list(self._drop_cols), axis=1, inplace=True)

        data.user_df = udf
        return data

    def process_repos(self, udf: pd.DataFrame, tags_per_repo=5, top_tags=5, top_lang=5):
        udf['repositories'] = udf['repositories'].apply(lambda r: r['nodes'])

        def explore_repos(s: pd.Series, with_social_utility=True):
            total_stars = 0
            pl_cnt = Counter()
            skills_cnt = Counter()
            s['descriptions'] = ''
            for r in s['repositories']:
                if r['isEmpty']:
                    continue
                total_stars += r['stargazerCount']
                utility = r['stargazerCount'] if with_social_utility else 0
                if langs := self.safe_get(r, 'languages', 'nodes'):
                    pl_cnt += Counter({l['name']: (utility + 1) * (len(langs) / (p + 1)) for p, l in enumerate(langs)})
                if skills := self.safe_get(r, 'repositoryTopics', 'nodes'):
                    sorted_skills = sorted([t['topic'] for t in skills], reverse=True,
                                           key=lambda t: t['stargazerCount'])[:tags_per_repo]
                    skills_cnt += Counter(
                        {t['name']: (utility + 1) * (len(skills) / (p + 1)) for p, t in enumerate(sorted_skills)}
                    )
                s['descriptions'] += (r['description'] if r['description'] else '') + '\n'
            s['descriptions'] = s['descriptions'].strip()
            s['skills'] = list(sorted(skills_cnt, key=skills_cnt.get, reverse=True))[:top_tags]
            s['programmingLanguages'] = list(sorted(pl_cnt, key=pl_cnt.get, reverse=True))[:top_lang]
            s['stars'] = total_stars
            return s

        udf = udf.apply(explore_repos, axis=1)
        return udf
