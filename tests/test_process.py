import pandas as pd

from chaos.process.clean.df import DFCleaner
from chaos.process.clean.humanize import TextConverter, ColumnFormatType
from chaos.process.extract.name import NameToGenderExtractor
from chaos.process.extract.nlp import NLPTokenExtractor, NLPEntityExtractor
from chaos.process.extract.reduce import MostUsedExtractor
from chaos.shared.model import DataModel


class TestProcessing:
    def test_cleaning(self):
        dm = DataModel({},
                       pd.DataFrame({'id': ['test'], 'text': ['<body>Hello world! I greet <b>everyone</b>!<body>']}))
        dm = dm.apply(TextConverter('text'))
        assert '<b>' not in dm.user_df['text']

        dm = DataModel({}, pd.DataFrame(
            {'id': ['test'], 'text': ['\n####README\n **Hello world!** _I_ greet <b>everyone</b>!']}))
        dm = dm.apply(TextConverter('text', source_format=ColumnFormatType.MARKDOWN))
        print(dm.user_df)
        assert '**' not in dm.user_df['text'] and '####' not in dm.user_df['text']

    def test_nlp(self):
        dm = DataModel({}, pd.DataFrame({'text': ['Hi, this is Chaos, a research framework '
                                                  'for Reciprocal Recommender Systems!',

                                                  'Fischers Fritz fischt frische Fische,'
                                                  'frische Fische fischt Fischers Fritz.']}))
        extractor = NLPTokenExtractor('text', 'utarget', unique=True)
        dm = dm.apply(extractor)
        # extractor.visualize()
        dm = dm.apply(NLPTokenExtractor('text', 'dtarget', unique=False))
        assert len(dm.user_df.loc[0]['utarget']) == len(dm.user_df.loc[0]['dtarget'])
        assert len(dm.user_df.loc[1]['utarget']) < len(dm.user_df.loc[1]['dtarget'])

        print()
        print(dm.user_df.head())

        examples = {'text': ["I am working at Apple for more than 3 years now."
                             "Before working at Apple, I worked at Google (ML department)."]}

        dm_less = DataModel({}, pd.DataFrame(examples)).apply(
            NLPEntityExtractor('text', 'entities', unique=False)
        )
        # Apple is repeated 2 times (unique=False):
        assert len(dm_less.user_df.loc[0]['entities']) >= 3

        dm_more = DataModel({}, pd.DataFrame(examples)).apply(
            NLPEntityExtractor('text', 'entities', unique=False)
        )
        print(dm_more)

    def test_nlp_add(self):
        dm = DataModel({}, pd.DataFrame({'text': ['I study Computer Science at the FH Aachen, sometimes also remote from Bali, Indonesia.\n'
                                                  'Okay, mostly when it\'s raining in Aachen, that is pretty much always, 24/7.\n'
                                                  'I am a huge Linux fan and started developing software by the age of 12.'],
                                         'location': ['Aachen, Germany'],
                                         'workplaces': [['Aachen']]}))
        token1 = NLPTokenExtractor('text', 'text_tokens')
        token2 = NLPTokenExtractor('location', 'text_tokens')
        dm = dm.apply(token1).apply(token2)
        assert 'Germany' in dm.user_df['text_tokens'].loc[0]
        ent1 = NLPEntityExtractor('location', {'GPE': 'workplaces'}, unique=True)
        ent2 = NLPEntityExtractor('text', {'GPE': 'workplaces'}, unique=True)
        dm = dm.apply(ent1).apply(ent2)
        assert {'Aachen', 'Germany', 'Bali', 'Indonesia'} == dm.user_df['workplaces'].loc[0]

    def test_count_occ(self):
        dm = DataModel({}, pd.DataFrame({'tags': [['Python', 'RS', 'Go'], ['Go', 'Python']]}))
        dm = dm.apply(MostUsedExtractor('tags', 'tags2', top=2, key=str))
        assert dm.user_df['tags2'].loc[0] == {'Go', 'Python'}
        assert dm.user_df['tags2'].loc[1] == {'Go', 'Python'}
        dm = dm.apply(MostUsedExtractor('tags', 'tags3', top=3, key=str))
        assert dm.user_df['tags3'].loc[0] == {'Go', 'Python', 'RS'}
        assert dm.user_df['tags3'].loc[1] == {'Go', 'Python'}

    def test_norm_str(self):
        dm = DataModel({}, pd.DataFrame({'company': ['Google Inc.', 'Google...'],
                                         'location': ['Aachen,  DE', 'Aachen-FH']}))
        dm = dm.apply(DFCleaner(['company', 'location'], str_clean_regex=r'[\.,]'))
        assert '.' not in dm.user_df['company'] and ',' not in dm.user_df['company']

    def test_name_to_gender(self):
        dm = DataModel({}, pd.DataFrame({'name': ["Mila", "Max Mustermann", "Erika Musterfrau"]}))
        dm = dm.apply(NameToGenderExtractor('name', 'gender'))
        assert dm.user_df['gender'].loc[0] == 'f'
        assert dm.user_df['gender'].loc[1] == 'm'
        assert dm.user_df['gender'].loc[2] == 'f'
