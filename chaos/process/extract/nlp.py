import itertools
from abc import ABC, abstractmethod
from itertools import chain
from typing import Optional, Union, Dict, Iterable

import spacy
from spacy import displacy

from chaos.process.processor import DFExtractor
from chaos.shared.mixins import VisualizableMixin
from chaos.shared.model import DataModel


class NLPExtractor(DFExtractor, VisualizableMixin, ABC):
    @abstractmethod
    def __init__(self, source_col: str,
                 target_col: Union[str, Dict[str, str]], lang='en_core_web_md',
                 merge: bool = True, unique: bool = True):
        super().__init__()
        self._source_col = source_col
        self._nlp = spacy.load(lang)
        self._target_col = target_col
        self._merge_existing = merge
        self._unique = unique
        self._doc = None

    def visualize(self, style: Optional[str] = None, **kwargs):
        if not self._doc:
            raise ValueError("Unable to serve entities diagram, "
                             "please start the extractor first by calling `execute`!")
        displacy.serve(self._doc, style=style, options=kwargs)

    def execute(self, data: 'DataModel') -> 'DataModel':
        return data


class NLPTokenExtractor(NLPExtractor):
    def __init__(self, source_col: str, target_col: str,
                 ignore_stopwords=True, ignore_nonalpha=True, use_baseform=True,
                 unique=True, merge=True):
        super().__init__(source_col, target_col, merge=merge)
        self._ignore_stopwords = ignore_stopwords
        self._ignore_nonalpha = ignore_nonalpha
        self._use_baseform = use_baseform
        self._unique = unique

    def execute(self, data: DataModel) -> DataModel:
        def extract(text_src):
            if type(text_src) == str:
                tokens = self.generator(text_src)
            elif isinstance(text_src, Iterable):
                tokens = chain(*[self.generator(t) for t in text_src])
            else:
                raise ValueError("Values typed other than str or Iterable are not supported for `source_col`")
            return self._collection(tokens)

        if self._target_col in data.user_df and self._target_col != self._source_col:
            # If target column exists, try to append
            if self._merge_existing:
                data.user_df[self._target_col] = data.user_df[[self._source_col, self._target_col]].apply(
                    lambda s: self._merge_iterables(extract(s[self._source_col]), s[self._target_col]), axis=1
                )
            else:
                raise ValueError(f"Can not append to existing {self._target_col}, make sure to set merge = True")
        else:
            data.user_df[self._target_col] = data.user_df[self._source_col].apply(
                extract
            )
        return data

    def visualize(self, style='dep', **kwargs):
        super().visualize(style)

    def generator(self, text):
        self._doc = self._nlp(text)
        for token in self._doc:
            if token.is_stop and self._ignore_stopwords:
                continue
            if not token.is_alpha and self._ignore_nonalpha:
                continue
            if self._use_baseform:
                yield token.text
            else:
                yield token.lemma_


class NLPEntityExtractor(NLPExtractor):
    def __init__(self, source_col: str, target_col: Union[str, Dict[str, str]],
                 ent_types=frozenset({'PERSON', 'NORP', 'FAC', 'ORG', 'GPE',
                                      'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART',
                                      'LAW', 'LANGUAGE'}),
                 unique=True, merge=True):
        """
        Per default, ent_types is set to regex to exclude the following entity types:
            TIME, DATE, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL
        See the default param of `ent_types` for the opposing set.
        This typically works good enough for RS, since numeric information is often too specific.

        Use-Cases:
        - Find company name(s)
        - Find out location information or languages
        - Detect political stuff, etc.

        See Also
            https://spacy.io/api/annotation#named-entities
        """

        super().__init__(source_col, target_col, merge=merge, unique=unique)
        self._ent_types = ent_types

    def execute(self, data: DataModel) -> DataModel:
        def extract(s):
            entities = self.generator(s[self._source_col])
            if type(self._target_col) == str:
                s[self._target_col] = self._collection(entities)
            elif type(self._target_col) == dict:
                ent_type_to_text = {
                    k: self._collection(map(lambda v: v[1], vs))
                    for k, vs in itertools.groupby(sorted(entities, key=lambda e: e[0]), key=lambda t: t[0])
                }
                for ent_type, text in ent_type_to_text.items():
                    target_col = self._target_col.get(ent_type)
                    # Column for remaining elements:
                    if not target_col:
                        target_col = self._target_col.get('%')
                    if target_col:
                        s[target_col] = self._merge_iterables(s.get(target_col, []), text)
            else:
                raise ValueError("Parameter `target_col` needs to be either str or dict type!")
            return s

        data.user_df = data.user_df.apply(extract, axis=1)
        return data

    def visualize(self, style='ent', **kwargs):
        super().visualize(style, )

    def generator(self, text) -> Dict:
        self._doc = self._nlp(text)
        for ent in self._doc.ents:
            if ent.label_ in self._ent_types:
                yield ent.label_, ent.text
