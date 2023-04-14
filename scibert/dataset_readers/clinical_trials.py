from typing import Dict, List, Sequence, Iterable
import os
import pandas as pd

from nltk.tokenize import WordPunctTokenizer

tokenizer = WordPunctTokenizer()
import itertools
import logging

from overrides import overrides

from glob import glob

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import MetadataField, TextField, SequenceLabelField, Field
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("clinical")
class ClinicalTrialsDatasetReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 label_namespace: str = "labels") -> None:
        super().__init__()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.label_namespace = label_namespace
        self.tokenizer = WordPunctTokenizer()

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        df = pd.read_csv(file_path)
        logger.info("Reading instances from lines in file at: %s", file_path)
        contexts = df.context.values.tolist()
        for context in contexts:
            # TextField requires ``Token`` objects
            tokens = [Token(token) for token in self.tokenizer.tokenize(context)]

            yield self.text_to_instance(tokens, None)

    @overrides
    def text_to_instance(self,
                         tokens: List[Token],
                         pico_tags: List[str] = None):
        sequence = TextField(tokens, self._token_indexers)
        instance_fields: Dict[str, Field] = {'tokens': sequence}
        instance_fields["metadata"] = MetadataField({"words": [x.text for x in tokens]})
        
        # Set the field 'labels' according to the specified PIO element
        if pico_tags is not None:
            instance_fields['tags'] = SequenceLabelField(pico_tags, sequence, self.label_namespace)

        return Instance(instance_fields)
