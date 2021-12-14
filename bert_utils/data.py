import math
import random

import pandas as pd
from typing import List, Union, Callable

from torch.utils.data import Dataset
from sentence_transformers.readers import InputExample

from utils import get_article_path, get_article_text


class WikiQAInputExample(InputExample):
    def __init__(self, guid: str = '', texts: List[str] = None,  label: Union[int, float] = 0, article_id: int = None):
        self.article_id = article_id
        super().__init__(guid, texts, label)


class WikiQADataset(Dataset):
    def __init__(self, examples: pd.DataFrame, filemap: pd.DataFrame):
        self.examples = examples
        self.filemap = filemap
        assert isinstance(examples, pd.DataFrame)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, ix):
        assert 0 <= ix < len(self)
        query, article_title, article_id, label = self.examples.iloc[ix][['query', 'title', 'article_id', 'label']]
        article = get_article_text(get_article_path(article_title, filemap=self.filemap))
        example = WikiQAInputExample(texts=[query, article], label=label, article_id=article_id)
        return example


class CustomNoDuplicatesDataLoader:

    def __init__(self, dataset: WikiQADataset, batch_size):
        """
        A special data loader to be used with MultipleNegativesRankingLoss.
        The data loader ensures that there are no duplicate sentences within the same batch
        """
        self.batch_size = batch_size
        self.collate_fn: Callable = None
        self.dataset = dataset
        self.data_pointer = 0
        self.index_order = list(range(len(dataset)))
        # shuffle index inplace before the first iteration
        random.shuffle(self.index_order)

    def __iter__(self):
        for _ in range(self.__len__()):
            batch = []
            texts_in_batch = set()  # TODO: it's better to store ids instead of texts

            while len(batch) < self.batch_size:
                ix = self.index_order[self.data_pointer]
                example = self.dataset[ix]

                valid_example = True
                for text in example.texts:
                    if text.strip().lower() in texts_in_batch:
                        valid_example = False
                        break

                if valid_example:
                    batch.append(example)
                    for text in example.texts:
                        texts_in_batch.add(text.strip().lower())

                self.data_pointer += 1
                if self.data_pointer >= len(self.dataset):
                    self.data_pointer = 0
                    random.shuffle(self.index_order)  # reshuffle index order

            yield self.collate_fn(batch) if self.collate_fn is not None else batch

    def __len__(self):
        return math.floor(len(self.dataset) / self.batch_size)


def generate_negative_examples(true_article_id: int, k: int, article_ids: set):
    return random.sample(list(article_ids - {true_article_id}), k=k)


def add_negative_samples_to_labels_df(
    labels_df: pd.DataFrame, 
    neg_to_pos_factor: int,
    all_article_ids: set, 
    filemap: pd.DataFrame
):
    labels_df = labels_df.copy()
    labels_df['label'] = 1

    # generate negative examples

    labels_neg_df = labels_df[['query', 'article_id']].copy()

    labels_neg_df['negative_article_ids'] = labels_neg_df['article_id'].apply(
        lambda article_id: generate_negative_examples(
            true_article_id=article_id, k=neg_to_pos_factor, article_ids=all_article_ids
        )
    )

    labels_neg_df.drop(columns='article_id', inplace=True)
    labels_neg_df.rename(columns={'negative_article_ids': 'article_id'}, inplace=True)
    labels_neg_df = labels_neg_df.explode('article_id')
    labels_neg_df['article_id'] = labels_neg_df['article_id'].astype('int')  # fix type after `explode`
    labels_neg_df['label'] = 0
    # add 'title' column
    labels_neg_df = labels_neg_df.merge(
        filemap[['filename', 'article_id']], on='article_id', how='inner'
    ).rename(columns={'filename': 'title'})
    assert labels_neg_df.shape[0] == labels_df.shape[0] * neg_to_pos_factor

    # concatenate tables
    res = pd.concat([labels_df, labels_neg_df], axis=0, ignore_index=True)
    return res
