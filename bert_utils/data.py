import math
import random

from typing import List, Union, Callable
import pandas as pd

from torch.utils.data import Dataset
from sentence_transformers.readers import InputExample

def foo():
    print('foo')

def get_article_path(title: str, filemap: pd.DataFrame):
    fp = filemap.query('filename == @title.lower().strip()')['path'].iloc[0]
    return fp


def get_article_text(fp):
    with open(fp) as fin:
        text = fin.read()
    return text


def two_sets_stats(arr1, arr2):
    print(f'init shapes: {len(arr1), len(arr2)}')
    s1 = set(arr1)
    s2 = set(arr2)
    print(f'unique elements: {len(s1), len(s2)}')
    print(f's1 & s2: {len(s1 & s2)}')
    print(f's1 ^ s2: {len(s1 ^ s2)}')
    print(f's1 - s2: {len(s1 - s2)}')
    print(f's2 - s1: {len(s2 - s1)}')


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
        query, article_title, article_id = self.examples.iloc[ix][[
            'query', 'title', 'article_id']]
        article = get_article_text(get_article_path(article_title, filemap=self.filemap))
        example = WikiQAInputExample(
            texts=[query, article], label=1, article_id=article_id)
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


def add_negative_samples_to_val_set(
    labels_val: pd.DataFrame, val_neg_to_pos_factor: int,
    all_article_ids: set, filemap: pd.DataFrame
):
    labels_val = labels_val.copy()
    labels_val['label'] = 1

    # generate negative examples

    labels_val_neg = labels_val[['query', 'article_id']].copy()

    labels_val_neg['negative_article_ids'] = labels_val_neg['article_id'].apply(
        lambda article_id: generate_negative_examples(
            true_article_id=article_id, k=val_neg_to_pos_factor, article_ids=all_article_ids
        )
    )

    labels_val_neg.drop(columns='article_id', inplace=True)
    labels_val_neg.rename(
        columns={'negative_article_ids': 'article_id'}, inplace=True)
    labels_val_neg = labels_val_neg.explode('article_id')
    labels_val_neg['article_id'] = labels_val_neg['article_id'].astype(
        'int')  # fix type after `explode`
    labels_val_neg['label'] = 0
    # add 'title' column
    labels_val_neg = labels_val_neg.merge(
        filemap[['filename', 'article_id']], on='article_id', how='inner'
    ).rename(columns={'filename': 'title'})
    assert labels_val_neg.shape[0] == labels_val.shape[0] * \
        val_neg_to_pos_factor

    # concatenate tables
    res = pd.concat([labels_val, labels_val_neg], axis=0, ignore_index=True)
    return res
