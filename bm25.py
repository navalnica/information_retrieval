import re
import math
import pandas as pd
from collections import defaultdict
from tqdm.auto import tqdm

from utils import get_article_path, get_article_text


def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s\-]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    tokens = text.split(' ')
    return tokens


def accuracy(*, titles_true, title_predictions, k=1):
    """ Assert title_predictions are sorted descending. """
    assert len(titles_true) == len(title_predictions)
    acc = 0
    for t_true, t_preds in zip(titles_true, title_predictions):
        acc += t_true in t_preds[:k]
    return acc / len(titles_true)


def mean_reciprocal_rank(*, titles_true, title_predictions, k=1):
    """ Assert title_predictions are sorted descending. """
    assert len(titles_true) == len(title_predictions)
    rr = 0
    for t_true, t_preds in zip(titles_true, title_predictions):
        for i in range(k):
            if t_true == t_preds[i]:
                rr += 1 / (i + 1)
                break
    return rr / len(titles_true)


def generate_predictions(
    predict_fn,
    queries: list[str],
    **predict_kwargs
) -> list[list[str]]:
    preds = []
    for query in tqdm(queries):
        q_tokens = tokenize(query)
        q_preds = predict_fn(q_tokens, **predict_kwargs)
        q_preds = [p[0] for p in q_preds]  # drop scores, keep titles only
        preds.append(q_preds)
    return preds


class BM25:

    def __init__(self, filemap: pd.DataFrame):
        self.n_articles = 0
        self.article_token_cnt = None       # article -> dict with token count in this article
        self.inverted_index = None          # token   -> set of article titles with this token
        self.n_articles_w_token = None      # token   -> number of articles with this token
        self.article_len = None             # article -> number of tokens in article
        self.filemap = filemap

    def fit(self, titles):
        assert len(titles) == len(set(titles))
        self.n_articles = len(titles)
        self.article_token_cnt = dict()
        self.inverted_index = defaultdict(set)
        self.article_len = dict()

        for title in tqdm(titles):
            text = get_article_text(get_article_path(title, filemap=self.filemap))
            tokens = tokenize(text)
            self.article_len[title] = len(tokens)
            local_article_token_cnt = defaultdict(int)
            for tok in tokens:
                local_article_token_cnt[tok] += 1
                self.inverted_index[tok].add(title)
            self.article_token_cnt[title] = local_article_token_cnt

        self.mean_article_len = sum(self.article_len.values()) / self.n_articles
        self.n_articles_w_token = {tok: len(articles) for tok, articles in self.inverted_index.items()}

    def predict(self, q_tokens, k1, k2, b, top_k=10) -> tuple[str, float]:
        query_token_cnt = defaultdict(int)
        for tok in q_tokens:
            query_token_cnt[tok] += 1

        article_score = defaultdict(float)
        rel_article_titles = set()

        for tok in q_tokens:
            if self.n_articles_w_token.get(tok, 0) == 0:  # use get to save memory accessing defaultdict
                continue

            rel_article_titles.update(self.inverted_index[tok])
            for title in rel_article_titles:
                K = k1 * (1 - b) + k1 * b * self.article_len[title] / self.mean_article_len
                article_cnt = self.article_token_cnt[title].get(tok, 0)  # use get to save memory accessing defaultdict
                x = math.log(self.n_articles + 1) - math.log(self.n_articles_w_token[tok])
                x *= (k1 + 1) * article_cnt / (K + article_cnt)
                x *= (k2 + 1) * query_token_cnt[tok] / (k2 + query_token_cnt[tok])
                article_score[title] += x

        article_score = sorted(article_score.items(), key=lambda x: x[1], reverse=True)
        article_score = article_score[:top_k]
        return article_score

    def predict_tfidf(self, q_tokens, top_k=10) -> tuple[str, float]:
        """ Use simple tf-idf features to predict. """
        query_token_cnt = defaultdict(int)
        for tok in q_tokens:
            query_token_cnt[tok] += 1

        article_score = defaultdict(float)
        rel_article_titles = set()

        for tok in q_tokens:
            if self.n_articles_w_token.get(tok, 0) == 0:  # use get to save memory accessing defaultdict
                continue

            rel_article_titles.update(self.inverted_index[tok])
            for title in rel_article_titles:
                article_cnt = self.article_token_cnt[title].get(tok, 0)  # use get to save memory accessing defaultdict
                x = math.log(self.n_articles + 1) - math.log(self.n_articles_w_token[tok])
                x *= math.log(article_cnt + 1)
                article_score[title] += x

        article_score = sorted(article_score.items(), key=lambda x: x[1], reverse=True)
        article_score = article_score[:top_k]
        return article_score
