import pandas as pd


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


def compute_metrices(*, predictions: list[list[str]], labels: list[str]):
    acc1 = accuracy(titles_true=labels, title_predictions=predictions, k=1)
    acc10 = accuracy(titles_true=labels, title_predictions=predictions, k=10)
    mrr10 = mean_reciprocal_rank(titles_true=labels, title_predictions=predictions, k=10)
    return dict(acc1=acc1, acc10=acc10, mrr10=mrr10, n_queries=len(labels))


def reranked_metrices_for_different_orig_top_n(
    reranked_preds: pd.DataFrame, 
    labels_test: pd.DataFrame, 
    score_col: str,
    n_list=[10, 25, 50],
) -> pd.DataFrame:
    res = []

    for n in n_list:
        sub = reranked_preds.query('pred_ix_orig < @n')
        assert sub.shape[0] == n * reranked_preds['query'].nunique()
        sub = sub.sort_values(['query', score_col], ascending=[True, False]).reset_index(drop=True)
        
        # `agg` should preserve order
        reranked_preds_agg = sub.groupby('query')['title'].agg(list).to_dict()
        
        queries = list(reranked_preds_agg.keys())
        titles_pred = [reranked_preds_agg[q] for q in queries]
        titles_true = labels_test.set_index('query').loc[queries, 'title'].tolist()
        
        m = compute_metrices(predictions=titles_pred, labels=titles_true)
        res.append(m)
        
    res = pd.DataFrame(res, index=[f'top_n: {n}' for n in n_list])
    return res