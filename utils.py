import pandas as pd


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


def process_title(title: pd.Series):
    title = title.str.lower().str.strip()
    title = title.str.replace(r'([^\w\s])|_|-', '_', regex=True)
    return title
