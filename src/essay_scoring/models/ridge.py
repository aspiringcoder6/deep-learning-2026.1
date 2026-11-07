"""TF-IDF with ridge regression for comparison on the shared split."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline

from essay_scoring.data import Essay


def fit_ridge(essays: list[Essay]) -> Pipeline:
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), max_features=30_000),
        Ridge(alpha=1.0, solver="lsqr"),
    )
    return model.fit([essay.text for essay in essays], [essay.score for essay in essays])
