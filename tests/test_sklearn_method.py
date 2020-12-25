import tempfile
import os

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import datasets
import joblib
import pytest

from mlgauge import SklearnMethod


SEED = 42


def test_regression():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    reg = LinearRegression().fit(X, y)

    skl = SklearnMethod(reg, ["neg_mean_squared_error", "r2"])
    train_scores = skl.train(X, y)
    test_scores = skl.test(X, y)

    assert len(train_scores) == len(test_scores) == 2


def test_cross_validation():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    reg = LinearRegression().fit(X, y)

    skl = SklearnMethod(reg, ["neg_mean_squared_error", "r2"], cv=3)
    skl.set_test_set(False)
    train_scores = skl.train(X, y)

    assert len(train_scores) == 2
    for score in train_scores:
        assert len(score) == 3

    with pytest.raises(AttributeError):
        skl.test(X, y)


def test_pipeline():
    # pipelines should be supported
    X, y = datasets.make_classification(random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    parameters = {"kernel": ("linear", "rbf"), "C": [1, 10]}
    grid_svm = GridSearchCV(SVC(), parameters)
    pipe = Pipeline([("scaler", StandardScaler()), ("svc", grid_svm)])

    skl = SklearnMethod(pipe, ["roc_auc", "f1", "accuracy"])
    train_scores = skl.train(X_train, y_train)
    test_scores = skl.test(X_test, y_test)

    assert len(train_scores) == len(test_scores) == 3


def test_export():
    # model should be exported
    with tempfile.TemporaryDirectory() as tmpdir:
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        y = np.dot(X, np.array([1, 2])) + 3
        reg = LinearRegression().fit(X, y)

        skl = SklearnMethod(reg, ["neg_mean_squared_error", "r2"], export_model=True)
        skl.set_output_dir(tmpdir)
        train_scores = skl.train(X, y)

        dumped = joblib.load(os.path.join(tmpdir, "estimator.joblib"))
        np.testing.assert_array_equal(reg.coef_, dumped.coef_)
