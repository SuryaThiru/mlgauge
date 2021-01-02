import tempfile
import os

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
import pytest

from mlgauge import Analysis, Method, SklearnMethod


SEED = 42


@pytest.fixture
def regressor():
    return SklearnMethod(DummyRegressor(), ["r2", "max_error"])


# Test all allowed data formats
class TestDataFormat:
    def test_string(self, regressor):
        # should work with "all", "classification", "regression"
        an = Analysis(
            methods=[("dummy", regressor)],
            metric_names=["r2", "max_error"],
            datasets="all",
            n_datasets=5,
            random_state=SEED,
        )
        assert len(an.datasets) == 5

        an = Analysis(
            methods=[("dummy", regressor)],
            metric_names=["r2", "max_error"],
            datasets="classification",
            n_datasets=5,
            random_state=SEED,
        )
        assert len(an.datasets) == 5

        an = Analysis(
            methods=[("dummy", regressor)],
            metric_names=["r2", "max_error"],
            datasets="regression",
            n_datasets=5,
            random_state=SEED,
        )
        assert len(an.datasets) == 5

    def test_list(self, regressor):
        # should work with a list of valid pmlb datasets names
        an = Analysis(
            methods=[("dummy", regressor)],
            metric_names=["r2", "max_error"],
            datasets=["503_wind", "581_fri_c3_500_25", "adult", "cars", "pima"],
            random_state=SEED,
        )
        assert len(an.datasets) == 5

        # throw error if invalid dataset name is passed
        with pytest.raises(ValueError):
            an = Analysis(
                methods=[("dummy", regressor)],
                metric_names=["r2", "max_error"],
                datasets=["adult", "invalid"],
                random_state=SEED,
            )

    def test_tuple(self, regressor):
        # should works with a list of (X, y) tuples
        datasets = [
            (
                "data_1",
                *make_regression(n_samples=200, n_features=5, random_state=SEED),
            ),
            (
                "data_2",
                *make_regression(n_samples=1000, n_features=50, random_state=SEED),
            ),
        ]
        an = Analysis(
            methods=[("dummy", regressor)],
            metric_names=["r2", "max_error"],
            datasets=datasets,
            random_state=SEED,
        )
        assert len(an.datasets) == 2

    def test_tuple_train_test(self, regressor):
        # should works with a list of (X_train, y_train, X_test, y_test) tuples
        datasets = [
            (
                "data_1",
                *make_regression(n_samples=200, n_features=5, random_state=SEED),
            ),
            (
                "data_2",
                *make_regression(n_samples=1000, n_features=50, random_state=SEED),
            ),
        ]

        def test_split(data):
            name, X, y = data
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=SEED)
            return name, (X_train, y_train), (X_test, y_test)

        datasets = list(map(test_split, datasets))

        an = Analysis(
            methods=[("dummy", regressor)],
            metric_names=["r2", "max_error"],
            datasets=datasets,
            random_state=SEED,
        )
        assert len(an.datasets) == 2

    def test_mixed(self, regressor):
        # should work with a mix of strings, tuples, tuple of tuples
        datasets = [
            (
                "data_1",
                *make_regression(n_samples=200, n_features=5, random_state=SEED),
            ),
            (
                "data_2",
                *make_regression(n_samples=1000, n_features=50, random_state=SEED),
            ),
        ]

        def test_split(data):
            name, X, y = data
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=SEED)
            return name, (X_train, y_train), (X_test, y_test)

        datasets = (
            list(map(test_split, datasets)) + datasets + ["adult", "cars", "pima"]
        )

        an = Analysis(
            methods=[("dummy", regressor)],
            metric_names=["r2", "max_error"],
            datasets=datasets,
            random_state=SEED,
        )
        assert len(an.datasets) == 7

# Test output directory
def test_output_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        skl = SklearnMethod(
            LinearRegression(), ["neg_mean_squared_error", "r2"], export_model=True
        )
        an = Analysis(
            methods=[("dummy", skl)],
            metric_names=["r2", "max_error"],
            datasets=["adult", "cars", "pima"],
            random_state=SEED,
            output_dir=tmpdir,
        )
        an.run()

        out_dir = os.path.join(tmpdir, "Analysis_1")
        exports = map(
            lambda x: os.path.join(out_dir, x, "dummy", "estimator.joblib"),
            ["adult", "cars", "pima"],
        )

        for export in exports:
            assert os.path.exists(export)