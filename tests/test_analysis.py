import tempfile
import os

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
import pmlb
import pytest

from mlgauge import Analysis, Method, SklearnMethod


SEED = 42
PMLB_CACHE = "./pmlb_cache"


@pytest.fixture
def regressor():
    return SklearnMethod(DummyRegressor(), ["r2", "max_error"])


# Test all allowed data formats
class TestDataFormat:
    def test_string(self, regressor, tmp_path):
        # should work with "all", "classification", "regression"
        an = Analysis(
            methods=[("dummy", regressor)],
            metric_names=["r2", "max_error"],
            datasets="all",
            n_datasets=5,
            random_state=SEED,
            output_dir=tmp_path,
            local_cache_dir=PMLB_CACHE,
        )
        assert len(an.datasets) == 5

        an = Analysis(
            methods=[("dummy", regressor)],
            metric_names=["r2", "max_error"],
            datasets="classification",
            n_datasets=5,
            random_state=SEED,
            output_dir=tmp_path,
            local_cache_dir=PMLB_CACHE,
        )
        assert len(an.datasets) == 5

        an = Analysis(
            methods=[("dummy", regressor)],
            metric_names=["r2", "max_error"],
            datasets="regression",
            n_datasets=5,
            random_state=SEED,
            output_dir=tmp_path,
            local_cache_dir=PMLB_CACHE,
        )
        assert len(an.datasets) == 5

    def test_list(self, regressor, tmp_path):
        # should work with a list of valid pmlb datasets names
        an = Analysis(
            methods=[("dummy", regressor)],
            metric_names=["r2", "max_error"],
            datasets=["503_wind", "581_fri_c3_500_25", "adult", "cars", "pima"],
            random_state=SEED,
            output_dir=tmp_path,
            local_cache_dir=PMLB_CACHE,
        )
        assert len(an.datasets) == 5

        # throw error if invalid dataset name is passed
        with pytest.raises(ValueError):
            an = Analysis(
                methods=[("dummy", regressor)],
                metric_names=["r2", "max_error"],
                datasets=["adult", "invalid"],
                random_state=SEED,
                output_dir=tmp_path,
                local_cache_dir=PMLB_CACHE,
            )

    def test_tuple(self, regressor, tmp_path):
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
            output_dir=tmp_path,
            local_cache_dir=PMLB_CACHE,
        )
        assert len(an.datasets) == 2

    def test_tuple_train_test(self, regressor, tmp_path):
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
            output_dir=tmp_path,
            local_cache_dir=PMLB_CACHE,
        )
        assert len(an.datasets) == 2

    def test_mixed(self, regressor, tmp_path):
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
            output_dir=tmp_path,
            local_cache_dir=PMLB_CACHE,
        )
        assert len(an.datasets) == 7


# Test dropna
class MockMethodNA(Method):
    """Class to check if missing values are properly dropped."""

    def __init__(self, dropna):
        self.drop_na = dropna
        super().__init__()

    def train(self, X, y, feature_names):
        dat = np.hstack([X, y.reshape(-1, 1)])
        if self.drop_na:
            assert not np.isnan(dat).any(), "data has missing values"
        else:
            assert np.isnan(dat).any(), "data does not have missing values"


@pytest.mark.parametrize("dropna", [True, False])
def test_dropna(dropna, tmp_path):
    X, y = make_regression(n_samples=200, n_features=5, random_state=SEED)
    X[0, 2] = np.nan
    y[3] = np.nan

    an = Analysis(
        methods=[("mock", MockMethodNA(dropna=dropna))],
        metric_names=["r2", "max_error"],
        datasets=[("data_with_na", (X, y))],
        random_state=SEED,
        drop_na=dropna,
        use_test_set=False,
        output_dir=tmp_path,
        local_cache_dir=PMLB_CACHE,
    )
    an.run()


# Test output directory
def test_output_dir(tmp_path):
    skl = SklearnMethod(
        LinearRegression(), ["neg_mean_squared_error", "r2"], export_model=True
    )
    an = Analysis(
        methods=[("dummy", skl)],
        metric_names=["r2", "max_error"],
        datasets=["adult", "cars", "pima"],
        random_state=SEED,
        output_dir=tmp_path,
        local_cache_dir=PMLB_CACHE,
    )
    an.run()

    out_dir = os.path.join(tmp_path, "Analysis_1")
    exports = map(
        lambda x: os.path.join(out_dir, x, "dummy", "estimator.joblib"),
        ["adult", "cars", "pima"],
    )

    for export in exports:
        assert os.path.exists(export)


# Test results with train-test split
def test_result_test_split(tmp_path):
    linear = SklearnMethod(LinearRegression(), ["r2", "max_error"])
    tree = SklearnMethod(
        DecisionTreeRegressor(random_state=SEED),
        ["r2", "max_error"],
    )
    dummy = SklearnMethod(DummyRegressor(), ["r2", "max_error"])
    an = Analysis(
        methods=[("linear", linear), ("tree", tree), ("dummy", dummy)],
        metric_names=["r2", "max_error"],
        datasets="regression",
        n_datasets=3,
        random_state=SEED,
        use_test_set=True,
        output_dir=tmp_path,
        local_cache_dir=PMLB_CACHE,
    )
    an.run()

    assert an.results.shape == (3, 3, 2, 2)
    assert not np.isnan(an.results.values).any()

    linear = SklearnMethod(LinearRegression(), ["r2", "max_error"])
    tree = SklearnMethod(
        DecisionTreeRegressor(random_state=SEED),
        ["r2", "max_error"],
    )
    # check if the results match
    for data in an.datasets:
        X, y = pmlb.fetch_data(data, return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, shuffle=True, random_state=SEED
        )

        linear_r2, linear_max = linear.train(X_train, y_train)
        tree_r2, tree_max = tree.train(X_train, y_train)

        assert round(
            float(an.results.loc[data, "linear", "max_error", "train"].values), 4
        ) == round(linear_max, 4)

        assert round(
            float(an.results.loc[data, "tree", "max_error", "train"].values), 4
        ) == round(tree_max, 4)


# Test results with 5-fold validation
def test_result_cv(tmp_path):
    linear = SklearnMethod(LinearRegression(), ["r2", "max_error"], cv=5)
    tree = SklearnMethod(
        DecisionTreeRegressor(random_state=SEED),
        ["r2", "max_error"],
        cv=5,
    )
    dummy = SklearnMethod(DummyRegressor(), ["r2", "max_error"])
    an = Analysis(
        methods=[("linear", linear), ("tree", tree), ("dummy", dummy)],
        metric_names=["r2", "max_error"],
        datasets="regression",
        n_datasets=3,
        random_state=SEED,
        use_test_set=False,
        output_dir=tmp_path,
        local_cache_dir=PMLB_CACHE,
    )
    an.run()

    assert an.results.shape == (3, 3, 2, 5)
    assert not np.isnan(an.results.values).any()

    linear = SklearnMethod(LinearRegression(), ["r2", "max_error"])
    tree = SklearnMethod(
        DecisionTreeRegressor(random_state=SEED),
        ["r2", "max_error"],
    )
    linear.set_test_set(False)
    tree.set_test_set(False)
    # check if the results match
    for data in an.datasets:
        X, y = pmlb.fetch_data(data, return_X_y=True)

        linear_r2, linear_max = linear.train(X, y)
        tree_r2, tree_max = tree.train(X, y)

        an_linear_max = an.results.loc[
            data, "linear", "max_error"
        ].values  # check all folds
        np.testing.assert_allclose(an_linear_max, linear_max, rtol=1e-13)

        an_tree_max = an.results.loc[
            data, "tree", "max_error"
        ].values  # check all folds
        np.testing.assert_allclose(an_tree_max, tree_max, rtol=1e-13)


# Test results when no metric
def test_results_none(tmp_path):
    linear = SklearnMethod(LinearRegression(), ["r2", "max_error"])
    tree = SklearnMethod(
        DecisionTreeRegressor(random_state=SEED),
        ["r2", "max_error"],
    )
    dummy = SklearnMethod(DummyRegressor(), ["r2", "max_error"])
    an = Analysis(
        methods=[("linear", linear), ("tree", tree), ("dummy", dummy)],
        metric_names=None,
        datasets="regression",
        n_datasets=3,
        random_state=SEED,
        use_test_set=True,
        output_dir=tmp_path,
        local_cache_dir=PMLB_CACHE,
    )
    an.run()
    assert an.results == None

    with pytest.raises(AttributeError):
        an.get_result_as_df()

    an = Analysis(
        methods=[("linear", linear), ("tree", tree), ("dummy", dummy)],
        metric_names=None,
        datasets="regression",
        n_datasets=3,
        random_state=SEED,
        use_test_set=False,
        output_dir=tmp_path,
        local_cache_dir=PMLB_CACHE,
    )
    an.run()
    assert an.results == None

    with pytest.raises(AttributeError):
        an.get_result_as_df()


# Test results as dataframe
def test_get_results_as_df(regressor, tmp_path):
    skl = SklearnMethod(LinearRegression(), ["r2", "max_error"], export_model=True)
    # using test set
    an = Analysis(
        methods=[("dummy", regressor), ("linear", skl)],
        metric_names=["r2", "max_error"],
        datasets=["adult", "cars", "pima"],
        use_test_set=True,
        random_state=SEED,
        output_dir=tmp_path,
        local_cache_dir=PMLB_CACHE,
    )
    an.run()

    # only test set
    df = an.get_result_as_df("r2")
    assert (
        df.loc["adult", "dummy"]
        == an.results.loc["adult", "dummy", "r2", "test"].values.item()
    )
    df = an.get_result_as_df("max_error")
    assert (
        df.loc["cars", "linear"]
        == an.results.loc["cars", "linear", "max_error", "test"].values.item()
    )

    # with train & test set
    df = an.get_result_as_df("r2", train=True)
    assert (
        df.loc["adult", ("linear", "train")]
        == an.results.loc["adult", "linear", "r2", "train"].values.item()
    )

    # using 5-fold validation
    an = Analysis(
        methods=[("dummy", regressor), ("linear", skl)],
        metric_names=["r2", "max_error"],
        datasets=["adult", "cars", "pima"],
        use_test_set=False,
        random_state=SEED,
        output_dir=tmp_path,
        local_cache_dir=PMLB_CACHE,
    )
    an.run()

    # check the mean and std
    df = an.get_result_as_df("r2")
    assert (
        df.loc["adult", ("dummy", "mean")]
        == an.results.loc["adult", "dummy", "r2", :].mean().item()
    )
    assert (
        df.loc["adult", ("dummy", "std")]
        == an.results.loc["adult", "dummy", "r2", :].std().item()
    )

    df = an.get_result_as_df("max_error")
    assert (
        df.loc["cars", ("linear", "mean")]
        == an.results.loc["cars", "linear", "max_error", :].mean().item()
    )
    assert (
        df.loc["cars", ("linear", "std")]
        == an.results.loc["cars", "linear", "max_error", :].std().item()
    )

    # check multiple folds
    df = an.get_result_as_df("r2", mean_folds=False)
    assert df.loc["adult", ("dummy", slice(None))].shape == (5,)
    np.testing.assert_array_equal(
        df.loc["adult", ("dummy", slice(None))], an.results.loc["adult", "dummy", "r2"]
    )

    df = an.get_result_as_df("max_error", mean_folds=False)
    np.testing.assert_array_equal(
        df.loc["cars", ("linear", slice(None))],
        an.results.loc["cars", "linear", "max_error"],
    )
