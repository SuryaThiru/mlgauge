import os
from copy import deepcopy

from tqdm import tqdm
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import pmlb
import openml

from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from mlgauge.method import Method
from mlgauge.utils import redirect_stdout, colors


class Analysis:
    """The analysis class to run the method comparisons.

    The class gathers datasets, methods and runs the given methods across
    different datasets and compiles the results.

    Attributes:
        results (xr.DataArray): Named array containing resulting metrics of the analysis.

                                The dimensions are named "datasets", "methods", "metrics", "splits".
                                You can index on each dimension using the name of the dataset, method, metrics and split ("train"/"test" if method uses test, "fold_1", "fold_2", ... otherwise) using the ``loc`` attribute similar to pandas.

                                For example to identify the `test` `mse` score of your `linear` model on the `houses` dataset:

                                .. code-block:: python

                                    result.loc['houses', 'linear', 'mse', 'test']


                                .. note::

                                    When integer IDs are specified for openml datasets, the ``results`` attribute's dataset key will be set as string.

                                Refer the documentation of `xarray <https://xarray.pydata.org/en/stable/quick-overview.html>`_ for a more detailed usage.
    """

    def __init__(
        self,
        methods,
        metric_names=None,
        datasets="all",
        n_datasets=20,
        data_source="pmlb",
        drop_na=False,
        use_test_set=True,
        test_size=0.25,
        random_state=None,
        output_dir=None,
        local_cache_dir=None,
    ):
        """Initialize analysis.

        Args:
            methods (list): List of tuple containing the method name and a method object.
            metric_names (list): List of strings representing the names of the metric. The names are only used to represent the metrics output by the method objects.  If `None` will not collect metrics from methods.

                             The size of the list should be the same as that returned by the `Method`'s instance train and test methods.
            datasets (str or list): One of the following options:

                                **"all"**: randomly select `n_datasets` from all available datasets in pmlb.

                                **"classification"**: randomly select `n_datasets` from all available classification datasets in pmlb.

                                **"regression"**: randomly select `n_datasets` from all available regression datasets in pmlb.

                                **list of strings**: a list of valid pmlb/openml dataset names.
                                **list of ints**: a list of valid openml dataset IDs. This is recommended for openml to avoid issues with versions.

                                **list of ('dataset_name', (X, y)) tuples**: Use the method to pass a custom dataset in the X y format.

                                **list of ('dataset_name', (X_train, y_train), (X_test, y_test)) tuples**: Use the method to pass a custom training and testing set in the X y format.

                            Here, X y could be a numpy array or a pandas DataFrame, using a DataFrame will allow the input feature names to be passed to the methods.

            n_datasets (int): Number of datasets to randomly sample from the available pmlb datasets. Ignored if `datasets` is not a string.
            data_source (str): Source to fetch from when dataset names/IDs are passed. 'pmlb' or 'openml'
            drop_na (bool): If True will drop all rows in the dataset with null values.
            random_state (None, int or RandomState instance): seed for the PRNG.
            use_test_set (bool): If the methods use a testing set.
            test_size (float): The size of the test set. Ignored if `use_test_set` is False.
            output_dir (str): Path of the output directory where method artifacts will be stored. A separate directory for each method will be created inside the directory. Defaults to an "output" directory in the current working directory.
            local_cache_dir (str): Local cache to use for pmlb datasets. If None will not use cached data.
        """
        self.random_state = check_random_state(random_state)
        self.seed = self.random_state.get_state()[1][
            0
        ]  # will be used with train-test split to ensure reproducibility outside class

        self.__methods = self._precheck_methods(methods)
        self.metric_names = metric_names

        if data_source != "openml" and data_source != "pmlb":
            raise TypeError("Data source must be 'openml' or 'pmlb'")
        if data_source == "openml" and not isinstance(datasets, list):
            raise TypeError("Provide list of dataset IDs/names for openml")

        self.data_source = data_source
        self.datasets = self._precheck_dataset(datasets)
        if isinstance(
            self.datasets, str
        ):  # expand "all", "classification" or "regression"
            self.datasets = self._expand_dataset_str(self.datasets, n_datasets)

            # display collected datasets
            print(f"{colors.GREEN}Collected datasets{colors.ENDC}")
            print(
                "\n".join(
                    f"{colors.CYAN}{{0: >2}}{colors.ENDC}: {{1}}".format(*k)
                    for k in enumerate(self.datasets, 1)
                )
            )
        self.drop_na = drop_na
        self.local_cache_dir = local_cache_dir

        self.use_test_set = use_test_set
        self.test_size = test_size

        # create output directory
        self.output_dir = (
            output_dir if output_dir else os.path.join(os.getcwd(), "output")
        )
        i = 1
        while os.path.exists(os.path.join(self.output_dir, f"Analysis_{i}")):
            i += 1
        self.output_dir = os.path.join(self.output_dir, f"Analysis_{i}")
        os.makedirs(self.output_dir, exist_ok=True)

        self.results = self._initialize_results()

    def run(self):
        """Load the datasets, run the methods and collect the results."""
        # redirect stdout
        with redirect_stdout() as stdout:

            # linespacing logic (18 additional chars for title etc.)
            _datasets = map(
                lambda x: x[0] if isinstance(x, tuple) else x, self.datasets
            )  # get dataset names
            maxl = min(max([len(str(x)) for x in _datasets]) + 18, 80)

            # iterate datasets
            datasets = tqdm(self.datasets, file=stdout, dynamic_ncols=True)
            for dataset in datasets:
                _dataset_name = dataset[0] if isinstance(dataset, tuple) else dataset
                datasets.set_description(
                    f"{{0: <{maxl}}}".format(
                        f"{colors.GREEN}Datasets [{_dataset_name}]{colors.ENDC}"
                    )
                )
                if self.use_test_set:
                    (
                        dataset_name,
                        feature_names,
                        category_indicator,
                        (X_train, y_train),
                        (X_test, y_test),
                    ) = self._get_dataset(dataset)
                else:
                    (
                        dataset_name,
                        feature_names,
                        category_indicator,
                        (X_train, y_train),
                    ) = self._get_dataset(dataset)

                # iterate methods
                methods = tqdm(
                    self.__methods, leave=False, file=stdout, dynamic_ncols=True
                )
                for method_name, method in methods:
                    methods.set_description(
                        f"{{0: <{maxl}}}".format(
                            f"{colors.CYAN}Models   [{method_name}]{colors.ENDC}"
                        )
                    )
                    method = deepcopy(method)
                    # set attributes for the dataset and method
                    method.set_test_set(self.use_test_set)
                    # create output directory
                    output_dir = os.path.join(
                        self.output_dir, str(dataset_name), method_name
                    )
                    os.makedirs(output_dir, exist_ok=True)
                    method.set_output_dir(output_dir)

                    # get training scores
                    train_scores = method.train(
                        X_train, y_train, feature_names, category_indicator
                    )

                    # change keys for result to string
                    if isinstance(dataset_name, int):
                        dataset_name = str(dataset_name)

                    # get optional testing scores
                    if self.use_test_set:
                        test_scores = method.test(
                            X_test, y_test, feature_names, category_indicator
                        )
                        if self.metric_names:
                            self.results.loc[
                                dataset_name, method_name, :, "train"
                            ] = np.array(train_scores)
                            self.results.loc[
                                dataset_name, method_name, :, "test"
                            ] = np.array(test_scores)
                    else:
                        if self.metric_names:
                            self.results.loc[dataset_name, method_name] = np.array(
                                train_scores
                            )

        # TODO recursively remove empty directories

    def _precheck_dataset(self, datasets):
        """Check if the passed value for the datasets input argument is correct.

        Raises:
            TypeError: raised if the methods argument of incorrect type is passed.
            ValueError: raised if an invalid value is passed.
        """
        if isinstance(datasets, str):
            if datasets not in ["all", "classification", "regression"]:
                raise ValueError(
                    "String input for datasets should be one of 'all', 'classification' or 'regression'."
                )
        elif isinstance(datasets, list):
            for d in datasets:
                if isinstance(d, str):
                    # should be a valid pmlb dataset name
                    if self.data_source == "pmlb":
                        if d not in pmlb.dataset_names:
                            raise ValueError(f"Dataset {d} not in pmlb")

                elif isinstance(d, int):
                    if self.data_source != "openml":
                        raise ValueError("Integer data IDs are only valid for OpenML")

                elif isinstance(d, tuple):
                    if not isinstance(d[0], str):
                        raise ValueError(
                            "First element of the tuple must be the name of the dataset"
                        )
                    if len(d) not in [2, 3]:
                        raise ValueError(
                            "Custom dataset input should be a tuple of length 2 or 3"
                        )

                else:
                    raise TypeError(f"Invalid type {type(d)} for dataset.")
        else:
            raise TypeError(f"Invalid type {type(datasets)} for datasets.")
        return datasets

    def _precheck_methods(self, methods):
        """Check if the passed value for the methods is a list of `Method` instances

        Raises:
            TypeError: raised if the dataset argument of incorrect type is passed
        """
        for _, m in methods:
            if not isinstance(m, Method):
                raise TypeError(
                    f"Input methods should be an instance of Method class, found {type(m)} instead"
                )
        return methods

    def _expand_dataset_str(self, dataset_str, n_datasets):
        """Convert the dataset string to list of pmlb dataset names"""
        if dataset_str == "all":
            datasets = self.random_state.choice(pmlb.dataset_names, n_datasets)
        elif dataset_str == "classification":
            datasets = self.random_state.choice(
                pmlb.classification_dataset_names, n_datasets
            )
        elif dataset_str == "regression":
            datasets = self.random_state.choice(
                pmlb.regression_dataset_names, n_datasets
            )
        return datasets

    def _initialize_results(self):
        """Define a results object to store results generated during analysis."""
        if not self.metric_names:
            return None

        dims = ["datasets", "methods", "metrics", "splits"]

        # co-ords
        dataset_names = [self._get_dataset_name(data) for data in self.datasets]
        method_names = [name for (name, _) in self.__methods]
        metric_names = self.metric_names
        coords = {
            "datasets": dataset_names,
            "methods": method_names,
            "metrics": metric_names,
        }

        if self.use_test_set:
            coords["splits"] = ["train", "test"]
        else:
            n_folds = self.__methods[0][1].cv  # get the number of folds from a method
            coords["splits"] = ["fold_" + str(i) for i in range(1, n_folds + 1)]

        return xr.DataArray(np.nan, coords=coords, dims=dims)

    def _get_dataset_name(self, dataset):
        """Get the supplied name of the dataset"""
        if isinstance(dataset, str) or isinstance(dataset, int):
            return dataset
        elif isinstance(dataset, tuple):
            return dataset[0]

    def _get_dataset(self, dataset):
        """Load and return the dataset as X, y numpy arrays"""
        category_indicator = None  # list indicating categorical columns

        if isinstance(dataset, str):  # Use pmlb or openml
            if self.data_source == "pmlb":
                data = pmlb.fetch_data(dataset, local_cache_dir=self.local_cache_dir)

                # Get feature names and get X,y numpy arrays
                X = data.drop("target", axis=1)
                y = data["target"]
            elif self.data_source == "openml":
                X, y, category_indicator = self._fetch_openml_data(dataset)

        elif isinstance(dataset, int):
            X, y, category_indicator = self._fetch_openml_data(dataset)

        elif isinstance(dataset, tuple):
            if len(dataset) == 2:
                dataset, (X, y) = dataset

            else:  # Test set present in the inputs, will directly return
                dataset, (X_train, y_train), (X_test, y_test) = dataset
                feature_names = self._get_feature_names(X_train)
                X_train, y_train = self._format_na(X_train, y_train)

                if self.use_test_set:
                    X_test, y_test = self._format_na(X_test, y_test)
                    return (
                        dataset,
                        feature_names,
                        category_indicator,
                        (X_train, y_train),
                        (X_test, y_test),
                    )
                else:
                    return (
                        dataset,
                        feature_names,
                        category_indicator,
                        (X_train, y_train),
                    )

        if self.use_test_set:  # Perform train-test splits
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                shuffle=True,
                random_state=self.seed,
            )
            feature_names = self._get_feature_names(X_train)
            X_train, y_train = self._format_na(X_train, y_train)
            X_test, y_test = self._format_na(X_test, y_test)
            return (
                dataset,
                feature_names,
                category_indicator,
                (X_train, y_train),
                (X_test, y_test),
            )
        else:  # Directly format and return train set
            feature_names = self._get_feature_names(X)
            X_train, y_train = self._format_na(X, y)
            return dataset, feature_names, category_indicator, (X_train, y_train)

    def _fetch_openml_data(self, dataset_id):
        """Get the openml dataset with the category indicator"""
        data = openml.datasets.get_dataset(dataset_id)
        X, y, category_indicator, attribute_names = data.get_data(
            dataset_format="dataframe", target=data.default_target_attribute
        )
        return X, y, category_indicator

    def _get_feature_names(self, X):
        """Get the list of feature names from input data"""
        if hasattr(X, "columns"):
            return X.columns.tolist()  # Dataframe column names
        else:
            return list(map(str, range(X.shape[1])))  # Array indices as strings

    def _format_na(self, X, y):
        """Convert data to numpy arrays and drop null valued rows"""
        # convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y = y.values

        # remove rows with NaNs
        if self.drop_na:
            idx = ~(np.isnan(X).any(axis=1) | np.isnan(y).any())
            X, y = X[idx], y[idx]

        return X, y

    def get_result(self):
        """get result of the analysis.

        Returns:
            (xr.DataArray): A 4d named array containing the result metrics.
        """
        return self.results

    def get_result_as_df(self, metric=None, train=False, mean_folds=True):
        """Get results as a pandas dataframe.

        Args:
            metric (str): Enter the metric string for which the result should
                            be displayed. Defaults to the first name in `metric_names`.
            train (bool): If true, will also return the train scores. Ignored if `use_test_set` is False.
            mean_folds (bool): If true, will return mean and std deviation of the k-fold results, otherwise returns all folds.
                               Ignored if `use_test_set` is True.

        Returns:
            (pd.DataFrame): Pandas dataframe with datasets for rows.
                            When `use_test_set` is True, the columns contain the train and test results
                            otherwise the mean and standard deviation of the k-fold validation is returned.
                            If `mean_folds` is set to False, all folds scores are returned.
        """
        if not self.metric_names:
            raise AttributeError("No results available to show.")

        if not metric:
            metric = self.metric_names[-1]

        dataset_names = [self._get_dataset_name(data) for data in self.datasets]
        method_names = [name for (name, _) in self.__methods]
        index = dataset_names

        if self.use_test_set:
            if train:  # train & test split scores
                columns = pd.MultiIndex.from_product([method_names, ["train", "test"]])

                df = pd.DataFrame(columns=columns, index=index)
                df.loc[:, (slice(None), "train")] = self.results.loc[
                    :, :, metric, "train"
                ].values
                df.loc[:, (slice(None), "test")] = self.results.loc[
                    :, :, metric, "test"
                ].values
            else:  # only test split scores
                columns = method_names

                df = pd.DataFrame(
                    self.results.loc[:, :, metric, "test"].values,
                    columns=columns,
                    index=index,
                )

        else:  # return mean & standard deviation across folds
            if mean_folds:
                columns = pd.MultiIndex.from_product([method_names, ["mean", "std"]])

                df = pd.DataFrame(columns=columns, index=index)
                df.loc[:, (slice(None), "mean")] = (
                    self.results.loc[:, :, metric].mean("splits").values
                )
                df.loc[:, (slice(None), "std")] = (
                    self.results.loc[:, :, metric].std("splits").values
                )
            else:
                n_folds = self.results.shape[-1]
                fold_cols = ["fold_" + str(i + 1) for i in range(n_folds)]
                columns = pd.MultiIndex.from_product([method_names, fold_cols])

                df = pd.DataFrame(columns=columns, index=index)
                for col in fold_cols:
                    df.loc[:, (slice(None), col)] = self.results.loc[:, :, metric, col]

        df = df.rename_axis(index="datasets")
        return df

    def plot_results(self, metric=None, ax=None):
        """Plot results as a bar plot.

        Args:
            metric (str): Enter the metric string for which the result should
                            be displayed.
            ax (matplotlib Axes): Axes in which to draw the plot, otherwise use the currently-active Axes.

        Returns:
            (matplotlib Axes): Axes containing the plot.
        """
        # if ax is None:
        #     ax = plt.gca()
        if not self.metric_names:
            raise AttributeError("No results available to show.")

        metric = metric if metric else self.metric_names[0]

        if self.use_test_set:  # only test set
            df = self.get_result_as_df(metric)
            df_bar = df.reset_index().melt(
                id_vars=["datasets"], var_name="methods", value_name=metric
            )
            return sns.barplot(
                data=df_bar, x="datasets", y=metric, hue="methods", ax=ax
            )

        else:  # k-fold validation
            df = self.get_result_as_df(metric, mean_folds=False)
            df_bar = (
                df.stack([0, 1])
                .rename_axis(index=["datasets", "methods", "folds"])
                .reset_index()
                .rename(columns={0: metric})
            )

            return sns.barplot(
                data=df_bar, x="datasets", y=metric, capsize=0.1, hue="methods", ax=ax
            )
