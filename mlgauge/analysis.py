import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
import pmlb

from mlgauge.method import Method


class Analysis:
    """The analysis class to run the method comparisons.

    The class gathers datasets, methods and runs the given methods across
    different datasets and compiles the results.
    """

    def __init__(
        self,
        methods,
        metric_names,
        datasets="all",
        n_datasets=None,
        drop_na=False,
        use_test_set=True,
        test_size=0.25,
        random_state=None,
        output_dir=None,
    ):
        """Initialize analysis.

        Args:
            methods (list): List of tuple containing the method name and a method object.
            metric_names (list): List of strings representing the names of the metric. The names are only used to represent the metrics output by the method objects.
                             The size of the list should be the same as that returned by the `Method`'s instance train and test methods.
            datasets (str or list): One of the following options:

                                *"all"*: randomly select `n_datasets` from all available datasets in pmlb.

                                *"classification"*: randomly select `n_datasets` from all available classification datasets in pmlb.

                                *"regression"*: randomly select `n_datasets` from all available regression datasets in pmlb.

                                *list of strings*: a list of valid pmlb dataset names.

                                *list of ('dataset_name', (X, y)) tuples*: Use the method to pass a custom dataset in the X y array format.

                                *list of ('dataset_name', (X_train, y_train), (X_test, y_test)) tuples*: Use the method to pass a custom training and testing set in the X y array format.

            n_datasets (int): Number of datasets to randomly sample from the available pmlb datasets. Ignored if `datasets` is a string.

            drop_na (bool): If True will drop all rows in the dataset with null values.
            random_state (None, int or RandomState instance): seed for the PRNG.
            use_test_set (bool): If the methods use a testing set.
            test_size (float): The size of the test set. Ignored if `use_test_set` is False.
            output_dir (str) : Path of the output directory where method artifacts will be stored. A separate directory for each method will be created inside the directory. Defaults to an "output" directory in the current working directory.
        """
        self.random_state = check_random_state(random_state)

        self.__methods = self._precheck_methods(methods)
        self.metric_names = metric_names

        self.datasets = self._precheck_dataset(datasets)
        if isinstance(self.datasets, str):
            self.datasets = self._expand_dataset_str(self.datasets, n_datasets)
        self.drop_na = drop_na

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

        self.results = None

    def run(self):
        """Load the datasets, run the methods and collect the results."""
        for dataset in self.datasets:
            if self.use_test_set:
                (
                    dataset_name,
                    feature_names,
                    (X_train, y_train),
                    (X_test, y_test),
                ) = self._get_dataset(dataset)
            else:
                dataset_name, feature_names, (X_train, y_train) = self._get_dataset(
                    dataset
                )

            for method_name, method in self.__methods:
                # set attributes for the dataset and method
                method.set_test_set(self.use_test_set)
                method.set_feature_names(feature_names)
                # create output directory
                output_dir = os.path.join(self.output_dir, dataset_name, method_name)
                os.makedirs(output_dir, exist_ok=True)
                method.set_output_dir(output_dir)

                # get training scores
                train_scores = method.train(X_train, y_train)

                # get optional testing scores
                if self.use_test_set:
                    test_scores = method.test(X_test, y_test)

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
                    if d not in pmlb.dataset_names:
                        raise ValueError(f"Dataset {d} not in pmlb")
                elif isinstance(d, tuple):
                    if not isinstance(d[0], str):
                        raise ValueError(
                            "First element of the tuple must be the name of the dataset"
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
        """Convert the dataset argument"""
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

    def _get_dataset(self, dataset):
        """Load and return the dataset as X, y numpy arrays"""
        if isinstance(dataset, str):  # Use pmlb
            data = pmlb.fetch_data(dataset)

            # Get feature names and get X,y numpy arrays
            X = data.drop("target", axis=1)
            y = data["target"]

        elif isinstance(dataset, tuple):
            if len(dataset) == 2:
                dataset, (X, y) = dataset

            else:  # Test set present in the inputs, will directly return
                dataset, (X_train, y_train), (X_test, y_test) = dataset
                feature_names = self._get_feature_names(X_train)
                X_train, y_train = self._format_na(X_train, y_train)

                if self.use_test_set:
                    X_test, y_test = self._format_na(X_test, y_test)
                    return dataset, feature_names, (X_train, y_train), (X_test, y_test)
                else:
                    return dataset, feature_names, (X_train, y_train)

        if self.use_test_set:  # Perform train-test splits
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                shuffle=True,
                random_state=self.random_state,
            )
            feature_names = self._get_feature_names(X_train)
            X_train, y_train = self._format_na(X_train, y_train)
            X_test, y_test = self._format_na(X_test, y_test)
            return dataset, feature_names, (X_train, y_train), (X_test, y_test)
        else:  # Directly format and return train set
            feature_names = self._get_feature_names(X)
            X_train, y_train = self._format_na(X, y)
            return dataset, feature_names, (X_train, y_train)

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
        if isinstance(y, pd.DataFrame):
            y = y.values

        # remove rows with NaNs
        if self.drop_na:
            idx = ~(np.isnan(X).any(axis=1) | np.isnan(y).any())
            X, y = X[idx], y[idx]

        return X, y

    def get_result(self):
        return self.results

    def print_results(self, metric=None):
        """
        Print results as a table.

        Args:
            metric (str) : Enter the metric string for which the result should
                            be displayed.
        """
        pass

    def plot_results(self, metric=None, ax=None):
        """
        Plot results in a bar plot.

        Args:
            metric (str) : Enter the metric string for which the result should
                            be displayed.
            ax (matplotlib Axes) : Axes in which to draw the plot, otherwise use the currently-active Axes.
        """
        # TODO  use plt.gca and see how seaborn uses the axes parameter
        pass
