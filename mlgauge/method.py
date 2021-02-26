import os
import joblib

from sklearn.metrics import check_scoring
from sklearn.model_selection import cross_validate


class Method:
    """
    The baseclass for method definitions that will be used in the analysis.
    Inherit this class into your class to define your own methods.

    An `Analysis` instance will dynamically set the different attributes to different values based on the dataset that is currently used. These attributes will be available to the inheriting class to provide additional information.

    Attributes:
        output_dir (str): Path of the directory where the outputs should be saved.
                          Will be automatically assigned by `Analysis` class.
                          When defining a method based on this class, use this attribute to save any artifacts like plots, model dumps etc.
        feature_names (list): List of feature names of the input dataset.
        use_test_set (bool): method implements a `test` method when set to `True`.
        cv (int): The value indicates the number of folds used when `use_test_set` is set to False.
    """

    def __init__(self):
        self.use_test_set = True
        self.output_dir = os.path.join(os.getcwd(), "output")
        self.feature_names = None
        self.cv = 5  # only valid if use_test_set is False

    def set_output_dir(self, path):
        """Set output directory to save results of the method.

        Args:
            path (str): path of the directory where the outputs should be saved.
        """
        self.output_dir = path

    def set_test_set(self, use_test_set):
        """Specify if the method requires a test set.

        Args:
            use_test_set (bool): method implements a `test` method when set to `True`.
        """
        self.use_test_set = use_test_set

    def train(self, X_train, y_train, feature_names=None, category_indicator=None):
        """Train the model and return the training score.

        Args:
            X_train (array): array of training vector.
            y_train (array): array of target vector.
            feature_names (list): list of names of the features in X_train.
            category_indicator (list): list of boolean indicating whether a feature is a categorical variable.

        Raises:
            NotImplementedError: raised when called from the base class.
        """
        raise NotImplementedError

    def test(self, X_test, y_test, feature_names=None, category_indicator=None):
        """Evaluate the model and return the test score.

        Args:
            X_test (array): array of training vector.
            y_test (array): array of target vector.
            feature_names (list): list of names of the features in X_test
            category_indicator (list): list of boolean indicating whether a feature is a categorical variable.

        Raises:
            NotImplementedError: raised when called from the base class.
        """
        raise NotImplementedError


class SklearnMethod(Method):
    """
    A wrapper to directly use an sklearn estimator with an analysis.
    """

    def __init__(self, estimator, metrics, export_model=False, cv=5):
        """Initialize sklearn method.

        Args:
            estimator (estimator): An sklearn estimator or a pipeline.
            metrics (list): list of metric string or an sklearn callable metric function. Refer sklearn documentation for metrics.
            export_model (bool): Exports the sklearn estimator through joblib as `estimator(_fold_k).joblib` if set to True.
            cv (int): The cross-validation to use when `use_test_set` is False. Ignored otherwise.
        """
        super().__init__()
        self.estimator = estimator
        self.metrics = metrics
        self.export_model = export_model
        self.cv = cv

    def train(self, X_train, y_train, feature_names=None, category_indicator=None):
        """Train the model and return the training score.

        Args:
            X_train (array): array of training vector.
            y_train (array): array of target vector.
            feature_names (list): list of names of the features in X_train
            category_indicator (list): list of boolean indicating whether a feature is a categorical variable.

        Returns:
            list: list of metric scores evaluated on the training data.

        Raises:
            AttributeError: raised when method is called with `use_test_set` set to false
        """
        if self.use_test_set:
            self.estimator.fit(X_train, y_train)

            if self.export_model:
                dump_path = os.path.join(self.output_dir, "estimator.joblib")
                joblib.dump(self.estimator, dump_path)

            scores = []
            for metric in self.metrics:
                scorer = check_scoring(self.estimator, metric)
                score = scorer(self.estimator, X_train, y_train)
                scores.append(score)
            return scores
        else:
            scores_dict = cross_validate(
                self.estimator,
                X_train,
                y_train,
                cv=self.cv,
                scoring=self.metrics,
                return_estimator=self.export_model,
            )
            if self.export_model:
                for i, fold_estimator in enumerate(scores_dict["estimator"]):
                    dump_path = os.path.join(
                        self.output_dir, f"estimator_fold_{i+1}.joblib"
                    )
                    joblib.dump(fold_estimator, dump_path)

            return [scores_dict["test_" + key] for key in self.metrics]

    def test(self, X_test, y_test, feature_names=None, category_indicator=None):
        """Evaluate the model and return the test score.

        Args:
            X_test (array): array of training vector.
            y_test (array): array of target vector.
            feature_names (list): list of names of the features in X_test
            category_indicator (list): list of boolean indicating whether a feature is a categorical variable.

        Returns:
            list: list of metric scores evaluated on the testing data.
        """
        if not self.use_test_set:
            raise AttributeError(
                "method test is not defined when use_test_set is False"
            )
        else:
            scores = []
            for metric in self.metrics:
                scorer = check_scoring(self.estimator, metric)
                score = scorer(self.estimator, X_test, y_test)
                scores.append(score)
            return scores
