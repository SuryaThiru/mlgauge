"""
mlgauge example to compare feature importance from different methods.

In this example, we plot the feature importance from lightgbm and xgboost for a few datasets and export the plots in a local output directory. The example is one of using mlgauge for qualitative comparisons.

"""
from mlgauge import Analysis, Method
from xgboost import XGBClassifier, plot_importance as xgbplot
from lightgbm import LGBMClassifier, plot_importance as lgbplot
import matplotlib.pyplot as plt
import pandas as pd
import os


SEED = 42


class GBImportancePlot(Method):
    def __init__(self, gbmestimator):
        super().__init__()

        self.gbmestimator = gbmestimator

        if isinstance(gbmestimator, XGBClassifier):
            self.plot_imp = xgbplot
        elif isinstance(gbmestimator, LGBMClassifier):
            self.plot_imp = lgbplot
        else:
            raise TypeError("gbmestimator must be an XGBClassifier or LGBMClassifier")

    def train(self, X_train, y_train, feature_names, category_indicator=None):
        # passing dataframes as input to xgboost/lightgbm lets the plotting function automatically add tick labels
        X_train = pd.DataFrame(X_train, columns=feature_names)
        self.gbmestimator.fit(X_train, y_train)

        fig, ax = plt.subplots()
        self.plot_imp(
            self.gbmestimator,
            ax=ax,
            xlabel="Feature importance",
            importance_type="gain",
        )
        # self.output_dir is made available through the Analysis class
        fig.savefig(
            os.path.join(self.output_dir, "importance.png"), bbox_inches="tight"
        )
        plt.close(fig)


methods = [
    ("xgb", GBImportancePlot(XGBClassifier(n_jobs=-1, verbosity=0))),
    ("lgb", GBImportancePlot(LGBMClassifier(n_jobs=-1, silent=True))),
]

an = Analysis(
    methods=methods,
    datasets="classification",
    n_datasets=3,
    random_state=SEED,
    use_test_set=False,  # our method does not implement testing
    output_dir="importance_output",
)

an.run()
