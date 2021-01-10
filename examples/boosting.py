"""
mlgauge example to compare different boosting models on a classification task.
"""
from mlgauge import Analysis, SklearnMethod
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt


SEED = 42


methods = [
    (
        "xgboost",
        SklearnMethod(XGBClassifier(n_jobs=-1, verbose=0), ["accuracy", "f1_micro"]),
    ),
    (
        "lightgbm",
        SklearnMethod(LGBMClassifier(n_jobs=-1, verbose=0), ["accuracy", "f1_micro"]),
    ),
    (
        "catboost",
        SklearnMethod(
            CatBoostClassifier(thread_count=-1, verbose=0), ["accuracy", "f1_micro"]
        ),
    ),
    (
        "gbm",
        SklearnMethod(GradientBoostingClassifier(verbose=0), ["accuracy", "f1_micro"]),
    ),
]


an = Analysis(
    methods=methods,
    metric_names=["accuracy", "f1 score"],
    datasets="classification",
    n_datasets=10,
    random_state=SEED,
    # use_test_set=False   # to use cross-validation
)
an.run()

print(an.get_result_as_df("f1 score"))
an.plot_results("f1 score")
plt.show()
