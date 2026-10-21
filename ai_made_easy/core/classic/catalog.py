"""Classic machine learning: scikit-learn estimators and transformers, plus
XGBoost, LightGBM and CatBoost through their scikit-learn APIs.

A classic project is a pipeline, not a tensor graph:
dataset → preprocessing → feature engineering → one estimator → metrics.
Each entry renders its constructor from resolved params; ``task`` decides
which metrics and evaluation code apply.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ai_made_easy.core.palette import family_color
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import BlockDefinition, P

CLASSIFICATION, REGRESSION, CLUSTERING, ANOMALY = (
    "classification", "regression", "clustering", "anomaly")


@dataclass(frozen=True)
class Estimator:
    type_id: str
    name: str
    cls: str                 # class name
    module: str              # import module
    task: str
    params: tuple = ()
    desc: str = ""
    extra: dict = field(default_factory=dict)   # fixed constructor kwargs (repr)
    checks: Callable | None = None
    package: str = "scikit-learn"
    category: str = ""
    proba: bool = True       # supports predict_proba
    balanced: bool = False   # supports class_weight="balanced"
    nonnegative: bool = False  # requires non-negative features
    seeded: bool = True      # accepts random_state

    def constructor(self, params: dict, seed: int | None = None,
                    extra: dict | None = None) -> str:
        args = []
        for spec in self.params:
            value = params.get(spec.name, spec.default)
            if spec.type == "str" and spec.name in _NONE_ON_EMPTY and not str(value).strip():
                continue
            if spec.type == "int" and spec.name in _NONE_ON_ZERO and int(value) == 0:
                continue
            if spec.name == "hidden_layer_sizes":
                sizes = tuple(int(v) for v in str(value).split(",") if v.strip())
                args.append(f"hidden_layer_sizes={sizes!r}")
                continue
            args.append(f"{spec.name}={_literal(spec, value)}")
        for k, v in {**self.extra, **(extra or {})}.items():
            args.append(f"{k}={v}")
        if self.seeded and seed is not None:
            args.append(f"{self._seed_kw}={seed}")
        return f"{self.cls}({', '.join(args)})"

    @property
    def _seed_kw(self) -> str:
        return "random_seed" if self.package == "catboost" else "random_state"


_NONE_ON_EMPTY = {"gamma_str", "max_features_str"}
_NONE_ON_ZERO = {"max_depth", "max_leaf_nodes"}


def _literal(spec, value) -> str:
    if spec.type == "enum":
        if value in ("None", "True", "False"):
            return value
        try:
            float(value)
            return str(value)
        except ValueError:
            return repr(value)
    if spec.type == "bool":
        return "True" if value else "False"
    if spec.type == "str":
        return repr(value)
    return repr(value)


def _range(lo, hi):
    def fn(p):
        if float(p[lo]) > float(p[hi]):
            return [("error", f"{lo} must not exceed {hi}")]
        return []
    return fn


# --------------------------------------------------------------- shared params
C = P("C", "float", 1.0, lo=1e-8, help="Inverse regularization strength")
ALPHA = lambda d=1.0: P("alpha", "float", d, lo=0.0, help="Regularization strength")  # noqa: E731
N_EST = lambda d=100: P("n_estimators", "int", d, lo=1)  # noqa: E731
MAX_DEPTH = lambda d=0: P("max_depth", "int", d, lo=0, help="0 = unlimited")  # noqa: E731
LR = lambda d=0.1: P("learning_rate", "float", d, lo=1e-6, hi=10.0)  # noqa: E731
MIN_SPLIT = P("min_samples_split", "int", 2, lo=2)
MIN_LEAF = P("min_samples_leaf", "int", 1, lo=1)
MAX_FEAT = P("max_features", "enum", "sqrt", options=("sqrt", "log2", "None", "1.0"))
N_JOBS = {"n_jobs": "-1"}
KERNEL = P("kernel", "enum", "rbf", options=("rbf", "linear", "poly", "sigmoid"))
GAMMA = P("gamma", "enum", "scale", options=("scale", "auto"))
HIDDEN = P("hidden_layer_sizes", "str", "100", help="Comma-separated layer widths")
MAX_ITER = lambda d=1000: P("max_iter", "int", d, lo=1)  # noqa: E731
N_NEIGH = P("n_neighbors", "int", 5, lo=1)
WEIGHTS = P("weights", "enum", "uniform", options=("uniform", "distance"))
N_CLUST = P("n_clusters", "int", 8, lo=2)


def _mlp(cls: str, task: str, tid: str, name: str) -> Estimator:
    return Estimator(
        tid, name, cls, "sklearn.neural_network", task,
        (HIDDEN, P("activation", "enum", "relu", options=("relu", "tanh", "logistic", "identity")),
         P("alpha", "float", 1e-4, lo=0.0), P("learning_rate_init", "float", 1e-3, lo=1e-8),
         MAX_ITER(300), P("early_stopping", "bool", False)),
        desc=f"Multi-layer perceptron ({'classifier' if task == CLASSIFICATION else 'regressor'}).",
        category="Neural (sklearn)")


ESTIMATORS: list[Estimator] = [
    # ------------------------------------------------------- linear models
    Estimator("ml.logistic_regression", "Logistic Regression", "LogisticRegression",
              "sklearn.linear_model", CLASSIFICATION,
              (C, P("penalty", "enum", "l2", options=("l2", "l1", "elasticnet", "None")),
               P("solver", "enum", "lbfgs", options=("lbfgs", "saga", "liblinear", "newton-cg")),
               P("l1_ratio", "float", 0.5, lo=0.0, hi=1.0, help="Only for elasticnet"),
               MAX_ITER()),
              desc="Linear classifier with a logistic link.", category="Linear Models",
              balanced=True,
              checks=lambda p: (
                  [("error", f"solver '{p['solver']}' does not support the "
                             f"{p['penalty']} penalty")]
                  if (p["penalty"] == "l1" and p["solver"] not in ("saga", "liblinear"))
                  or (p["penalty"] == "elasticnet" and p["solver"] != "saga") else [])),
    Estimator("ml.ridge_classifier", "Ridge Classifier", "RidgeClassifier",
              "sklearn.linear_model", CLASSIFICATION, (ALPHA(),),
              desc="Least-squares classifier with L2 regularization.", category="Linear Models",
              proba=False, balanced=True),
    Estimator("ml.sgd_classifier", "SGD Classifier", "SGDClassifier", "sklearn.linear_model",
              CLASSIFICATION,
              (P("loss", "enum", "hinge", options=("hinge", "log_loss", "modified_huber",
                                                   "squared_hinge", "perceptron")),
               ALPHA(1e-4), P("penalty", "enum", "l2", options=("l2", "l1", "elasticnet")),
               MAX_ITER()),
              desc="Linear model trained with stochastic gradient descent.",
              category="Linear Models", proba=False, balanced=True),
    Estimator("ml.perceptron", "Perceptron", "Perceptron", "sklearn.linear_model",
              CLASSIFICATION, (ALPHA(1e-4), MAX_ITER()), desc="Rosenblatt perceptron.",
              category="Linear Models", proba=False, balanced=True),
    Estimator("ml.linear_regression", "Linear Regression", "LinearRegression",
              "sklearn.linear_model", REGRESSION, (P("fit_intercept", "bool", True),),
              desc="Ordinary least squares.", category="Linear Models", seeded=False),
    Estimator("ml.ridge", "Ridge", "Ridge", "sklearn.linear_model", REGRESSION, (ALPHA(),),
              desc="Least squares with L2 regularization.", category="Linear Models"),
    Estimator("ml.lasso", "Lasso", "Lasso", "sklearn.linear_model", REGRESSION,
              (ALPHA(), MAX_ITER()), desc="Least squares with L1 regularization (sparse).",
              category="Linear Models"),
    Estimator("ml.elastic_net", "Elastic Net", "ElasticNet", "sklearn.linear_model",
              REGRESSION, (ALPHA(), P("l1_ratio", "float", 0.5, lo=0.0, hi=1.0), MAX_ITER()),
              desc="Combined L1 + L2 regularization.", category="Linear Models"),
    Estimator("ml.bayesian_ridge", "Bayesian Ridge", "BayesianRidge", "sklearn.linear_model",
              REGRESSION, (MAX_ITER(300),), desc="Bayesian linear regression.",
              category="Linear Models", seeded=False),
    Estimator("ml.huber", "Huber Regressor", "HuberRegressor", "sklearn.linear_model",
              REGRESSION, (P("epsilon", "float", 1.35, lo=1.0), ALPHA(1e-4), MAX_ITER(100)),
              desc="Linear regression robust to outliers.", category="Linear Models",
              seeded=False),
    Estimator("ml.sgd_regressor", "SGD Regressor", "SGDRegressor", "sklearn.linear_model",
              REGRESSION, (P("loss", "enum", "squared_error",
                             options=("squared_error", "huber", "epsilon_insensitive")),
                           ALPHA(1e-4), MAX_ITER()),
              desc="Linear regression trained with SGD.", category="Linear Models"),
    Estimator("ml.poisson", "Poisson Regressor", "PoissonRegressor", "sklearn.linear_model",
              REGRESSION, (ALPHA(), MAX_ITER(100)), desc="GLM for count targets.",
              category="Linear Models", seeded=False),
    # ------------------------------------------------------- support vectors
    Estimator("ml.svc", "SVC", "SVC", "sklearn.svm", CLASSIFICATION, (C, KERNEL, GAMMA,
              P("degree", "int", 3, lo=1)), desc="Kernel support vector classifier.",
              category="Support Vector Machines", proba=False, balanced=True),
    Estimator("ml.linear_svc", "Linear SVC", "LinearSVC", "sklearn.svm", CLASSIFICATION,
              (C, MAX_ITER()), desc="Linear support vector classifier (large data).",
              category="Support Vector Machines", proba=False, balanced=True),
    Estimator("ml.svr", "SVR", "SVR", "sklearn.svm", REGRESSION,
              (C, KERNEL, GAMMA, P("epsilon", "float", 0.1, lo=0.0)),
              desc="Kernel support vector regression.", category="Support Vector Machines",
              seeded=False),
    Estimator("ml.linear_svr", "Linear SVR", "LinearSVR", "sklearn.svm", REGRESSION,
              (C, P("epsilon", "float", 0.0, lo=0.0), MAX_ITER()),
              desc="Linear support vector regression.", category="Support Vector Machines"),
    # ------------------------------------------------------- neighbors / bayes
    Estimator("ml.knn_classifier", "k-Nearest Neighbors Classifier", "KNeighborsClassifier",
              "sklearn.neighbors", CLASSIFICATION, (N_NEIGH, WEIGHTS,
              P("metric", "enum", "minkowski", options=("minkowski", "euclidean", "manhattan",
                                                        "cosine"))),
              desc="Vote among the k closest training samples.", category="Neighbors & Bayes",
              seeded=False, extra=N_JOBS),
    Estimator("ml.knn_regressor", "k-Nearest Neighbors Regressor", "KNeighborsRegressor",
              "sklearn.neighbors", REGRESSION, (N_NEIGH, WEIGHTS),
              desc="Average of the k closest training targets.", category="Neighbors & Bayes",
              seeded=False, extra=N_JOBS),
    Estimator("ml.gaussian_nb", "Gaussian Naive Bayes", "GaussianNB", "sklearn.naive_bayes",
              CLASSIFICATION, (P("var_smoothing", "float", 1e-9, lo=0.0),),
              desc="Naive Bayes with Gaussian likelihoods.", category="Neighbors & Bayes",
              seeded=False),
    Estimator("ml.multinomial_nb", "Multinomial Naive Bayes", "MultinomialNB",
              "sklearn.naive_bayes", CLASSIFICATION, (ALPHA(),),
              desc="Naive Bayes for counts / TF-IDF (non-negative features).",
              category="Neighbors & Bayes", seeded=False, nonnegative=True),
    Estimator("ml.bernoulli_nb", "Bernoulli Naive Bayes", "BernoulliNB", "sklearn.naive_bayes",
              CLASSIFICATION, (ALPHA(),), desc="Naive Bayes for binary features.",
              category="Neighbors & Bayes", seeded=False),
    Estimator("ml.lda", "Linear Discriminant Analysis", "LinearDiscriminantAnalysis",
              "sklearn.discriminant_analysis", CLASSIFICATION,
              (P("solver", "enum", "svd", options=("svd", "lsqr", "eigen")),),
              desc="Gaussian classes with a shared covariance.", category="Neighbors & Bayes",
              seeded=False),
    Estimator("ml.qda", "Quadratic Discriminant Analysis", "QuadraticDiscriminantAnalysis",
              "sklearn.discriminant_analysis", CLASSIFICATION,
              (P("reg_param", "float", 0.0, lo=0.0, hi=1.0),),
              desc="Gaussian classes with per-class covariance.", category="Neighbors & Bayes",
              seeded=False),
    # ------------------------------------------------------- trees & ensembles
    Estimator("ml.decision_tree_classifier", "Decision Tree Classifier",
              "DecisionTreeClassifier", "sklearn.tree", CLASSIFICATION,
              (P("criterion", "enum", "gini", options=("gini", "entropy", "log_loss")),
               MAX_DEPTH(), MIN_SPLIT, MIN_LEAF),
              desc="Single decision tree.", category="Trees & Ensembles", balanced=True),
    Estimator("ml.decision_tree_regressor", "Decision Tree Regressor", "DecisionTreeRegressor",
              "sklearn.tree", REGRESSION,
              (P("criterion", "enum", "squared_error",
                 options=("squared_error", "absolute_error", "friedman_mse", "poisson")),
               MAX_DEPTH(), MIN_SPLIT, MIN_LEAF),
              desc="Single regression tree.", category="Trees & Ensembles"),
    Estimator("ml.random_forest_classifier", "Random Forest Classifier",
              "RandomForestClassifier", "sklearn.ensemble", CLASSIFICATION,
              (N_EST(), P("criterion", "enum", "gini", options=("gini", "entropy", "log_loss")),
               MAX_DEPTH(), MIN_SPLIT, MIN_LEAF, MAX_FEAT, P("bootstrap", "bool", True)),
              desc="Bagged decision trees with feature subsampling.",
              category="Trees & Ensembles", balanced=True, extra=N_JOBS),
    Estimator("ml.random_forest_regressor", "Random Forest Regressor", "RandomForestRegressor",
              "sklearn.ensemble", REGRESSION,
              (N_EST(), MAX_DEPTH(), MIN_SPLIT, MIN_LEAF,
               P("max_features", "enum", "1.0", options=("1.0", "sqrt", "log2", "None"))),
              desc="Bagged regression trees.", category="Trees & Ensembles", extra=N_JOBS),
    Estimator("ml.extra_trees_classifier", "Extra Trees Classifier", "ExtraTreesClassifier",
              "sklearn.ensemble", CLASSIFICATION, (N_EST(), MAX_DEPTH(), MIN_SPLIT, MIN_LEAF,
                                                   MAX_FEAT),
              desc="Extremely randomized trees.", category="Trees & Ensembles",
              balanced=True, extra=N_JOBS),
    Estimator("ml.extra_trees_regressor", "Extra Trees Regressor", "ExtraTreesRegressor",
              "sklearn.ensemble", REGRESSION, (N_EST(), MAX_DEPTH(), MIN_SPLIT, MIN_LEAF),
              desc="Extremely randomized regression trees.", category="Trees & Ensembles",
              extra=N_JOBS),
    Estimator("ml.gradient_boosting_classifier", "Gradient Boosting Classifier",
              "GradientBoostingClassifier", "sklearn.ensemble", CLASSIFICATION,
              (N_EST(), LR(), P("max_depth", "int", 3, lo=1), P("subsample", "float", 1.0,
                                                               lo=0.01, hi=1.0)),
              desc="Additive boosted trees (exact).", category="Trees & Ensembles"),
    Estimator("ml.gradient_boosting_regressor", "Gradient Boosting Regressor",
              "GradientBoostingRegressor", "sklearn.ensemble", REGRESSION,
              (N_EST(), LR(), P("max_depth", "int", 3, lo=1),
               P("loss", "enum", "squared_error",
                 options=("squared_error", "absolute_error", "huber", "quantile"))),
              desc="Additive boosted regression trees.", category="Trees & Ensembles"),
    Estimator("ml.hist_gradient_boosting_classifier", "Histogram Gradient Boosting Classifier",
              "HistGradientBoostingClassifier", "sklearn.ensemble", CLASSIFICATION,
              (LR(), P("max_iter", "int", 100, lo=1), MAX_DEPTH(),
               P("max_leaf_nodes", "int", 31, lo=2), P("l2_regularization", "float", 0.0, lo=0.0)),
              desc="Fast histogram-based boosting (handles missing values).",
              category="Trees & Ensembles", balanced=True),
    Estimator("ml.hist_gradient_boosting_regressor", "Histogram Gradient Boosting Regressor",
              "HistGradientBoostingRegressor", "sklearn.ensemble", REGRESSION,
              (LR(), P("max_iter", "int", 100, lo=1), MAX_DEPTH(),
               P("max_leaf_nodes", "int", 31, lo=2), P("l2_regularization", "float", 0.0, lo=0.0)),
              desc="Fast histogram-based boosting regressor.", category="Trees & Ensembles"),
    Estimator("ml.adaboost_classifier", "AdaBoost Classifier", "AdaBoostClassifier",
              "sklearn.ensemble", CLASSIFICATION, (N_EST(50), LR(1.0)),
              desc="Adaptive boosting of decision stumps.", category="Trees & Ensembles"),
    Estimator("ml.adaboost_regressor", "AdaBoost Regressor", "AdaBoostRegressor",
              "sklearn.ensemble", REGRESSION, (N_EST(50), LR(1.0)),
              desc="Adaptive boosting for regression.", category="Trees & Ensembles"),
    Estimator("ml.bagging_classifier", "Bagging Classifier", "BaggingClassifier",
              "sklearn.ensemble", CLASSIFICATION,
              (N_EST(10), P("max_samples", "float", 1.0, lo=0.01, hi=1.0)),
              desc="Bootstrap-aggregated decision trees.", category="Trees & Ensembles",
              extra=N_JOBS),
    # ------------------------------------------------------- gradient boosting libs
    Estimator("ml.xgb_classifier", "XGBoost Classifier", "XGBClassifier", "xgboost",
              CLASSIFICATION,
              (N_EST(300), LR(0.1), P("max_depth", "int", 6, lo=1),
               P("subsample", "float", 1.0, lo=0.01, hi=1.0),
               P("colsample_bytree", "float", 1.0, lo=0.01, hi=1.0),
               P("reg_lambda", "float", 1.0, lo=0.0), P("reg_alpha", "float", 0.0, lo=0.0)),
              desc="XGBoost gradient-boosted trees.", category="Gradient Boosting",
              package="xgboost", extra=N_JOBS),
    Estimator("ml.xgb_regressor", "XGBoost Regressor", "XGBRegressor", "xgboost", REGRESSION,
              (N_EST(300), LR(0.1), P("max_depth", "int", 6, lo=1),
               P("subsample", "float", 1.0, lo=0.01, hi=1.0),
               P("colsample_bytree", "float", 1.0, lo=0.01, hi=1.0),
               P("reg_lambda", "float", 1.0, lo=0.0)),
              desc="XGBoost gradient-boosted regression trees.", category="Gradient Boosting",
              package="xgboost", extra=N_JOBS),
    Estimator("ml.lgbm_classifier", "LightGBM Classifier", "LGBMClassifier", "lightgbm",
              CLASSIFICATION,
              (N_EST(300), LR(0.1), P("num_leaves", "int", 31, lo=2),
               P("max_depth", "int", -1, lo=-1, help="-1 = unlimited"),
               P("subsample", "float", 1.0, lo=0.01, hi=1.0),
               P("colsample_bytree", "float", 1.0, lo=0.01, hi=1.0)),
              desc="LightGBM leaf-wise gradient boosting.", category="Gradient Boosting",
              package="lightgbm", balanced=True, extra={"verbose": "-1", **N_JOBS}),
    Estimator("ml.lgbm_regressor", "LightGBM Regressor", "LGBMRegressor", "lightgbm",
              REGRESSION,
              (N_EST(300), LR(0.1), P("num_leaves", "int", 31, lo=2),
               P("max_depth", "int", -1, lo=-1, help="-1 = unlimited")),
              desc="LightGBM gradient-boosted regression.", category="Gradient Boosting",
              package="lightgbm", extra={"verbose": "-1", **N_JOBS}),
    Estimator("ml.catboost_classifier", "CatBoost Classifier", "CatBoostClassifier",
              "catboost", CLASSIFICATION,
              (P("iterations", "int", 500, lo=1), LR(0.05), P("depth", "int", 6, lo=1, hi=16),
               P("l2_leaf_reg", "float", 3.0, lo=0.0)),
              desc="CatBoost ordered boosting.", category="Gradient Boosting",
              package="catboost", extra={"verbose": "0"}),
    Estimator("ml.catboost_regressor", "CatBoost Regressor", "CatBoostRegressor", "catboost",
              REGRESSION,
              (P("iterations", "int", 500, lo=1), LR(0.05), P("depth", "int", 6, lo=1, hi=16),
               P("l2_leaf_reg", "float", 3.0, lo=0.0)),
              desc="CatBoost ordered boosting regressor.", category="Gradient Boosting",
              package="catboost", extra={"verbose": "0"}),
    _mlp("MLPClassifier", CLASSIFICATION, "ml.mlp_classifier", "MLP Classifier"),
    _mlp("MLPRegressor", REGRESSION, "ml.mlp_regressor", "MLP Regressor"),
    # ------------------------------------------------------- clustering
    Estimator("ml.kmeans", "k-Means", "KMeans", "sklearn.cluster", CLUSTERING,
              (N_CLUST, P("n_init", "int", 10, lo=1), MAX_ITER(300)),
              desc="Partition samples into k clusters around centroids.", category="Clustering"),
    Estimator("ml.minibatch_kmeans", "Mini-Batch k-Means", "MiniBatchKMeans", "sklearn.cluster",
              CLUSTERING, (N_CLUST, P("batch_size", "int", 1024, lo=1)),
              desc="k-Means on mini-batches (large data).", category="Clustering"),
    Estimator("ml.dbscan", "DBSCAN", "DBSCAN", "sklearn.cluster", CLUSTERING,
              (P("eps", "float", 0.5, lo=1e-9), P("min_samples", "int", 5, lo=1)),
              desc="Density-based clustering (finds noise points).", category="Clustering",
              seeded=False),
    Estimator("ml.hdbscan", "HDBSCAN", "HDBSCAN", "sklearn.cluster", CLUSTERING,
              (P("min_cluster_size", "int", 5, lo=2),),
              desc="Hierarchical density-based clustering.", category="Clustering", seeded=False),
    Estimator("ml.agglomerative", "Agglomerative Clustering", "AgglomerativeClustering",
              "sklearn.cluster", CLUSTERING,
              (N_CLUST, P("linkage", "enum", "ward", options=("ward", "complete", "average",
                                                              "single"))),
              desc="Bottom-up hierarchical clustering.", category="Clustering", seeded=False),
    Estimator("ml.gaussian_mixture", "Gaussian Mixture", "GaussianMixture", "sklearn.mixture",
              CLUSTERING,
              (P("n_components", "int", 3, lo=1),
               P("covariance_type", "enum", "full", options=("full", "tied", "diag", "spherical"))),
              desc="Mixture of Gaussians fitted with EM.", category="Clustering"),
    Estimator("ml.birch", "BIRCH", "Birch", "sklearn.cluster", CLUSTERING,
              (N_CLUST, P("threshold", "float", 0.5, lo=1e-9)),
              desc="Incremental tree-based clustering.", category="Clustering", seeded=False),
    Estimator("ml.spectral", "Spectral Clustering", "SpectralClustering", "sklearn.cluster",
              CLUSTERING, (N_CLUST,), desc="Graph-Laplacian clustering.", category="Clustering"),
    # ------------------------------------------------------- anomaly detection
    Estimator("ml.isolation_forest", "Isolation Forest", "IsolationForest", "sklearn.ensemble",
              ANOMALY, (N_EST(), P("contamination", "enum", "auto",
                                   options=("auto", "0.01", "0.05", "0.1", "0.2"))),
              desc="Isolate anomalies with random partitioning trees.",
              category="Anomaly Detection", extra=N_JOBS),
    Estimator("ml.one_class_svm", "One-Class SVM", "OneClassSVM", "sklearn.svm", ANOMALY,
              (KERNEL, GAMMA, P("nu", "float", 0.5, lo=1e-6, hi=1.0)),
              desc="Boundary of the normal data in kernel space.",
              category="Anomaly Detection", seeded=False),
    Estimator("ml.local_outlier_factor", "Local Outlier Factor", "LocalOutlierFactor",
              "sklearn.neighbors", ANOMALY,
              (P("n_neighbors", "int", 20, lo=1),
               P("contamination", "enum", "auto", options=("auto", "0.01", "0.05", "0.1", "0.2"))),
              desc="Density deviation relative to neighbors.", category="Anomaly Detection",
              seeded=False, extra={"novelty": "True"}),
]


# ------------------------------------------------------------ transformers

@dataclass(frozen=True)
class Transformer:
    type_id: str
    name: str
    cls: str
    module: str
    params: tuple = ()
    desc: str = ""
    render: Callable | None = None   # custom constructor
    stage: str = "features"          # features | text
    checks: Callable | None = None
    seeded: bool = False
    package: str = ""                # pip package outside scikit-learn ("" = none)
    extra: str = ""                  # the ai-made-easy extra that installs it


def _smooth(value) -> str:
    text = str(value).strip()
    return '"auto"' if text in ("", "auto") else str(float(text))


def _smooth_check(p) -> list:
    try:
        _smooth(p.get("smooth", "auto"))
    except ValueError:
        return [("error", "smooth must be auto or a number")]
    return []


def _ctor(cls: str, **kwargs) -> str:
    return f"{cls}({', '.join(f'{k}={v}' for k, v in kwargs.items())})"


TRANSFORMERS: list[Transformer] = [
    Transformer("ml.polynomial", "Polynomial Features", "PolynomialFeatures",
                "sklearn.preprocessing",
                (P("degree", "int", 2, lo=1, hi=5), P("interaction_only", "bool", False),
                 P("include_bias", "bool", False)),
                desc="Add polynomial and interaction terms."),
    Transformer("ml.spline", "Spline Features", "SplineTransformer", "sklearn.preprocessing",
                (P("n_knots", "int", 5, lo=2), P("degree", "int", 3, lo=1)),
                desc="Piecewise-polynomial basis per feature."),
    Transformer("ml.power_transform", "Power Transform", "PowerTransformer",
                "sklearn.preprocessing",
                (P("method", "enum", "yeo-johnson", options=("yeo-johnson", "box-cox")),),
                desc="Make features more Gaussian (Yeo-Johnson / Box-Cox)."),
    Transformer("ml.quantile_transform", "Quantile Transform", "QuantileTransformer",
                "sklearn.preprocessing",
                (P("n_quantiles", "int", 1000, lo=10),
                 P("output_distribution", "enum", "uniform", options=("uniform", "normal"))),
                desc="Map features to a uniform or normal distribution.", seeded=True),
    Transformer("ml.kbins", "Discretize (k bins)", "KBinsDiscretizer", "sklearn.preprocessing",
                (P("n_bins", "int", 5, lo=2),
                 P("strategy", "enum", "quantile", options=("quantile", "uniform", "kmeans"))),
                desc="Bin continuous features into intervals.",
                render=lambda p: _ctor("KBinsDiscretizer", n_bins=p["n_bins"],
                                       encode='"onehot-dense"', strategy=repr(p["strategy"]))),
    Transformer("ml.select_k_best", "Select K Best", "SelectKBest", "sklearn.feature_selection",
                (P("k", "int", 10, lo=1),
                 P("score_func", "enum", "f_classif",
                   options=("f_classif", "f_regression", "mutual_info_classif",
                            "mutual_info_regression", "chi2"))),
                desc="Keep the k features with the best univariate score.",
                render=lambda p: f"SelectKBest(score_func={p['score_func']}, k={p['k']})"),
    Transformer("ml.pca", "PCA", "PCA", "sklearn.decomposition",
                (P("n_components", "float", 0.95, lo=0.0,
                   help="Integer count, or a fraction of variance to keep (0-1)"),
                 P("whiten", "bool", False)),
                desc="Principal component analysis.", seeded=True,
                render=lambda p: _ctor("PCA", n_components=(
                    int(p["n_components"]) if float(p["n_components"]) >= 1
                    else float(p["n_components"])), whiten=p["whiten"])),
    Transformer("ml.truncated_svd", "Truncated SVD", "TruncatedSVD", "sklearn.decomposition",
                (P("n_components", "int", 100, lo=1),),
                desc="Linear dimensionality reduction for sparse data (LSA).", seeded=True),
    Transformer("ml.kernel_pca", "Kernel PCA", "KernelPCA", "sklearn.decomposition",
                (P("n_components", "int", 10, lo=1), KERNEL),
                desc="Non-linear PCA via kernels.", seeded=True),
    Transformer("ml.fast_ica", "Independent Components", "FastICA", "sklearn.decomposition",
                (P("n_components", "int", 10, lo=1),),
                desc="Separate statistically independent sources.", seeded=True),
    Transformer("ml.isomap", "Isomap", "Isomap", "sklearn.manifold",
                (P("n_components", "int", 2, lo=1), P("n_neighbors", "int", 5, lo=1)),
                desc="Non-linear embedding that keeps geodesic distances along the data's "
                     "manifold."),
    Transformer("ml.nystroem", "Kernel Features (Nystroem)", "Nystroem",
                "sklearn.kernel_approximation",
                (P("kernel", "enum", "rbf", options=("rbf", "poly", "sigmoid", "laplacian")),
                 P("n_components", "int", 100, lo=1, help="Features to build")),
                desc="Approximate kernel features: a linear model on them acts like a kernel "
                     "SVM, but scales to large tables.", seeded=True),
    Transformer("ml.umap", "UMAP", "UMAP", "umap",
                (P("n_components", "int", 2, lo=1), P("n_neighbors", "int", 15, lo=2),
                 P("min_dist", "float", 0.1, lo=0.0, hi=1.0)),
                desc="Uniform manifold approximation: a fast non-linear embedding that keeps "
                     "local and much of the global structure. Good before clustering.",
                seeded=True, package="umap-learn", extra="umap"),
    Transformer("ml.target_encode", "Target Encoding", "TargetEncoder", "sklearn.preprocessing",
                (P("smooth", "str", "auto", help="Shrinkage toward the overall mean: auto or "
                                                 "a number"),
                 P("cv", "int", 5, lo=2, hi=20, help="Folds used to encode the training rows")),
                desc="Replaces each category with the average target of its rows, learned "
                     "with cross-fitting so a row never sees its own label. Good for columns "
                     "with many categories, where one-hot would explode.",
                render=lambda p: (f"TargetEncoder(smooth={_smooth(p['smooth'])}, "
                                  f"cv={int(p['cv'])})"),
                stage="categorical", seeded=True, checks=lambda p: _smooth_check(p)),
    Transformer("ml.smote", "SMOTE Oversampling", "SMOTE", "imblearn.over_sampling",
                (P("k_neighbors", "int", 5, lo=1, help="Neighbours used to make new rows"),
                 P("sampling_strategy", "enum", "auto",
                   options=("auto", "minority", "not majority", "all"))),
                desc="Synthesizes new rows of the rarer classes between existing ones "
                     "(Chawla et al., 2002). Applied to the training rows only, while fitting "
                     "(also inside cross-validation), never to the test set.",
                stage="sampler", seeded=True, package="imbalanced-learn", extra="imbalance"),
    Transformer("ml.tfidf", "TF-IDF Vectorizer", "TfidfVectorizer",
                "sklearn.feature_extraction.text",
                (P("max_features", "int", 20000, lo=1), P("ngram_max", "int", 2, lo=1, hi=5),
                 P("min_df", "int", 1, lo=1), P("sublinear_tf", "bool", True),
                 P("analyzer", "enum", "word", options=("word", "char_wb"))),
                desc="Term frequency–inverse document frequency features for text.",
                stage="text",
                render=lambda p: _ctor("TfidfVectorizer", max_features=p["max_features"],
                                       ngram_range=f"(1, {p['ngram_max']})", min_df=p["min_df"],
                                       sublinear_tf=p["sublinear_tf"],
                                       analyzer=repr(p["analyzer"]),
                                       preprocessor="clean_text")),
    Transformer("ml.count_vectorizer", "Count Vectorizer", "CountVectorizer",
                "sklearn.feature_extraction.text",
                (P("max_features", "int", 20000, lo=1), P("ngram_max", "int", 1, lo=1, hi=5),
                 P("binary", "bool", False)),
                desc="Bag-of-words token counts for text.", stage="text",
                render=lambda p: _ctor("CountVectorizer", max_features=p["max_features"],
                                       ngram_range=f"(1, {p['ngram_max']})", binary=p["binary"],
                                       preprocessor="clean_text")),
]

# 3.5: the rest of scikit-learn's estimators and transformers that fit a pipeline
_LIN, _TREES, _NB, _SVM = ("Linear Models", "Trees & Ensembles", "Neighbors & Bayes",
                           "Support Vector Machines")
_CV = P("cv", "int", 5, lo=2, hi=20, help="Cross-validation folds for the built-in search")
ESTIMATORS += [
    # ---------------------------------------------------------------- classification
    Estimator("ml.nu_svc", "Nu-SVC", "NuSVC", "sklearn.svm", CLASSIFICATION,
              (P("nu", "float", 0.5, lo=0.01, hi=1.0, help="Upper bound on the fraction of "
                                                           "margin errors"), KERNEL, GAMMA),
              desc="Kernel SVM parameterized by ν, the share of support vectors, instead of C.",
              category=_SVM, balanced=True, extra={"probability": "True"}),
    Estimator("ml.logistic_regression_cv", "Logistic Regression (CV)", "LogisticRegressionCV",
              "sklearn.linear_model", CLASSIFICATION, (P("Cs", "int", 10, lo=1, help="Values "
                                                         "of C tried"), _CV, MAX_ITER(1000)),
              desc="Logistic regression that picks its regularization strength by "
                   "cross-validation.", category=_LIN, balanced=True),
    Estimator("ml.ridge_classifier_cv", "Ridge Classifier (CV)", "RidgeClassifierCV",
              "sklearn.linear_model", CLASSIFICATION, (),
              desc="Ridge classifier with the regularization strength chosen by efficient "
                   "leave-one-out cross-validation.", category=_LIN, proba=False,
              balanced=True, seeded=False),
    Estimator("ml.passive_aggressive_classifier", "Passive-Aggressive Classifier",
              "PassiveAggressiveClassifier", "sklearn.linear_model", CLASSIFICATION,
              (C, MAX_ITER(1000)),
              desc="Online large-margin linear classifier: updates only on mistakes.",
              category=_LIN, proba=False, balanced=True),
    Estimator("ml.nearest_centroid", "Nearest Centroid", "NearestCentroid",
              "sklearn.neighbors", CLASSIFICATION, (),
              desc="Each class is the mean of its examples; a new example takes the nearest "
                   "one. A strong, simple baseline.", category=_NB, proba=False, seeded=False),
    Estimator("ml.complement_nb", "Complement Naive Bayes", "ComplementNB",
              "sklearn.naive_bayes", CLASSIFICATION, (ALPHA(1.0),),
              desc="Naive Bayes for imbalanced count features (text): learns from the "
                   "complement of each class.", category=_NB, nonnegative=True, seeded=False),
    Estimator("ml.extra_tree_classifier", "Extra Tree Classifier", "ExtraTreeClassifier",
              "sklearn.tree", CLASSIFICATION, (MAX_DEPTH(), MIN_SPLIT, MIN_LEAF),
              desc="One extremely randomized tree (splits drawn at random).", category=_TREES,
              balanced=True),
    Estimator("ml.dummy_classifier", "Baseline (Dummy) Classifier", "DummyClassifier",
              "sklearn.dummy", CLASSIFICATION,
              (P("strategy", "enum", "prior",
                 options=("prior", "most_frequent", "stratified", "uniform")),),
              desc="Ignores the features: the score any real model must beat.",
              category=_LIN),
    Estimator("ml.calibrated_classifier", "Calibrated Linear SVM", "CalibratedClassifierCV",
              "sklearn.calibration", CLASSIFICATION,
              (P("method", "enum", "sigmoid", options=("sigmoid", "isotonic")), _CV),
              desc="A linear SVM with probabilities calibrated by cross-validation "
                   "(Platt scaling or isotonic): trustworthy confidence scores.",
              category=_SVM, seeded=False),
    # ---------------------------------------------------------------- regression
    Estimator("ml.nu_svr", "Nu-SVR", "NuSVR", "sklearn.svm", REGRESSION,
              (P("nu", "float", 0.5, lo=0.01, hi=1.0), C, KERNEL, GAMMA),
              desc="Kernel support vector regression parameterized by ν.", category=_SVM,
              seeded=False),
    Estimator("ml.kernel_ridge", "Kernel Ridge", "KernelRidge", "sklearn.kernel_ridge",
              REGRESSION, (ALPHA(1.0), P("kernel", "enum", "rbf",
                                         options=("rbf", "linear", "poly", "laplacian"))),
              desc="Ridge regression with the kernel trick: smooth non-linear fits on small "
                   "tables.", category=_SVM, seeded=False),
    Estimator("ml.ridge_cv", "Ridge (CV)", "RidgeCV", "sklearn.linear_model", REGRESSION, (),
              desc="Ridge with its regularization strength chosen by leave-one-out CV.",
              category=_LIN, seeded=False),
    Estimator("ml.lasso_cv", "Lasso (CV)", "LassoCV", "sklearn.linear_model", REGRESSION,
              (_CV, MAX_ITER(1000)), desc="Lasso with the strength chosen along a path by "
                                         "cross-validation.", category=_LIN),
    Estimator("ml.elastic_net_cv", "Elastic Net (CV)", "ElasticNetCV", "sklearn.linear_model",
              REGRESSION, (P("l1_ratio", "float", 0.5, lo=0.0, hi=1.0), _CV, MAX_ITER(1000)),
              desc="Elastic net with the strength chosen by cross-validation.", category=_LIN),
    Estimator("ml.lars", "Least-Angle Regression", "Lars", "sklearn.linear_model", REGRESSION,
              (P("n_nonzero_coefs", "int", 500, lo=1),),
              desc="Least-angle regression (Efron et al., 2004): adds features one at a "
                   "time, for tables with many features.", category=_LIN),
    Estimator("ml.lasso_lars", "Lasso (LARS)", "LassoLars", "sklearn.linear_model",
              REGRESSION, (ALPHA(0.01),), desc="Lasso fitted with least-angle regression.",
              category=_LIN),
    Estimator("ml.lasso_lars_ic", "Lasso (information criterion)", "LassoLarsIC",
              "sklearn.linear_model", REGRESSION,
              (P("criterion", "enum", "aic", options=("aic", "bic")),),
              desc="Lasso whose strength minimizes AIC or BIC: no cross-validation needed.",
              category=_LIN, seeded=False),
    Estimator("ml.omp", "Orthogonal Matching Pursuit", "OrthogonalMatchingPursuit",
              "sklearn.linear_model", REGRESSION, (),
              desc="Greedy sparse regression: picks the most correlated feature, refits, "
                   "repeats.", category=_LIN, seeded=False),
    Estimator("ml.ard_regression", "ARD Regression", "ARDRegression", "sklearn.linear_model",
              REGRESSION, (MAX_ITER(300),),
              desc="Bayesian linear regression with one prior per weight (automatic "
                   "relevance determination): prunes useless features.", category=_LIN,
              seeded=False),
    Estimator("ml.theil_sen", "Theil-Sen Regression", "TheilSenRegressor",
              "sklearn.linear_model", REGRESSION, (MAX_ITER(300),),
              desc="Robust linear fit from the median of many small-subset fits: up to ~29% "
                   "outliers barely move it.", category=_LIN),
    Estimator("ml.ransac", "RANSAC Regression", "RANSACRegressor", "sklearn.linear_model",
              REGRESSION, (P("max_trials", "int", 100, lo=1),),
              desc="Fits a line to random subsets and keeps the one most rows agree with: "
                   "ignores gross outliers.", category=_LIN),
    Estimator("ml.quantile_regression", "Quantile Regression", "QuantileRegressor",
              "sklearn.linear_model", REGRESSION,
              (P("quantile", "float", 0.5, lo=0.01, hi=0.99, help="0.5 = the median"),
               ALPHA(0.0)),
              desc="Predicts a quantile (e.g. the median, or the 90th percentile for a "
                   "safety margin) instead of the mean.", category=_LIN, seeded=False,
              extra={"solver": "'highs'"}),
    Estimator("ml.tweedie_regression", "Tweedie Regression", "TweedieRegressor",
              "sklearn.linear_model", REGRESSION,
              (P("power", "float", 0.0, lo=0.0, hi=3.0, help="0 normal, 1 Poisson, 2 gamma; "
                                                             "1-2 compound Poisson-gamma"),
               ALPHA(1.0), MAX_ITER(300)),
              desc="Generalized linear model for insurance claims, rainfall and other "
                   "zero-heavy positive targets.", category=_LIN, seeded=False),
    Estimator("ml.passive_aggressive_regressor", "Passive-Aggressive Regressor",
              "PassiveAggressiveRegressor", "sklearn.linear_model", REGRESSION,
              (C, MAX_ITER(1000)), desc="Online large-margin linear regression.",
              category=_LIN),
    Estimator("ml.extra_tree_regressor", "Extra Tree Regressor", "ExtraTreeRegressor",
              "sklearn.tree", REGRESSION, (MAX_DEPTH(), MIN_SPLIT, MIN_LEAF),
              desc="One extremely randomized regression tree.", category=_TREES),
    Estimator("ml.bagging_regressor", "Bagging Regressor", "BaggingRegressor",
              "sklearn.ensemble", REGRESSION,
              (N_EST(10), P("max_samples", "float", 1.0, lo=0.05, hi=1.0)),
              desc="Averages trees fitted on bootstrap samples.", category=_TREES,
              extra=N_JOBS),
    Estimator("ml.dummy_regressor", "Baseline (Dummy) Regressor", "DummyRegressor",
              "sklearn.dummy", REGRESSION,
              (P("strategy", "enum", "mean", options=("mean", "median")),),
              desc="Predicts the training mean (or median): the score any real model must "
                   "beat.", category=_LIN, seeded=False),
    Estimator("ml.pls_regression", "PLS Regression", "PLSRegression",
              "sklearn.cross_decomposition", REGRESSION, (P("n_components", "int", 2, lo=1),),
              desc="Partial least squares: regression on a few components chosen to predict "
                   "the target, for many correlated features (spectra, sensors).",
              category=_LIN, seeded=False),
    # ---------------------------------------------------------------- clustering
    Estimator("ml.affinity_propagation", "Affinity Propagation", "AffinityPropagation",
              "sklearn.cluster", CLUSTERING, (P("damping", "float", 0.5, lo=0.5, hi=0.99),),
              desc="Finds the number of clusters itself by passing messages between rows; "
                   "each cluster is represented by a real example.", category="Clustering"),
    Estimator("ml.mean_shift", "Mean Shift", "MeanShift", "sklearn.cluster", CLUSTERING, (),
              desc="Moves each point uphill on the data's density: clusters are the peaks "
                   "(no cluster count needed).", category="Clustering", seeded=False,
              extra=N_JOBS),
    Estimator("ml.optics", "OPTICS", "OPTICS", "sklearn.cluster", CLUSTERING,
              (P("min_samples", "int", 5, lo=2),),
              desc="Density clustering like DBSCAN, but for clusters of different densities.",
              category="Clustering", seeded=False, extra=N_JOBS),
    Estimator("ml.bisecting_kmeans", "Bisecting K-Means", "BisectingKMeans", "sklearn.cluster",
              CLUSTERING, (N_CLUST,),
              desc="Splits the largest cluster in two, repeatedly: a hierarchy of K-Means.",
              category="Clustering"),
    Estimator("ml.bayesian_gaussian_mixture", "Bayesian Gaussian Mixture",
              "BayesianGaussianMixture", "sklearn.mixture", CLUSTERING,
              (P("n_components", "int", 8, lo=1, help="An upper bound: unneeded components "
                                                      "shrink away"),),
              desc="Gaussian mixture that switches off the components it doesn't need "
                   "(variational Dirichlet process).", category="Clustering"),
    # ---------------------------------------------------------------- anomaly detection
    Estimator("ml.elliptic_envelope", "Elliptic Envelope", "EllipticEnvelope",
              "sklearn.covariance", ANOMALY,
              (P("contamination", "float", 0.1, lo=0.001, hi=0.5),),
              desc="Fits a robust Gaussian to the normal rows; points far outside its "
                   "ellipse are anomalies.", category="Anomaly Detection"),
    Estimator("ml.sgd_one_class_svm", "One-Class SVM (SGD)", "SGDOneClassSVM",
              "sklearn.linear_model", ANOMALY, (P("nu", "float", 0.1, lo=0.001, hi=1.0),),
              desc="A linear one-class SVM trained by SGD: scales to large tables.",
              category="Anomaly Detection"),
]

_SCORE = P("score_func", "enum", "f_classif", options=("f_classif", "f_regression", "chi2"))
_N_COMP = lambda d=10: P("n_components", "int", d, lo=1)  # noqa: E731
TRANSFORMERS += [
    Transformer("ml.max_abs_scale", "Max-Abs Scaling", "MaxAbsScaler", "sklearn.preprocessing",
                desc="Divides each feature by its largest absolute value (keeps zeros and "
                     "sparsity)."),
    Transformer("ml.normalizer", "Row Normalizer", "Normalizer", "sklearn.preprocessing",
                (P("norm", "enum", "l2", options=("l2", "l1", "max")),),
                desc="Scales every row to unit length (for cosine-like comparisons)."),
    Transformer("ml.binarizer", "Binarizer", "Binarizer", "sklearn.preprocessing",
                (P("threshold", "float", 0.0, lo=-1e6, hi=1e6),),
                desc="Turns each value into 1 above the threshold, else 0."),
    Transformer("ml.variance_threshold", "Variance Threshold", "VarianceThreshold",
                "sklearn.feature_selection", (P("threshold", "float", 0.0, lo=0.0),),
                desc="Drops features whose variance is at most the threshold (constants)."),
    Transformer("ml.select_percentile", "Select Percentile", "SelectPercentile",
                "sklearn.feature_selection", (P("percentile", "int", 50, lo=1, hi=100), _SCORE),
                desc="Keeps the best-scoring percentage of the features.",
                render=lambda p: (f"SelectPercentile(score_func={p['score_func']}, "
                                  f"percentile={int(p['percentile'])})")),
    Transformer("ml.select_fpr", "Select by False Positive Rate", "SelectFpr",
                "sklearn.feature_selection", (P("alpha", "float", 0.05, lo=1e-6, hi=1.0),
                                              _SCORE),
                desc="Keeps the features whose univariate test p-value is below alpha.",
                render=lambda p: f"SelectFpr(score_func={p['score_func']}, alpha={p['alpha']})"),
    Transformer("ml.select_fdr", "Select by False Discovery Rate", "SelectFdr",
                "sklearn.feature_selection", (P("alpha", "float", 0.05, lo=1e-6, hi=1.0),
                                              _SCORE),
                desc="Keeps features while controlling the expected share of false "
                     "discoveries (Benjamini-Hochberg).",
                render=lambda p: f"SelectFdr(score_func={p['score_func']}, alpha={p['alpha']})"),
    Transformer("ml.select_fwe", "Select by Family-Wise Error", "SelectFwe",
                "sklearn.feature_selection", (P("alpha", "float", 0.05, lo=1e-6, hi=1.0),
                                              _SCORE),
                desc="Keeps features while controlling the chance of any false discovery "
                     "(Bonferroni).",
                render=lambda p: f"SelectFwe(score_func={p['score_func']}, alpha={p['alpha']})"),
    Transformer("ml.incremental_pca", "Incremental PCA", "IncrementalPCA",
                "sklearn.decomposition", (_N_COMP(),),
                desc="PCA fitted in mini-batches: the same components for tables too big for "
                     "memory."),
    Transformer("ml.sparse_pca", "Sparse PCA", "SparsePCA", "sklearn.decomposition",
                (_N_COMP(), P("alpha", "float", 1.0, lo=0.0)), seeded=True,
                desc="Components that each use only a few features: easier to interpret."),
    Transformer("ml.factor_analysis", "Factor Analysis", "FactorAnalysis",
                "sklearn.decomposition", (_N_COMP(),), seeded=True,
                desc="Explains the features by a few hidden factors plus per-feature noise."),
    Transformer("ml.nmf", "Non-Negative Matrix Factorization", "NMF", "sklearn.decomposition",
                (_N_COMP(), MAX_ITER(500)), seeded=True,
                desc="Parts-based components with no negative values (needs non-negative "
                     "features: scale them to 0-1 first)."),
    Transformer("ml.lda_topics", "Topics (Latent Dirichlet Allocation)",
                "LatentDirichletAllocation", "sklearn.decomposition", (_N_COMP(),),
                seeded=True,
                desc="Each row as a mix of topics (word counts or other non-negative "
                     "features)."),
    Transformer("ml.feature_agglomeration", "Feature Agglomeration", "FeatureAgglomeration",
                "sklearn.cluster", (P("n_clusters", "int", 5, lo=1),),
                desc="Merges similar features by hierarchical clustering (averages each "
                     "group)."),
    Transformer("ml.gaussian_projection", "Gaussian Random Projection",
                "GaussianRandomProjection", "sklearn.random_projection", (_N_COMP(),),
                seeded=True,
                desc="A random linear map to fewer features that roughly keeps distances "
                     "(Johnson-Lindenstrauss)."),
    Transformer("ml.sparse_projection", "Sparse Random Projection", "SparseRandomProjection",
                "sklearn.random_projection", (_N_COMP(),), seeded=True,
                desc="A cheaper, sparse random projection."),
    Transformer("ml.rbf_sampler", "Random Fourier Features", "RBFSampler",
                "sklearn.kernel_approximation",
                (P("gamma", "float", 1.0, lo=1e-6), _N_COMP(100)), seeded=True,
                desc="Random features approximating an RBF kernel (Rahimi & Recht, 2007): a "
                     "linear model on them acts like a kernel SVM."),
    Transformer("ml.skewed_chi2_sampler", "Skewed Chi² Features", "SkewedChi2Sampler",
                "sklearn.kernel_approximation",
                (P("skewedness", "float", 1.0, lo=1e-6), _N_COMP(100)), seeded=True,
                desc="Random features for the skewed chi-squared kernel (features above "
                     "−skewedness)."),
    Transformer("ml.additive_chi2_sampler", "Additive Chi² Features", "AdditiveChi2Sampler",
                "sklearn.kernel_approximation", (P("sample_steps", "int", 2, lo=1, hi=3),),
                desc="Features for the additive chi-squared kernel, for histograms "
                     "(non-negative features)."),
    Transformer("ml.polynomial_sketch", "Polynomial Kernel Sketch", "PolynomialCountSketch",
                "sklearn.kernel_approximation",
                (P("degree", "int", 2, lo=1, hi=5), _N_COMP(100)), seeded=True,
                desc="Approximates a polynomial kernel's features (Tensor Sketch)."),
    Transformer("ml.lle", "Locally Linear Embedding", "LocallyLinearEmbedding",
                "sklearn.manifold", (P("n_components", "int", 2, lo=1),
                                     P("n_neighbors", "int", 10, lo=2)), seeded=True,
                desc="A non-linear embedding that keeps each point's relation to its "
                     "neighbours."),
]
UNIVARIATE_SELECTORS = ("ml.select_k_best", "ml.select_percentile", "ml.select_fpr",
                        "ml.select_fdr", "ml.select_fwe")
NONNEGATIVE_TRANSFORMERS = ("ml.nmf", "ml.lda_topics", "ml.additive_chi2_sampler")


SEARCH = BlockDefinition(
    type_id="ml.hyperparameter_search", display_name="Hyperparameter Search",
    category="Model Selection", color=family_color("classic"),
    params=(P("method", "enum", "grid", options=("grid", "random", "halving")),
            P("param_grid", "str", "",
              help="Per line or ';'-separated: name: v1, v2, v3   e.g. max_depth: 3, 6, None"),
            P("cv", "int", 5, lo=2, hi=20), P("n_iter", "int", 20, lo=1,
                                                help="Candidates for random search"),
            P("scoring", "enum", "auto", options=("auto", "accuracy", "f1_macro", "roc_auc",
                                                   "neg_log_loss", "r2",
                                                   "neg_mean_absolute_error",
                                                   "neg_root_mean_squared_error"))),
    description="Tune estimator hyperparameters with cross-validated search.",
    library="scikit-learn",
    checks_fn=lambda p: _grid_checks(p["param_grid"]),
)


def parse_grid(text: str) -> dict[str, list[str]]:
    grid: dict[str, list[str]] = {}
    for line in str(text).replace(";", "\n").splitlines():
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(f"expected 'name: v1, v2' but got {line.strip()!r}")
        name, values = line.split(":", 1)
        vals = [v.strip() for v in values.split(",") if v.strip()]
        if not name.strip().isidentifier() or not vals:
            raise ValueError(f"invalid grid entry {line.strip()!r}")
        grid[name.strip()] = vals
    return grid


def _grid_checks(text) -> list:
    try:
        grid = parse_grid(text)
    except ValueError as exc:
        return [("error", f"param_grid: {exc}")]
    if not grid:
        return [("warning", "param_grid is empty — the search only evaluates the defaults")]
    return []


ESTIMATOR_IDS = tuple(e.type_id for e in ESTIMATORS)
TRANSFORMER_IDS = tuple(t.type_id for t in TRANSFORMERS)
BY_ID: dict[str, Estimator | Transformer] = {b.type_id: b for b in (*ESTIMATORS, *TRANSFORMERS)}

CLUSTER_METRICS = [
    ("eval.silhouette", "Silhouette Score", "silhouette", "Cluster cohesion vs separation (−1..1)."),
    ("eval.davies_bouldin", "Davies-Bouldin Index", "davies_bouldin",
     "Average cluster similarity (lower is better)."),
    ("eval.calinski_harabasz", "Calinski-Harabasz Index", "calinski_harabasz",
     "Between- vs within-cluster dispersion (higher is better)."),
    ("eval.adjusted_rand", "Adjusted Rand Index", "adjusted_rand",
     "Agreement with true labels, chance-adjusted."),
    ("eval.nmi", "Normalized Mutual Information", "nmi", "Mutual information with true labels."),
]


def register_all() -> None:
    reg = get_registry()
    color = family_color("classic")
    for est in ESTIMATORS:
        lib = {"scikit-learn": "scikit-learn", "xgboost": "XGBoost", "lightgbm": "LightGBM",
               "catboost": "CatBoost"}[est.package]
        reg.register(BlockDefinition(
            type_id=est.type_id, display_name=est.name, category=est.category, color=color,
            params=tuple(est.params), checks_fn=est.checks, description=est.desc, library=lib,
            meta={"kind": "estimator", "task": est.task, "package": est.package}))
    for tr in TRANSFORMERS:
        reg.register(BlockDefinition(
            type_id=tr.type_id, display_name=tr.name,
            category="Text Features" if tr.stage == "text" else "Feature Engineering",
            color=color, params=tuple(tr.params), checks_fn=tr.checks, description=tr.desc,
            library={"umap-learn": "umap-learn", "imbalanced-learn": "imbalanced-learn"}.get(
                tr.package, "scikit-learn"),
            meta={"kind": "transformer", "stage": tr.stage},
            requires=(tr.module.split(".")[0],) if tr.package else (), extra=tr.extra))
    reg.register(SEARCH)
    for tid, name, key, desc in CLUSTER_METRICS:
        reg.register(BlockDefinition(
            type_id=tid, display_name=name, category="Metrics", color=family_color("evaluation"),
            description=desc, library="scikit-learn",
            meta={"kind": "metrics", "tasks": ["clustering"], "key": key}))
