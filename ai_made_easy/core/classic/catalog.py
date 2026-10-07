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
            library="scikit-learn", meta={"kind": "transformer", "stage": tr.stage}))
    reg.register(SEARCH)
    for tid, name, key, desc in CLUSTER_METRICS:
        reg.register(BlockDefinition(
            type_id=tid, display_name=name, category="Metrics", color=family_color("evaluation"),
            description=desc, library="scikit-learn",
            meta={"kind": "metrics", "tasks": ["clustering"], "key": key}))
