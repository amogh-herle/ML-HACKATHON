"""Every model the team can train, with its untuned (baseline) configuration.

Add a model = add one ModelSpec here; the harness in evaluate.py does the rest.
Models sensitive to feature scale (linear, KNN, SVM, MLP) get a StandardScaler
inside a Pipeline, so the scaler is fitted on the training fold only.
"""
from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from pathsnap.config import SEED
from pathsnap.models.evaluate import ModelSpec

JOBS = 8  # leave cores/RAM for the rest of the laptop

REGISTRY: dict[str, ModelSpec] = {s.name: s for s in [
    # ---------------- recommended five (one per family) ----------------
    ModelSpec("logreg", "Linear / probabilistic",
              make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)),
              "StandardScaler + LogisticRegression(C=1.0, lbfgs)"),
    ModelSpec("knn", "Instance-based",
              make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance",
                                                                   n_jobs=JOBS)),
              "StandardScaler + KNN(k=25, distance-weighted), fit on 10% of GPS points",
              train_fraction=0.10,
              notes=["KNN prediction cost grows with training size; fitted on a 10% random "
                     "sample of GPS points (all candidates of each sampled point kept)."]),
    ModelSpec("rf", "Ensemble (bagging)",
              RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=JOBS,
                                     random_state=SEED),
              "RandomForest(200 trees, min_samples_leaf=5, max_features=sqrt)"),
    ModelSpec("lgbm", "Boosting",
              LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, n_jobs=JOBS,
                             random_state=SEED, verbose=-1),
              "LightGBM(400 trees, lr=0.05, num_leaves=63)"),
    ModelSpec("mlp", "Neural network",
              make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32),
                                                            early_stopping=True, max_iter=60,
                                                            random_state=SEED)),
              "StandardScaler + MLP(64-32, ReLU, Adam, early stopping)",
              notes=["early_stopping holds out a random 10% of training rows (not grouped by "
                     "trip) to decide when to stop."]),
    # ---------------- alternatives ----------------
    ModelSpec("xgb", "Boosting",
              XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, tree_method="hist",
                            n_jobs=JOBS, random_state=SEED, eval_metric="logloss"),
              "XGBoost(400 trees, lr=0.05, max_depth=6, hist)"),
    ModelSpec("hgb", "Boosting",
              HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED),
              "HistGradientBoosting(300 iters, lr=0.05)"),
    ModelSpec("extratrees", "Ensemble (bagging)",
              ExtraTreesClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=JOBS,
                                   random_state=SEED),
              "ExtraTrees(200 trees, min_samples_leaf=5)"),
    ModelSpec("dtree", "Tree-based",
              DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=SEED),
              "DecisionTree(max_depth=12, min_samples_leaf=20)"),
    ModelSpec("adaboost", "Boosting",
              AdaBoostClassifier(n_estimators=200, random_state=SEED),
              "AdaBoost(200 stumps)"),
    ModelSpec("gnb", "Linear / probabilistic", GaussianNB(), "GaussianNB"),
    ModelSpec("sgd", "Linear (SVM-style)",
              make_pipeline(StandardScaler(), SGDClassifier(loss="modified_huber", random_state=SEED)),
              "StandardScaler + linear SVM via SGD (modified Huber loss, gives probabilities)"),
]}

RECOMMENDED = ["logreg", "knn", "rf", "lgbm", "mlp"]
