"""
Modulo de definicion, construccion y optimizacion de modelos de Machine Learning
utilizando Pipelines estrictos de scikit-learn para evitar fuga de informacion.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def build_pipeline(classifier_name: str, use_pca: bool = False, pca_components: float = 0.95,
                   random_state: int = 42) -> tuple:
    """
    Construye un Pipeline de scikit-learn con escalado, reduccion de dimensionalidad opcional
    y el clasificador seleccionado, junto con su malla de hiperparametros.
    """
    steps = [("scaler", StandardScaler())]

    if use_pca:
        steps.append(("pca", PCA(n_components=pca_components, random_state=random_state)))

    param_grid = {}

    if classifier_name == "svm_linear":
        clf = SVC(kernel="linear", probability=True, random_state=random_state)
        steps.append(("clf", clf))
        param_grid = {
            "clf__C": [0.01, 0.1, 1.0, 10.0]
        }

    elif classifier_name == "svm_rbf":
        clf = SVC(kernel="rbf", probability=True, random_state=random_state)
        steps.append(("clf", clf))
        param_grid = {
            "clf__C": [0.1, 1.0, 10.0, 100.0],
            "clf__gamma": ["scale", "auto", 0.01, 0.001]
        }

    elif classifier_name == "knn":
        clf = KNeighborsClassifier()
        steps.append(("clf", clf))
        param_grid = {
            "clf__n_neighbors": [3, 5, 7, 9],
            "clf__weights": ["uniform", "distance"],
            "clf__metric": ["euclidean", "manhattan"]
        }

    elif classifier_name == "random_forest":
        # Random Forest no requiere escalador estricto, pero el scaler no le afecta
        clf = RandomForestClassifier(random_state=random_state)
        steps.append(("clf", clf))
        param_grid = {
            "clf__n_estimators": [50, 100, 200],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_split": [2, 5]
        }

    elif classifier_name == "mlp":
        clf = MLPClassifier(max_iter=600, early_stopping=True, random_state=random_state)
        steps.append(("clf", clf))
        param_grid = {
            "clf__hidden_layer_sizes": [(50,), (100,), (100, 50)],
            "clf__alpha": [0.0001, 0.01, 0.1],
            "clf__learning_rate_init": [0.001, 0.01]
        }

    else:
        raise ValueError(f"Clasificador no reconocido: {classifier_name}")

    pipeline = Pipeline(steps)
    return pipeline, param_grid


def optimize_classifier(pipeline: Pipeline, param_grid: dict, X_train, y_train,
                        cv_folds: int = 5, scoring: str = "f1_macro", n_jobs: int = 1):
    """
    Optimiza los hiperparametros del pipeline utilizando GridSearchCV
    con validacion cruzada estratificada sobre el conjunto de entrenamiento.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
        verbose=0
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_
