from typing import Dict, Any
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

def get_models_and_params() -> Dict[str, Any]:
    """
    Devuelve diccionarios con modelos base y grids de hiperparámetros.
    """
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(),
        'RandomForest': RandomForestClassifier(random_state=42),
        'MLP': MLPClassifier(max_iter=500, random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }

    param_grids = {
        'LogisticRegression': {
            'clf__C': [0.01, 0.1, 1, 10],
            'clf__penalty': ['l2'],
            'clf__solver': ['lbfgs']
        },
        'KNN': {
            'clf__n_neighbors': [3, 5, 7, 11],
            'clf__metric': ['euclidean']
        },
        'RandomForest': {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [None, 10, 20],
            'clf__max_features': ['sqrt', 'log2']
        },
        'MLP': {
            'clf__hidden_layer_sizes': [(50,), (100,), (100, 50)],
            'clf__activation': ['relu', 'tanh'],
            'clf__learning_rate_init': [0.001, 0.01]
        },
        'SVM': {
            'clf__C': [0.1, 1, 10],
            'clf__kernel': ['linear', 'rbf'],
            'clf__gamma': ['scale', 'auto']
        }
    }

    return models, param_grids


def train_with_gridsearch(X, y, model_name: str, model, param_grid) -> Dict[str, Any]:
    """
    Entrena un modelo con SMOTE + GridSearchCV + 10-fold CV.
    Devuelve mejores parámetros y puntaje.
    """
    pipe = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', model)
    ])

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='f1',
        cv=skf,
        n_jobs=-1
    )

    grid.fit(X, y)

    result = {
        'model_name': model_name,
        'best_params': grid.best_params_,
        'best_score': grid.best_score_
    }

    return result
