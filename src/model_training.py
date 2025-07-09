from typing import Dict, Any
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def get_models_and_params() -> Dict[str, Any]:
    """
    Devuelve modelos base + grids reducidos para búsqueda rápida.
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
            'clf__max_depth': [None, 10],
            'clf__max_features': ['sqrt']
        },
        'MLP': {
            'clf__hidden_layer_sizes': [(50,), (100,), (100, 50)],
            'clf__activation': ['relu'],
            'clf__learning_rate_init': [0.001]
        },
        'SVM': {
            'clf__C': [0.1, 1, 10],
            'clf__kernel': ['linear'],  # RBF se omite para acelerar
            'clf__gamma': ['scale']     # solo aplica para RBF
        }
    }

    return models, param_grids


def train_with_randomsearch(X, y, model_name: str, model, param_grid) -> Dict[str, Any]:
    """
    Entrena usando SMOTE + RandomizedSearchCV + 3-fold CV.
    """
    pipe = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', model)
    ])

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_grid,
        scoring='f1',
        n_iter=5,  # máximo 5 combinaciones aleatorias
        cv=skf,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X, y)

    result = {
        'model_name': model_name,
        'best_params': search.best_params_,
        'best_score': search.best_score_
    }

    return result
