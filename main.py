from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.model_selection import train_test_split

from src.data_preprocessing import load_data, encode_and_scale, count_yes_no
from src.model_evaluation import (
    plot_confusion_matrix,
    plot_roc_curve,
    show_classification_report
)


def main():
    df = load_data('data/bank-direct-marketing-campaigns.csv')
    count_yes_no(df)
    X, y = encode_and_scale(df)

   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )


    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            max_features='sqrt',
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)

  
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    plot_confusion_matrix(y_test, y_pred, title='Matriz de Confusión')
    auc = plot_roc_curve(y_test, y_proba, title='Curva ROC')
    show_classification_report(y_test, y_pred)

    print(f"\nAUC final: {auc:.4f}")


if __name__ == '__main__':
    main()
