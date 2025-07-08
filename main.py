from src.data_preprocessing import load_data, encode_and_scale
from src.model_training import get_models_and_params, train_with_gridsearch


def main():
    # Ruta al dataset
    filepath = 'data/bank-direct-marketing-campaigns.csv'

    # 1. Cargar
    df = load_data(filepath)
    print(df.columns)
    print("Primeras filas del dataset:")
    print(df.head())

    # 2. Preprocesar
    X, y = encode_and_scale(df)
    print(f"Shape de X: {X.shape}")
    print(f"Distribución de clases:\n{y.value_counts(normalize=True)}")
    

    models, param_grids = get_models_and_params()

    results = []

    for name, model in models.items():
        print(f"\nEntrenando: {name}")
        result = train_with_gridsearch(X, y, name, model, param_grids[name])
        print(f"Mejores parámetros para {name}: {result['best_params']}")
        print(f"Mejor F1: {result['best_score']:.4f}")
        results.append(result)

    # Guardar resultados en CSV
    import pandas as pd
    pd.DataFrame(results).to_csv('outputs/gridsearch_results.csv', index=False)


if __name__ == '__main__':
    main()
