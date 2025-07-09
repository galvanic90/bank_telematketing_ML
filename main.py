from src.data_preprocessing import load_data, encode_and_scale
from src.model_training import get_models_and_params, train_with_randomsearch

def main():
    filepath = 'data/bank-direct-marketing-campaigns.csv'
    df = load_data(filepath)
    X, y = encode_and_scale(df)

    # 🔑 Usa solo una fracción para explorar más rápido
    X_sample = X.sample(frac=0.2, random_state=42)
    y_sample = y.loc[X_sample.index]

    models, param_grids = get_models_and_params()

    results = []

    for name, model in models.items():
        print(f"\nEntrenando: {name}")
        result = train_with_randomsearch(X_sample, y_sample, name, model, param_grids[name])
        print(f"Mejores parámetros para {name}: {result['best_params']}")
        print(f"Mejor F1: {result['best_score']:.4f}")
        results.append(result)

    import pandas as pd
    pd.DataFrame(results).to_csv('outputs/randomsearch_results.csv', index=False)


if __name__ == '__main__':
    main()


