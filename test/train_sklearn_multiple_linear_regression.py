import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = "../house_prices_linear.csv"
FEATURE_COLS = ["sqft", "bedrooms", "age"]
TARGET_COL = "price"


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[("scale", StandardScaler(), FEATURE_COLS)],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LinearRegression(fit_intercept=True)),
        ]
    )


def main():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    reg = model.named_steps["model"]
    scaler = model.named_steps["preprocess"].named_transformers_["scale"]

    scaled_coefs = reg.coef_
    scaled_intercept = reg.intercept_

    raw_coefs = scaled_coefs / scaler.scale_
    raw_intercept = (
        scaled_intercept - (scaled_coefs * scaler.mean_ / scaler.scale_).sum()
    )

    mse = mean_squared_error(y_test, preds)
    rmse = mse**0.5
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("Scikit-learn Multiple Linear Regression")
    print(f"Rows: {len(df)}")
    print(f"Train/Test: {len(X_train)}/{len(X_test)}")
    print(f"R^2: {r2:.6f}")
    print(f"MAE: {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MSE: {mse:,.2f}")

    print("\nWeights in standardized feature space:")
    for name, coef in zip(FEATURE_COLS, scaled_coefs):
        print(f"  {name}: {coef:.6f}")
    print(f"  intercept: {scaled_intercept:.6f}")

    print("\nWeights converted back to original feature units:")
    for name, coef in zip(FEATURE_COLS, raw_coefs):
        print(f"  {name}: {coef:.6f}")
    print(f"  intercept: {raw_intercept:.6f}")


if __name__ == "__main__":
    main()
