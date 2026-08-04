import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def load_data():
    data_path = "data/AmesHousing.csv"
    df = pd.read_csv(data_path)
    return df


def clean_data(df):
    # Drop ID-like columns if present
    df = df.drop(columns=["PID"], errors="ignore")

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna("Missing")

    return df


def encode_and_split(df):
    df_encoded = pd.get_dummies(df, drop_first=True)

    X = df_encoded.drop(columns=["SalePrice"])
    y = df_encoded["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    results = {}

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lin = lin_reg.predict(X_test)
    rmse_lin = mean_squared_error(y_test, y_pred_lin) ** 0.5
    r2_lin = r2_score(y_test, y_pred_lin)
    results["Linear Regression"] = {"RMSE": rmse_lin, "R2": r2_lin}

    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rmse_rf = mean_squared_error(y_test, y_pred_rf) ** 0.5
    r2_rf = r2_score(y_test, y_pred_rf)
    results["Random Forest"] = {"RMSE": rmse_rf, "R2": r2_rf}

    return results

def main():
    print("Loading data...")
    df = load_data()

    print("Cleaning data...")
    df = clean_data(df)

    print("Encoding and splitting...")
    X_train, X_test, y_train, y_test = encode_and_split(df)

    print("Training and evaluating models...")
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    print("\nModel performance:")
    for name, metrics in results.items():
        print(f"{name}: RMSE = {metrics['RMSE']:.2f}, R2 = {metrics['R2']:.3f}")

if __name__ == "__main__":
    main()
