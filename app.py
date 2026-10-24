"""
Automobile Fuel Efficiency (MPG) Prediction App
------------------------------------------------
Pipeline: data extraction -> cleaning -> missing value treatment ->
          scaling -> encoding -> model selection -> prediction (Streamlit UI)

Dataset: UCI "Auto MPG" (398 cars, 1970-1982).
Put your own copy next to this file as `auto-mpg.csv` to use it instead of
downloading from UCI.
"""

import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="MPG Predictor", page_icon="🚗", layout="wide")

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
)
LOCAL_FILE = "auto-mpg.csv"
RAW_COLUMNS = [
    "mpg", "cylinders", "displacement", "horsepower", "weight",
    "acceleration", "model_year", "origin", "car_name",
]
TARGET = "mpg"
NUMERIC_FEATURES = [
    "cylinders", "displacement", "horsepower", "weight",
    "acceleration", "model_year", "hp_per_weight",
]
CATEGORICAL_FEATURES = ["origin"]
ORIGIN_MAP = {1: "USA", 2: "Europe", 3: "Japan"}
RANDOM_STATE = 42


# ----------------------------------------------------------------------------
# 1. Data extraction
# ----------------------------------------------------------------------------
def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make column names consistent for locally supplied CSV files."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(r"[\s\-]+", "_", regex=True)
    )
    df = df.rename(columns={"name": "car_name", "year": "model_year", "model": "model_year"})
    return df


@st.cache_data(show_spinner="Loading data...")
def load_data() -> pd.DataFrame:
    if os.path.exists(LOCAL_FILE):
        df = _standardise_columns(pd.read_csv(LOCAL_FILE, na_values=["?", "NA", ""]))
    else:
        df = pd.read_csv(
            DATA_URL,
            names=RAW_COLUMNS,
            sep=r"\s+",
            quotechar='"',
            na_values="?",
        )
    missing_cols = [c for c in RAW_COLUMNS[:-1] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
    return df


# ----------------------------------------------------------------------------
# 2. Cleaning + feature engineering
# ----------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Force numeric types (turns stray '?' etc. into NaN)
    for col in ["mpg", "cylinders", "displacement", "horsepower",
                "weight", "acceleration", "model_year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalise 'origin' to readable labels (handles 1/2/3 or text values)
    if pd.api.types.is_numeric_dtype(df["origin"]):
        df["origin"] = df["origin"].map(ORIGIN_MAP)
    else:
        df["origin"] = df["origin"].astype(str).str.strip().str.title()
        df["origin"] = df["origin"].replace({"Usa": "USA"})

    # Model year: 70 -> 1970
    df["model_year"] = np.where(df["model_year"] < 100, df["model_year"] + 1900, df["model_year"])

    # Remove duplicates and rows with no target
    df = df.drop_duplicates().dropna(subset=[TARGET])

    # Remove impossible values
    df = df[(df["weight"].isna() | (df["weight"] > 0)) & (df[TARGET] > 0)]
    return df.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineered feature used both in training and at prediction time."""
    df = df.copy()
    df["hp_per_weight"] = df["horsepower"] / df["weight"]  # NaN if hp missing -> imputed
    return df


# ----------------------------------------------------------------------------
# 3. Preprocessing pipeline (missing values + scaling + encoding)
# ----------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),   # missing value treatment
        ("scaler", StandardScaler()),                    # scaling
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),  # encoding
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def candidate_models() -> dict:
    return {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


# ----------------------------------------------------------------------------
# 4. Training + model selection
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training models...")
def train_best_model():
    raw = load_data()
    df = add_features(clean_data(raw))

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    results, fitted = [], {}
    for name, model in candidate_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        cv_r2 = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2").mean()
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        results.append({
            "Model": name,
            "CV R²": round(cv_r2, 4),
            "Test R²": round(r2_score(y_test, preds), 4),
            "Test RMSE": round(float(np.sqrt(mean_squared_error(y_test, preds))), 3),
            "Test MAE": round(mean_absolute_error(y_test, preds), 3),
        })
        fitted[name] = pipe

    results_df = pd.DataFrame(results).sort_values("CV R²", ascending=False).reset_index(drop=True)
    best_name = results_df.loc[0, "Model"]

    # Refit the winner on ALL data for the final app
    best_pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("model", candidate_models()[best_name]),
    ]).fit(X, y)

    test_preds = fitted[best_name].predict(X_test)
    scatter_df = pd.DataFrame({"Actual MPG": y_test.values, "Predicted MPG": test_preds})

    return best_pipe, best_name, results_df, scatter_df, df


# ----------------------------------------------------------------------------
# 5. Streamlit UI
# ----------------------------------------------------------------------------
def main():
    st.title("🚗 Automobile Fuel Efficiency (MPG) Predictor")
    st.caption("Enter a car's specifications and get a predicted miles-per-gallon figure.")

    try:
        model, best_name, results_df, scatter_df, df = train_best_model()
    except Exception as e:
        st.error(f"Could not load data or train the model: {e}")
        st.info(
            "If the download fails, place an `auto-mpg.csv` file next to app.py "
            "with columns: mpg, cylinders, displacement, horsepower, weight, "
            "acceleration, model_year, origin."
        )
        st.stop()

    tab_predict, tab_model, tab_data = st.tabs(["🔮 Predict", "📊 Model performance", "🗂 Data"])

    # ---------------- Predict tab ----------------
    with tab_predict:
        st.sidebar.header("Car specifications")
        cylinders = st.sidebar.selectbox("Cylinders", [3, 4, 5, 6, 8], index=1)
        displacement = st.sidebar.slider(
            "Displacement (cu. inches)",
            float(df.displacement.min()), float(df.displacement.max()),
            float(df.displacement.median()),
        )
        hp_unknown = st.sidebar.checkbox("Horsepower unknown (auto-fill)")
        horsepower = st.sidebar.slider(
            "Horsepower",
            float(df.horsepower.min()), float(df.horsepower.max()),
            float(df.horsepower.median()),
            disabled=hp_unknown,
        )
        weight = st.sidebar.slider(
            "Weight (lbs)",
            float(df.weight.min()), float(df.weight.max()), float(df.weight.median()),
        )
        acceleration = st.sidebar.slider(
            "Acceleration (0-60 mph, sec)",
            float(df.acceleration.min()), float(df.acceleration.max()),
            float(df.acceleration.median()),
        )
        model_year = st.sidebar.slider(
            "Model year", int(df.model_year.min()), int(df.model_year.max()),
            int(df.model_year.median()),
        )
        origin = st.sidebar.selectbox("Origin", sorted(df.origin.dropna().unique()))

        input_df = pd.DataFrame([{
            "cylinders": cylinders,
            "displacement": displacement,
            "horsepower": np.nan if hp_unknown else horsepower,
            "weight": weight,
            "acceleration": acceleration,
            "model_year": model_year,
            "origin": origin,
        }])
        input_df = add_features(input_df)[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

        st.subheader("Your input")
        st.dataframe(input_df.drop(columns=["hp_per_weight"]), hide_index=True, use_container_width=True)

        if st.button("Predict MPG", type="primary"):
            pred = float(model.predict(input_df)[0])
            kmpl = pred * 0.425144  # 1 mpg (US) = 0.425144 km/L
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted fuel efficiency", f"{pred:.1f} MPG")
            c2.metric("Approx. km per litre", f"{kmpl:.1f} km/L")
            c3.metric("Model used", best_name)
            if hp_unknown:
                st.info("Horsepower was missing, so the median value learned from training data was used.")

    # ---------------- Model tab ----------------
    with tab_model:
        st.subheader("Model comparison")
        st.caption(f"Best model (by 5-fold CV R²): **{best_name}**. It is refit on all data for predictions.")
        st.dataframe(results_df, hide_index=True, use_container_width=True)
        st.subheader("Actual vs predicted (held-out test set)")
        st.scatter_chart(scatter_df, x="Actual MPG", y="Predicted MPG")

    # ---------------- Data tab ----------------
    with tab_data:
        st.subheader("Cleaned dataset")
        st.write(f"{df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df, use_container_width=True)
        st.subheader("Missing values before imputation")
        st.dataframe(
            load_data().isna().sum().rename("Missing").to_frame(), use_container_width=True
        )
        st.subheader("Summary statistics")
        st.dataframe(df.describe().T, use_container_width=True)


if __name__ == "__main__":
    main()
