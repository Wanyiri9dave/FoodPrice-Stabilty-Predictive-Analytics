import streamlit as st
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Food Price Stability Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }

    .header-container h1 {
        font-size: 3em;
        margin-bottom: 10px;
        font-weight: bold;
    }

    .stat-box {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin: 10px;
        border-top: 4px solid #667eea;
        text-align: center;
    }

    .stat-box h4 {
        color: #667eea;
        margin-bottom: 10px;
        font-size: 1.1em;
    }

    .stat-number {
        font-size: 2em;
        font-weight: bold;
        color: #764ba2;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 12px;
        margin: 15px 0;
        color: white;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }

    .metric-value {
        font-size: 2.2em;
        font-weight: bold;
        color: #ffffff;
    }

    .prediction-result {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        margin: 20px 0;
    }

    .tab-content {
        background: white;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="header-container">
        <h1>🌾 Food Price Stability 📊</h1>
        <p>Predictive Analytics for Kenyan Agricultural Commodities</p>
        <p style="font-size: 0.9em; margin-top: 10px;">
            Food Prices + Rainfall + Fuel Prices | Machine Learning Powered
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================
if "model_cache" not in st.session_state:
    st.session_state.model_cache = {}

if "data_cache" not in st.session_state:
    st.session_state.data_cache = {}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_one_hot_encoder():
    """
    Creates OneHotEncoder that works with both old and new sklearn versions.
    New sklearn uses sparse_output=False.
    Old sklearn uses sparse=False.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


@st.cache_data
def load_data():
    """
    Load the three datasets from the data folder.
    Make sure the file names match exactly.
    """
    try:
        food_df = pd.read_csv("data/wfp_food_prices_ken.csv")
        rainfall_df = pd.read_csv("data/ken-rainfall-subnat-full.csv")
        fuel_df = pd.read_csv("data/Pump Prices Energy and Petroleum Regulatory Authority.csv")

        return food_df, rainfall_df, fuel_df

    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return None, None, None

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None


@st.cache_data
def preprocess_data(food_df, rainfall_df, fuel_df):
    """
    Clean, merge, and prepare food price, rainfall, and fuel datasets.
    """
    try:
        # -----------------------------
        # Process food data
        # -----------------------------
        food_df = food_df.copy()
        rainfall_df = rainfall_df.copy()
        fuel_df = fuel_df.copy()

        food_df["date"] = pd.to_datetime(food_df["date"], errors="coerce")
        food_df = food_df.dropna(subset=["date", "price"])

        food_df["year"] = food_df["date"].dt.year
        food_df["month"] = food_df["date"].dt.month
        food_df["date"] = food_df["date"].dt.to_period("M").dt.to_timestamp()

        # Some WFP data uses admin1 for county. We rename it to admin2 for merging.
        if "admin1" in food_df.columns and "admin2" not in food_df.columns:
            food_df = food_df.rename(columns={"admin1": "admin2"})

        required_food_cols = ["date", "admin2", "commodity", "pricetype", "price", "year", "month"]
        missing_food_cols = [col for col in required_food_cols if col not in food_df.columns]
        if missing_food_cols:
            st.error(f"Missing columns in food dataset: {missing_food_cols}")
            return None

        # -----------------------------
        # Process rainfall data
        # -----------------------------
        rainfall_df["date"] = pd.to_datetime(rainfall_df["date"], errors="coerce")
        rainfall_df = rainfall_df.dropna(subset=["date"])

        required_rainfall_cols = ["PCODE", "rfh", "r3q"]
        missing_rainfall_cols = [col for col in required_rainfall_cols if col not in rainfall_df.columns]
        if missing_rainfall_cols:
            st.error(f"Missing columns in rainfall dataset: {missing_rainfall_cols}")
            return None

        rainfall_df["date"] = rainfall_df["date"].dt.to_period("M").dt.to_timestamp()

        rainfall_df = (
            rainfall_df
            .groupby(["PCODE", "date"], as_index=False)
            .agg({"rfh": "mean", "r3q": "mean"})
        )

        pcode_to_county = {
            "KE001": "Mombasa", "KE002": "Kwale", "KE003": "Kilifi", "KE004": "Tana River",
            "KE005": "Lamu", "KE006": "Taita Taveta", "KE007": "Garissa", "KE008": "Wajir",
            "KE009": "Mandera", "KE010": "Marsabit", "KE011": "Isiolo", "KE012": "Meru",
            "KE013": "Tharaka-Nithi", "KE014": "Embu", "KE015": "Kitui", "KE016": "Machakos",
            "KE017": "Makueni", "KE018": "Nyandarua", "KE019": "Nyeri", "KE020": "Kirinyaga",
            "KE021": "Murang'a", "KE022": "Kiambu", "KE023": "Turkana", "KE024": "West Pokot",
            "KE025": "Samburu", "KE026": "Trans Nzoia", "KE027": "Uasin Gishu",
            "KE028": "Elgeyo-Marakwet", "KE029": "Nandi", "KE030": "Baringo",
            "KE031": "Laikipia", "KE032": "Nakuru", "KE033": "Narok", "KE034": "Kajiado",
            "KE035": "Kericho", "KE036": "Bomet", "KE037": "Kakamega", "KE038": "Vihiga",
            "KE039": "Bungoma", "KE040": "Busia", "KE041": "Siaya", "KE042": "Kisumu",
            "KE043": "Homa Bay", "KE044": "Migori", "KE045": "Kisii", "KE046": "Nyamira",
            "KE047": "Nairobi"
        }

        rainfall_df["admin2"] = rainfall_df["PCODE"].astype(str).str[:5].map(pcode_to_county)
        rainfall_df = rainfall_df.dropna(subset=["admin2"])

        rainfall_df = rainfall_df.sort_values(["admin2", "date"])
        rainfall_df["r3q_lag_3"] = rainfall_df.groupby("admin2")["r3q"].shift(3)
        rainfall_df = rainfall_df.dropna(subset=["r3q_lag_3"])

        # -----------------------------
        # Process fuel data
        # -----------------------------
        required_fuel_cols = ["Date", "Diesel (AGO)"]
        missing_fuel_cols = [col for col in required_fuel_cols if col not in fuel_df.columns]
        if missing_fuel_cols:
            st.error(f"Missing columns in fuel dataset: {missing_fuel_cols}")
            return None

        fuel_df["Date"] = pd.to_datetime(fuel_df["Date"], errors="coerce")
        fuel_df = fuel_df.dropna(subset=["Date"])

        fuel_df["Date"] = fuel_df["Date"].dt.to_period("M").dt.to_timestamp()
        fuel_df = fuel_df.rename(columns={"Date": "date", "Diesel (AGO)": "diesel_price"})

        fuel_df["diesel_price"] = pd.to_numeric(fuel_df["diesel_price"], errors="coerce")
        fuel_df = fuel_df.dropna(subset=["diesel_price"])

        fuel_df = (
            fuel_df
            .groupby("date", as_index=False)
            .agg({"diesel_price": "mean"})
            .sort_values("date")
        )

        fuel_df["diesel_price_lag_1"] = fuel_df["diesel_price"].shift(1)
        fuel_df = fuel_df.dropna(subset=["diesel_price_lag_1"])

        # -----------------------------
        # Merge all datasets
        # -----------------------------
        merged_df = food_df.merge(
            rainfall_df[["date", "admin2", "rfh", "r3q_lag_3"]],
            on=["date", "admin2"],
            how="left"
        )

        merged_df = merged_df.merge(
            fuel_df[["date", "diesel_price_lag_1"]],
            on="date",
            how="left"
        )

        merged_df = merged_df.dropna(subset=[
            "admin2", "commodity", "pricetype", "price",
            "year", "month", "rfh", "r3q_lag_3", "diesel_price_lag_1"
        ])

        return merged_df

    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return None


def prepare_train_test_data(merged_df):
    """
    Split data, scale numeric columns, encode categorical columns.
    """
    try:
        features = [
            "admin2", "commodity", "pricetype",
            "year", "month", "rfh", "r3q_lag_3", "diesel_price_lag_1"
        ]

        target = "price"

        X = merged_df[features].copy()
        y = merged_df[target].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        numeric_cols = ["year", "month", "rfh", "r3q_lag_3", "diesel_price_lag_1"]
        categorical_cols = ["admin2", "commodity", "pricetype"]

        scaler = StandardScaler()
        encoder = create_one_hot_encoder()

        X_train_num = scaler.fit_transform(X_train[numeric_cols])
        X_test_num = scaler.transform(X_test[numeric_cols])

        X_train_cat = encoder.fit_transform(X_train[categorical_cols])
        X_test_cat = encoder.transform(X_test[categorical_cols])

        feature_names = numeric_cols + list(encoder.get_feature_names_out(categorical_cols))

        X_train_final = np.hstack([X_train_num, X_train_cat])
        X_test_final = np.hstack([X_test_num, X_test_cat])

        X_train_final = pd.DataFrame(X_train_final, columns=feature_names)
        X_test_final = pd.DataFrame(X_test_final, columns=feature_names)

        return {
            "X_train": X_train_final,
            "X_test": X_test_final,
            "y_train": y_train,
            "y_test": y_test,
            "scaler": scaler,
            "encoder": encoder,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "feature_names": feature_names
        }

    except Exception as e:
        st.error(f"Error preparing train/test data: {e}")
        return None


def train_models(X_train, y_train):
    """
    Train selected machine learning models.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )

    trained_models = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[model_name] = model

    return trained_models


def evaluate_models(models, X_test, y_test):
    """
    Evaluate all trained models.
    """
    results = {}

    for model_name, model in models.items():
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[model_name] = {
            "model": model,
            "y_pred": y_pred,
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        }

    return results


def prepare_prediction_input(
    location,
    commodity,
    price_type,
    year,
    month,
    rainfall,
    rainfall_lag,
    diesel_lag,
    scaler,
    encoder,
    numeric_cols,
    categorical_cols
):
    """
    Prepare one row of user input for prediction.
    """
    prediction_df = pd.DataFrame({
        "admin2": [location],
        "commodity": [commodity],
        "pricetype": [price_type],
        "year": [year],
        "month": [month],
        "rfh": [rainfall],
        "r3q_lag_3": [rainfall_lag],
        "diesel_price_lag_1": [diesel_lag]
    })

    pred_num = scaler.transform(prediction_df[numeric_cols])
    pred_cat = encoder.transform(prediction_df[categorical_cols])

    pred_final = np.hstack([pred_num, pred_cat])

    return pred_final


# ============================================================
# LOAD DATA
# ============================================================
food_df, rainfall_df, fuel_df = load_data()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px; color: white; margin-bottom: 20px;">
            <h2>⚙️ Configuration</h2>
            <p style="font-size: 0.9em;">Customize your prediction</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("🔮 Make a Prediction")

    with st.form("prediction_form"):
        location = st.selectbox(
            "📍 Location / County",
            [
                "Nairobi", "Mombasa", "Kisumu", "Nakuru", "Kiambu", "Kajiado",
                "Uasin Gishu", "Kericho", "Nyeri", "Kirinyaga", "Machakos",
                "Murang'a", "Bungoma", "Kakamega", "Kisii", "Migori",
                "Homa Bay", "Garissa"
            ]
        )

        commodity = st.selectbox(
            "🌾 Commodity",
            [
                "Maize", "Rice", "Beans", "Sorghum", "Millet", "Wheat",
                "White maize", "Kidney beans", "Pigeon peas", "Cassava",
                "Potatoes", "Onions"
            ]
        )

        price_type = st.selectbox("💳 Price Type", ["Retail", "Wholesale"])

        year = st.slider("📆 Year", 2020, 2026, 2024)
        month = st.slider("📅 Month", 1, 12, 6)

        rainfall = st.number_input("🌧️ Rainfall rfh", min_value=0.0, value=50.0)
        rainfall_lag = st.number_input("🌧️ Rainfall lag 3 months", min_value=0.0, value=50.0)
        diesel_lag = st.number_input("⛽ Diesel price lag 1 month", min_value=0.0, value=180.0)

        submit_prediction = st.form_submit_button("Predict Price", use_container_width=True)


# ============================================================
# MAIN APP
# ============================================================
if food_df is None or rainfall_df is None or fuel_df is None:
    st.error("Unable to load data. Please check your data folder and file names.")

else:
    tab1, tab2, tab3 = st.tabs(["📈 Dashboard", "🤖 Model Training", "📊 Predictions"])

    # ========================================================
    # TAB 1: DASHBOARD
    # ========================================================
    with tab1:
        st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
        st.markdown("## 📊 Data Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
                <div class="stat-box">
                    <h4>🌽 Food Price Records</h4>
                    <div class="stat-number">{len(food_df):,}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
                <div class="stat-box">
                    <h4>🌧️ Rainfall Records</h4>
                    <div class="stat-number">{len(rainfall_df):,}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:
            st.markdown(
                f"""
                <div class="stat-box">
                    <h4>⛽ Fuel Records</h4>
                    <div class="stat-number">{len(fuel_df):,}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("---")

        st.markdown("### 📈 Food Price Trends")

        try:
            food_chart_df = food_df.copy()
            food_chart_df["date"] = pd.to_datetime(food_chart_df["date"], errors="coerce")
            food_chart_df = food_chart_df.dropna(subset=["date", "price"])

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                monthly_avg = (
                    food_chart_df
                    .groupby(food_chart_df["date"].dt.to_period("M"))["price"]
                    .mean()
                )

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(range(len(monthly_avg)), monthly_avg.values, marker="o")
                ax.set_title("Average Food Prices Over Time")
                ax.set_xlabel("Time")
                ax.set_ylabel("Average Price")
                ax.grid(True, alpha=0.3)
                ax.set_xticks([])
                st.pyplot(fig)

            with chart_col2:
                commodity_avg = (
                    food_chart_df
                    .groupby("commodity")["price"]
                    .mean()
                    .sort_values(ascending=False)
                    .head(10)
                )

                fig, ax = plt.subplots(figsize=(10, 5))
                commodity_avg.plot(kind="barh", ax=ax)
                ax.set_title("Top Commodities by Average Price")
                ax.set_xlabel("Average Price")
                ax.grid(True, alpha=0.3, axis="x")
                st.pyplot(fig)

        except Exception as e:
            st.warning(f"Could not create charts: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # ========================================================
    # TAB 2: MODEL TRAINING
    # ========================================================
    with tab2:
        st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
        st.markdown("## 🤖 Machine Learning Model Training")

        st.info("Click the button below to preprocess the data, train the models, and compare performance.")

        train_button = st.button("🚀 Train Models", use_container_width=True)

        if train_button:
            with st.spinner("Training models..."):
                merged_df = preprocess_data(food_df, rainfall_df, fuel_df)

                if merged_df is not None and len(merged_df) > 0:
                    prepared_data = prepare_train_test_data(merged_df)

                    if prepared_data is not None:
                        models = train_models(prepared_data["X_train"], prepared_data["y_train"])
                        results = evaluate_models(
                            models,
                            prepared_data["X_test"],
                            prepared_data["y_test"]
                        )

                        st.session_state.model_cache = results
                        st.session_state.data_cache = prepared_data
                        st.session_state.data_cache["merged_df"] = merged_df

                        st.success("✅ Models trained successfully!")

                else:
                    st.error("Merged dataset is empty. Please check date formats, county names, and missing values.")

        if st.session_state.model_cache:
            st.markdown("---")
            st.markdown("### 📊 Model Performance Comparison")

            comparison_df = pd.DataFrame([
                {
                    "Model": model_name,
                    "MAE": values["mae"],
                    "RMSE": values["rmse"],
                    "R² Score": values["r2"]
                }
                for model_name, values in st.session_state.model_cache.items()
            ])

            st.dataframe(
                comparison_df.style.format({
                    "MAE": "{:.2f}",
                    "RMSE": "{:.2f}",
                    "R² Score": "{:.4f}"
                }),
                use_container_width=True
            )

            st.markdown("### 📈 Performance Charts")

            chart_col1, chart_col2, chart_col3 = st.columns(3)

            with chart_col1:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(comparison_df["Model"], comparison_df["MAE"])
                ax.set_title("MAE")
                ax.set_xlabel("Mean Absolute Error")
                ax.grid(True, alpha=0.3, axis="x")
                st.pyplot(fig)

            with chart_col2:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(comparison_df["Model"], comparison_df["RMSE"])
                ax.set_title("RMSE")
                ax.set_xlabel("Root Mean Squared Error")
                ax.grid(True, alpha=0.3, axis="x")
                st.pyplot(fig)

            with chart_col3:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(comparison_df["Model"], comparison_df["R² Score"])
                ax.set_title("R² Score")
                ax.set_xlabel("R² Score")
                ax.grid(True, alpha=0.3, axis="x")
                st.pyplot(fig)

            st.markdown("### 🎯 Actual vs Predicted Prices")

            for model_name, values in st.session_state.model_cache.items():
                with st.expander(f"{model_name} Prediction Plot"):
                    y_test = st.session_state.data_cache["y_test"]
                    y_pred = values["y_pred"]

                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(y_test, y_pred, alpha=0.6)
                    ax.plot(
                        [y_test.min(), y_test.max()],
                        [y_test.min(), y_test.max()],
                        linestyle="--"
                    )
                    ax.set_xlabel("Actual Price")
                    ax.set_ylabel("Predicted Price")
                    ax.set_title(f"{model_name}: Actual vs Predicted")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)

    # ========================================================
    # TAB 3: PREDICTIONS
    # ========================================================
    with tab3:
        st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
        st.markdown("## 🔮 Food Price Prediction")

        if not st.session_state.model_cache:
            st.warning("Please train the models first in the Model Training tab.")

        else:
            if submit_prediction:
                prediction_results = {}

                for model_name, values in st.session_state.model_cache.items():
                    try:
                        pred_input = prepare_prediction_input(
                            location=location,
                            commodity=commodity,
                            price_type=price_type,
                            year=year,
                            month=month,
                            rainfall=rainfall,
                            rainfall_lag=rainfall_lag,
                            diesel_lag=diesel_lag,
                            scaler=st.session_state.data_cache["scaler"],
                            encoder=st.session_state.data_cache["encoder"],
                            numeric_cols=st.session_state.data_cache["numeric_cols"],
                            categorical_cols=st.session_state.data_cache["categorical_cols"]
                        )

                        predicted_price = values["model"].predict(pred_input)[0]
                        prediction_results[model_name] = predicted_price

                    except Exception as e:
                        st.error(f"{model_name} prediction failed: {e}")

                if prediction_results:
                    st.markdown(
                        f"""
                        <div class="prediction-result">
                            <h2>🎯 Prediction Result</h2>
                            <p><strong>{commodity}</strong> in <strong>{location}</strong></p>
                            <p>{price_type} price | {month}/{year}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    cols = st.columns(len(prediction_results))

                    for idx, (model_name, predicted_price) in enumerate(prediction_results.items()):
                        with cols[idx]:
                            st.markdown(
                                f"""
                                <div class="metric-card">
                                    <h3>{model_name}</h3>
                                    <div class="metric-value">KES {predicted_price:,.2f}</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                    avg_prediction = np.mean(list(prediction_results.values()))
                    min_prediction = min(prediction_results.values())
                    max_prediction = max(prediction_results.values())
                    price_range = max_prediction - min_prediction

                    st.markdown("---")
                    st.markdown("### 📊 Prediction Summary")

                    s1, s2, s3 = st.columns(3)

                    with s1:
                        st.metric("Average Prediction", f"KES {avg_prediction:,.2f}")

                    with s2:
                        st.metric("Lowest Estimate", f"KES {min_prediction:,.2f}")

                    with s3:
                        st.metric("Highest Estimate", f"KES {max_prediction:,.2f}")

                    if avg_prediction > 0:
                        confidence = 100 - ((price_range / avg_prediction) * 100)
                        confidence = max(0, min(100, confidence))
                    else:
                        confidence = 0

                    st.markdown("### 🎯 Model Agreement Confidence")
                    st.progress(confidence / 100)
                    st.write(f"Confidence: **{confidence:.1f}%**")

                    if confidence >= 80:
                        st.success("High confidence: the models are giving similar predictions.")
                    elif confidence >= 60:
                        st.info("Moderate confidence: the models are fairly close.")
                    else:
                        st.warning("Low confidence: the models are giving different predictions.")

        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 35px; border-radius: 15px; color: white; margin-top: 30px; text-align:center;">
        <h2>📚 About This Application</h2>
        <p>
            This app predicts food prices in Kenya using food price records,
            rainfall indicators, and diesel fuel prices.
        </p>
        <p>
            Models used: Linear Regression, Random Forest, and XGBoost if installed.
        </p>
        <p style="font-size: 0.85em; opacity: 0.9;">
            Built with Streamlit for Food Price Stability Predictive Analytics.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
