import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Food Price Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🌾 Food Price Stability - Predictive Analytics")
st.markdown("### Predicting Food Prices in Kenya Using Machine Learning")

# Initialize session state
if 'model_cache' not in st.session_state:
    st.session_state.model_cache = {}
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}

@st.cache_resource
def load_data():
    """Load and preprocess data from CSV files"""
    try:
        # Load datasets
        food_df = pd.read_csv('data/wfp_food_prices_ken.csv')
        rainfall_df = pd.read_csv('data/ken-rainfall-subnat-full.csv')
        fuel_df = pd.read_csv('data/Pump Prices Energy and Petroleum Regulatory Authority.csv')
        
        return food_df, rainfall_df, fuel_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

@st.cache_resource
def preprocess_and_train_models(food_df, rainfall_df, fuel_df):
    """Preprocess data and train models"""
    try:
        # Process food prices
        food_df['date'] = pd.to_datetime(food_df['date'])
        food_df['year'] = food_df['date'].dt.year
        food_df['month'] = food_df['date'].dt.month
        food_df = food_df.rename(columns={'admin1': 'admin2'})
        
        # Process rainfall
        rainfall_df['date'] = pd.to_datetime(rainfall_df['date'])
        rainfall_df = rainfall_df.groupby(['PCODE', pd.Grouper(key='date', freq='MS')]).agg({
            'rfh': 'mean',
            'r3q': 'mean'
        }).reset_index()
        
        # Map PCODE to County
        pcode_to_county = {
            'KE001': 'Mombasa', 'KE002': 'Kwale', 'KE003': 'Kilifi', 'KE004': 'Tana River',
            'KE005': 'Lamu', 'KE006': 'Taita Taveta', 'KE007': 'Garissa', 'KE008': 'Wajir',
            'KE009': 'Mandera', 'KE010': 'Marsabit', 'KE011': 'Isiolo', 'KE012': 'Meru',
            'KE013': 'Tharaka-Nithi', 'KE014': 'Embu', 'KE015': 'Kitui', 'KE016': 'Machakos',
            'KE017': 'Makueni', 'KE018': 'Nyandarua', 'KE019': 'Nyeri', 'KE020': 'Kirinyaga',
            'KE021': 'Murang\'a', 'KE022': 'Kiambu', 'KE023': 'Turkana', 'KE024': 'West Pokot',
            'KE025': 'Samburu', 'KE026': 'Trans Nzoia', 'KE027': 'Uasin Gishu', 'KE028': 'Elgeyo-Marakwet',
            'KE029': 'Nandi', 'KE030': 'Baringo', 'KE031': 'Laikipia', 'KE032': 'Nakuru',
            'KE033': 'Narok', 'KE034': 'Kajiado', 'KE035': 'Kericho', 'KE036': 'Bomet',
            'KE037': 'Kakamega', 'KE038': 'Vihiga', 'KE039': 'Bungoma', 'KE040': 'Busia',
            'KE041': 'Siaya', 'KE042': 'Kisumu', 'KE043': 'Homa Bay', 'KE044': 'Migori',
            'KE045': 'Kisii', 'KE046': 'Nyamira', 'KE047': 'Nairobi'
        }
        rainfall_df['admin2'] = rainfall_df['PCODE'].str[:5].map(pcode_to_county)
        rainfall_df = rainfall_df.sort_values(['admin2', 'date'])
        rainfall_df['r3q_lag_3'] = rainfall_df.groupby('admin2')['r3q'].shift(3)
        rainfall_df.dropna(subset=['r3q_lag_3'], inplace=True)
        
        # Process fuel prices
        fuel_df['Date'] = pd.to_datetime(fuel_df['Date'])
        fuel_df['Date'] = fuel_df['Date'].dt.to_period('M').dt.to_timestamp()
        fuel_df = fuel_df.rename(columns={'Date': 'date', 'Diesel (AGO)': 'diesel_price'})
        fuel_df = fuel_df[['date', 'diesel_price']]
        fuel_df = fuel_df.sort_values('date')
        fuel_df['diesel_price_lag_1'] = fuel_df['diesel_price'].shift(1)
        
        # Merge datasets
        merged_df = food_df.merge(rainfall_df[['date', 'admin2', 'rfh', 'r3q_lag_3']], 
                                   on=['date', 'admin2'], how='left')
        merged_df = merged_df.merge(fuel_df[['date', 'diesel_price_lag_1']], 
                                     on='date', how='left')
        merged_df.dropna(inplace=True)
        
        # Prepare features and target
        X = merged_df[['admin2', 'commodity', 'pricetype', 'year', 'month', 'rfh', 'r3q_lag_3', 'diesel_price_lag_1']]
        y = merged_df['price']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardize numeric features
        numeric_features = ['year', 'month', 'rfh', 'r3q_lag_3', 'diesel_price_lag_1']
        scaler = StandardScaler()
        X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
        X_test[numeric_features] = scaler.transform(X_test[numeric_features])
        
        # One-hot encode categorical features
        categorical_features = ['admin2', 'commodity', 'pricetype']
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        X_train_encoded = encoder.fit_transform(X_train[categorical_features])
        X_test_encoded = encoder.transform(X_test[categorical_features])
        
        X_train_encoded_df = pd.DataFrame(
            X_train_encoded,
            columns=encoder.get_feature_names_out(categorical_features),
            index=X_train.index
        )
        X_test_encoded_df = pd.DataFrame(
            X_test_encoded,
            columns=encoder.get_feature_names_out(categorical_features),
            index=X_test.index
        )
        
        X_train = pd.concat([X_train.drop(columns=categorical_features), X_train_encoded_df], axis=1)
        X_test = pd.concat([X_test.drop(columns=categorical_features), X_test_encoded_df], axis=1)
        
        return X_train, X_test, y_train, y_test, scaler, encoder, merged_df
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return None, None, None, None, None, None, None

def train_linear_regression(X_train, y_train):
    """Train Linear Regression model"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Train Random Forest model"""
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    """Train XGBoost model"""
    if not XGBOOST_AVAILABLE:
        return None
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, 
                             random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = model.score(X_test, y_test)
    return y_pred, mae, rmse, r2

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    st.subheader("Model Selection")
    available_models = ["Linear Regression", "Random Forest"]
    if XGBOOST_AVAILABLE:
        available_models.append("XGBoost")
    
    selected_models = st.multiselect(
        "Choose models to compare:",
        available_models,
        default=["Random Forest"] if len(available_models) > 1 else ["Linear Regression"]
    )
    
    # Prediction section
    st.subheader("📊 Make a Prediction")
    with st.form("prediction_form"):
        location = st.selectbox("Location (County):", [
            "Nairobi", "Mombasa", "Kisumu", "Nakuru", "Kiambu", "Kajiado",
            "Uasin Gishu", "Kericho", "Nyeri", "Kirinyaga"
        ])
        commodity = st.selectbox("Commodity:", [
            "Maize", "Beans", "Rice", "Sorghum", "Millet", "Wheat"
        ])
        price_type = st.selectbox("Price Type:", ["Retail", "Wholesale"])
        month = st.slider("Month:", 1, 12, 6)
        year = st.slider("Year:", 2020, 2026, 2024)
        
        submit = st.form_submit_button("🔮 Predict Price")

# Main content
st.markdown("---")

# Load data
food_df, rainfall_df, fuel_df = load_data()

if food_df is not None and rainfall_df is not None and fuel_df is not None:
    # Data overview
    tab1, tab2, tab3 = st.tabs(["📈 Dashboard", "🤖 Model Training", "📊 Predictions"])
    
    with tab1:
        st.header("Data Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Food Price Records", len(food_df))
        with col2:
            st.metric("Rainfall Data Points", len(rainfall_df))
        with col3:
            st.metric("Fuel Price Records", len(fuel_df))
        
        st.subheader("Data Samples")
        sample_tabs = st.tabs(["Food Prices", "Rainfall", "Fuel Prices"])
        
        with sample_tabs[0]:
            st.dataframe(food_df.head(10), use_container_width=True)
        with sample_tabs[1]:
            st.dataframe(rainfall_df.head(10), use_container_width=True)
        with sample_tabs[2]:
            st.dataframe(fuel_df.head(10), use_container_width=True)
        
        # Visualizations
        st.subheader("Price Trends")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 5))
            food_df['date'] = pd.to_datetime(food_df['date'])
            daily_avg = food_df.groupby(food_df['date'].dt.to_period('M'))['price'].mean()
            ax.plot(daily_avg.index.astype(str), daily_avg.values, marker='o', linewidth=2)
            ax.set_title("Average Food Prices Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (KES)")
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 5))
            commodity_avg = food_df.groupby('commodity')['price'].mean().sort_values(ascending=False).head(10)
            commodity_avg.plot(kind='barh', ax=ax, color='steelblue')
            ax.set_title("Average Price by Commodity")
            ax.set_xlabel("Price (KES)")
            st.pyplot(fig)
    
    with tab2:
        st.header("🤖 Model Training & Evaluation")
        
        if st.button("Train Models", key="train_btn"):
            with st.spinner("Training models... This may take a few minutes"):
                # Preprocess data
                X_train, X_test, y_train, y_test, scaler, encoder, merged_df = preprocess_and_train_models(
                    food_df, rainfall_df, fuel_df
                )
                
                if X_train is not None:
                    st.session_state.model_cache = {}
                    
                    # Train models
                    models_to_train = {
                        "Linear Regression": train_linear_regression,
                        "Random Forest": train_random_forest,
                    }
                    
                    if XGBOOST_AVAILABLE:
                        models_to_train["XGBoost"] = train_xgboost
                    
                    for model_name, train_func in models_to_train.items():
                        model = train_func(X_train, y_train)
                        if model is not None:
                            y_pred, mae, rmse, r2 = evaluate_model(model, X_test, y_test)
                            st.session_state.model_cache[model_name] = {
                                'model': model,
                                'mae': mae,
                                'rmse': rmse,
                                'r2': r2,
                                'y_pred': y_pred
                            }
                    
                    st.session_state.data_cache = {
                        'X_train': X_train,
                        'X_test': X_test,
                        'y_train': y_train,
                        'y_test': y_test,
                        'scaler': scaler,
                        'encoder': encoder,
                        'merged_df': merged_df
                    }
                    
                    st.success("✅ Models trained successfully!")
        
        # Display model comparison
        if st.session_state.model_cache:
            st.subheader("Model Performance Comparison")
            
            # Create comparison DataFrame
            comparison_data = []
            for model_name, metrics in st.session_state.model_cache.items():
                comparison_data.append({
                    'Model': model_name,
                    'MAE': metrics['mae'],
                    'RMSE': metrics['rmse'],
                    'R² Score': metrics['r2']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Visualize comparison
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(comparison_df['Model'], comparison_df['MAE'], color='coral')
                ax.set_title("Mean Absolute Error (MAE)")
                ax.set_xlabel("MAE")
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(comparison_df['Model'], comparison_df['RMSE'], color='skyblue')
                ax.set_title("Root Mean Squared Error (RMSE)")
                ax.set_xlabel("RMSE")
                st.pyplot(fig)
            
            with col3:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(comparison_df['Model'], comparison_df['R² Score'], color='lightgreen')
                ax.set_title("R² Score")
                ax.set_xlabel("R² Score")
                st.pyplot(fig)
            
            # Actual vs Predicted
            st.subheader("Actual vs Predicted Prices")
            for model_name in comparison_df['Model']:
                fig, ax = plt.subplots(figsize=(10, 5))
                y_test = st.session_state.data_cache['y_test']
                y_pred = st.session_state.model_cache[model_name]['y_pred']
                
                ax.scatter(y_test, y_pred, alpha=0.6)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                ax.set_xlabel("Actual Prices")
                ax.set_ylabel("Predicted Prices")
                ax.set_title(f"{model_name}: Actual vs Predicted")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
    
    with tab3:
        st.header("🔮 Price Predictions")
        
        if st.session_state.model_cache:
            if submit:
                st.subheader(f"Prediction for {commodity} in {location} ({price_type})")
                st.info(f"📍 Location: {location} | 📦 Commodity: {commodity} | 💳 Type: {price_type} | 📅 {month}/{year}")
                
                # Create prediction data
                prediction_data = pd.DataFrame({
                    'admin2': [location],
                    'commodity': [commodity],
                    'pricetype': [price_type],
                    'year': [year],
                    'month': [month],
                    'rfh': [50.0],  # Default values
                    'r3q_lag_3': [50.0],
                    'diesel_price_lag_1': [100.0]
                })
                
                # Display predictions from different models
                col1, col2, col3 = st.columns(3)
                
                prediction_results = {}
                for idx, model_name in enumerate(comparison_df['Model']):
                    try:
                        model = st.session_state.model_cache[model_name]['model']
                        scaler = st.session_state.data_cache['scaler']
                        encoder = st.session_state.data_cache['encoder']
                        
                        # Prepare prediction input
                        numeric_cols = ['year', 'month', 'rfh', 'r3q_lag_3', 'diesel_price_lag_1']
                        cat_cols = ['admin2', 'commodity', 'pricetype']
                        
                        pred_numeric = scaler.transform(prediction_data[numeric_cols])
                        pred_categorical = encoder.transform(prediction_data[cat_cols])
                        
                        pred_input = np.hstack([pred_numeric, pred_categorical])
                        price_pred = model.predict(pred_input)[0]
                        prediction_results[model_name] = price_pred
                        
                        if idx % 3 == 0:
                            col = col1
                        elif idx % 3 == 1:
                            col = col2
                        else:
                            col = col3
                        
                        with col:
                            st.metric(model_name, f"KES {price_pred:.2f}")
                    
                    except Exception as e:
                        st.warning(f"Could not predict using {model_name}: {e}")
                
                # Average prediction
                if prediction_results:
                    avg_pred = np.mean(list(prediction_results.values()))
                    st.success(f"### 💰 Average Predicted Price: **KES {avg_pred:.2f}**")
        
        else:
            st.warning("⚠️ Please train models first in the 'Model Training' tab!")

else:
    st.error("Unable to load data. Please check if all CSV files exist in the 'data' folder.")

# Footer
st.markdown("---")
st.markdown("""
    ### 📚 About This App
    This application uses machine learning models trained on:
    - **Food Price Data**: WFP Kenya Food Prices
    - **Rainfall Data**: Kenya Rainfall Data
    - **Fuel Prices**: EPRA Pump Prices
    
    **Models Used:**
    - Linear Regression: Simple baseline model
    - Random Forest: Captures non-linear relationships
    - XGBoost: Advanced gradient boosting model
    
    ---
    *Food Price Stability - Predictive Analytics | Kenya | 2024*
""")
