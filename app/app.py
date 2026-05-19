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

# Custom CSS with beautiful styling
st.markdown("""
    <style>
    * {
        margin: 0;
        padding: 0;
    }
    
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
    
    .header-container p {
        font-size: 1.2em;
        opacity: 0.95;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        color: white;
        text-align: center;
        border-left: 5px solid #f093fb;
    }
    
    .metric-card h3 {
        font-size: 1.5em;
        margin-bottom: 10px;
    }
    
    .metric-card .metric-value {
        font-size: 2.5em;
        font-weight: bold;
        color: #f093fb;
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
    
    .stat-box .stat-number {
        font-size: 2em;
        font-weight: bold;
        color: #764ba2;
    }
    
    .commodity-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        text-align: center;
    }
    
    .commodity-icon {
        font-size: 2.5em;
        margin-bottom: 10px;
    }
    
    .model-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 15px 0;
        border-left: 5px solid #667eea;
    }
    
    .model-card h3 {
        color: #667eea;
        margin-bottom: 15px;
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
    
    .prediction-result h2 {
        font-size: 2.5em;
        margin-bottom: 15px;
    }
    
    .prediction-value {
        font-size: 3em;
        font-weight: bold;
        color: #fff;
    }
    
    .tab-content {
        background: white;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
    }
    
    .feature-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    
    .feature-box h3 {
        margin-bottom: 10px;
    }
    
    .feature-box .number {
        font-size: 2em;
        font-weight: bold;
        color: #f093fb;
    }
    
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown("""
    <div class="header-container">
        <h1>🌾 Food Price Stability 📊</h1>
        <p>Predictive Analytics for Kenyan Agricultural Commodities</p>
        <p style="font-size: 0.9em; margin-top: 10px;">Powered by Machine Learning | Real-time Price Forecasting</p>
    </div>
    """, unsafe_allow_html=True)

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
        # STEP 1: Clean and prepare food prices
        food_df = food_df.copy()
        food_df.columns = food_df.columns.str.lower().str.strip()
        food_df['date'] = pd.to_datetime(food_df['date'])
        food_df['year'] = food_df['date'].dt.year
        food_df['month'] = food_df['date'].dt.month
        
        # Rename admin1 to admin2 if it exists
        if 'admin1' in food_df.columns:
            food_df = food_df.rename(columns={'admin1': 'admin2'})
        
        food_df = food_df.reset_index(drop=True)
        
        # STEP 2: Clean and prepare rainfall
        rainfall_df = rainfall_df.copy()
        rainfall_df.columns = rainfall_df.columns.str.lower().str.strip()
        rainfall_df['date'] = pd.to_datetime(rainfall_df['date'])
        
        # Group and aggregate
        rainfall_agg = rainfall_df.groupby(['pcode', pd.Grouper(key='date', freq='MS')]).agg({
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
        rainfall_agg['pcode_short'] = rainfall_agg['pcode'].str[:5]
        rainfall_agg['admin2'] = rainfall_agg['pcode_short'].map(pcode_to_county)
        rainfall_agg = rainfall_agg.dropna(subset=['admin2'])
        
        # Calculate lag
        rainfall_agg = rainfall_agg.sort_values(['admin2', 'date']).reset_index(drop=True)
        rainfall_agg['r3q_lag_3'] = rainfall_agg.groupby('admin2')['r3q'].shift(3)
        rainfall_agg = rainfall_agg.dropna(subset=['r3q_lag_3']).reset_index(drop=True)
        
        # Keep only needed columns
        rainfall_clean = rainfall_agg[['date', 'admin2', 'rfh', 'r3q_lag_3']].copy()
        
        # STEP 3: Clean and prepare fuel
        fuel_df = fuel_df.copy()
        fuel_df.columns = fuel_df.columns.str.lower().str.strip()
        
        # Find the date column (might be called 'date' or 'Date')
        date_col = [c for c in fuel_df.columns if 'date' in c.lower()][0] if any('date' in c.lower() for c in fuel_df.columns) else None
        if not date_col:
            raise ValueError("No date column found in fuel data")
        
        fuel_df['date'] = pd.to_datetime(fuel_df[date_col])
        fuel_df['date'] = fuel_df['date'].dt.to_period('M').dt.to_timestamp()
        
        # Find diesel price column
        diesel_col = [c for c in fuel_df.columns if 'diesel' in c.lower() or 'ago' in c.lower()][0] if any('diesel' in c.lower() or 'ago' in c.lower() for c in fuel_df.columns) else None
        if not diesel_col:
            raise ValueError("No diesel price column found in fuel data")
        
        fuel_df['diesel_price'] = pd.to_numeric(fuel_df[diesel_col], errors='coerce')
        fuel_df = fuel_df[['date', 'diesel_price']].dropna().drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
        fuel_df['diesel_price_lag_1'] = fuel_df['diesel_price'].shift(1)
        fuel_clean = fuel_df[['date', 'diesel_price_lag_1']].dropna().reset_index(drop=True)
        
        # STEP 4: Merge datasets carefully
        # Start with food data
        result = food_df[['date', 'admin2', 'commodity', 'pricetype', 'year', 'month', 'price']].copy()
        
        # Merge with rainfall
        result = result.merge(rainfall_clean, on=['date', 'admin2'], how='left', validate='many_to_one')
        
        # Merge with fuel
        result = result.merge(fuel_clean, on='date', how='left', validate='many_to_one')
        
        # STEP 5: Final cleaning
        result = result.dropna(subset=['rfh', 'r3q_lag_3', 'diesel_price_lag_1']).reset_index(drop=True)
        
        # Ensure no duplicate columns
        result = result.loc[:, ~result.columns.duplicated()]
        
        # Prepare features and target
        X = result[['admin2', 'commodity', 'pricetype', 'year', 'month', 'rfh', 'r3q_lag_3', 'diesel_price_lag_1']].copy()
        y = result['price'].copy()
        
        # STEP 6: Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        
        # STEP 7: Scale numeric features
        numeric_features = ['year', 'month', 'rfh', 'r3q_lag_3', 'diesel_price_lag_1']
        scaler = StandardScaler()
        X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
        X_test[numeric_features] = scaler.transform(X_test[numeric_features])
        
        # STEP 8: Encode categorical features
        categorical_features = ['admin2', 'commodity', 'pricetype']
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        X_train_cat = X_train[categorical_features].copy()
        X_test_cat = X_test[categorical_features].copy()
        
        X_train_encoded = encoder.fit_transform(X_train_cat)
        X_test_encoded = encoder.transform(X_test_cat)
        
        # Get feature names
        feature_names = encoder.get_feature_names_out(categorical_features)
        
        # Create dataframes with encoded features
        X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=feature_names, index=X_train.index)
        X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=feature_names, index=X_test.index)
        
        # Remove categorical features from original and concat with encoded
        X_train = X_train.drop(columns=categorical_features)
        X_test = X_test.drop(columns=categorical_features)
        
        X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded_df.reset_index(drop=True)], axis=1)
        X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded_df.reset_index(drop=True)], axis=1)
        
        return X_train, X_test, y_train.reset_index(drop=True), y_test.reset_index(drop=True), scaler, encoder, result
        
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        import traceback
        st.error(traceback.format_exc())
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
    st.markdown("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 10px; color: white; margin-bottom: 20px;">
            <h2 style="margin-bottom: 10px;">⚙️ Configuration</h2>
            <p style="opacity: 0.9; font-size: 0.9em;">Customize your analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Model selection
    st.subheader("🤖 Model Selection")
    available_models = ["Linear Regression", "Random Forest"]
    if XGBOOST_AVAILABLE:
        available_models.append("XGBoost")
    
    selected_models = st.multiselect(
        "Choose models to compare:",
        available_models,
        default=["Random Forest"] if len(available_models) > 1 else ["Linear Regression"]
    )
    
    st.markdown("---")
    
    # Prediction section
    st.subheader("🔮 Make a Prediction")
    st.markdown("**Fill in the details below to predict food prices**")
    
    with st.form("prediction_form"):
        location = st.selectbox("📍 Location (County):", [
            "Nairobi", "Mombasa", "Kisumu", "Nakuru", "Kiambu", "Kajiado",
            "Uasin Gishu", "Kericho", "Nyeri", "Kirinyaga", "Machakos", "Muranga",
            "Bungoma", "Kakamega", "Kisii", "Migori", "Homa Bay", "Garissa"
        ])
        commodity = st.selectbox("🌾 Commodity:", [
            "Maize", "Rice", "Beans", "Sorghum", "Millet", "Wheat", "White maize",
            "Kidney beans", "Pigeon peas", "Cassava", "Potatoes", "Onions"
        ])
        price_type = st.selectbox("💳 Price Type:", ["Retail", "Wholesale"])
        month = st.slider("📅 Month:", 1, 12, 6)
        year = st.slider("📆 Year:", 2020, 2026, 2024)
        
        submit = st.form_submit_button("🔮 Predict Price", use_container_width=True)

# Main content
st.markdown("---")

# Load data
food_df, rainfall_df, fuel_df = load_data()

if food_df is not None and rainfall_df is not None and fuel_df is not None:
    # Data overview
    tab1, tab2, tab3 = st.tabs(["📈 Dashboard", "🤖 Model Training", "📊 Predictions"])
    
    with tab1:
        st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
        st.markdown("## 📊 Data Overview & Insights", unsafe_allow_html=True)
        
        # Create beautiful stat cards
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        
        with stat_col1:
            st.markdown("""
                <div class="stat-box">
                    <h4>🌽 Food Price Records</h4>
                    <div class="stat-number">""" + f"{len(food_df):,}" + """</div>
                </div>
                """, unsafe_allow_html=True)
        
        with stat_col2:
            st.markdown("""
                <div class="stat-box">
                    <h4>💧 Rainfall Data Points</h4>
                    <div class="stat-number">""" + f"{len(rainfall_df):,}" + """</div>
                </div>
                """, unsafe_allow_html=True)
        
        with stat_col3:
            st.markdown("""
                <div class="stat-box">
                    <h4>⛽ Fuel Price Records</h4>
                    <div class="stat-number">""" + f"{len(fuel_df):,}" + """</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Commodities section
        st.markdown("### 🌾 Key Commodities Tracked")
        commodities = food_df['commodity'].unique()[:6]
        
        commodity_icons = {
            'Maize': '🌽', 'Rice': '🍚', 'Beans': '🫘', 
            'Wheat': '🌾', 'Sorghum': '🌾', 'Millet': '🌾'
        }
        
        comm_cols = st.columns(len(commodities))
        for idx, commodity in enumerate(commodities):
            with comm_cols[idx]:
                icon = commodity_icons.get(commodity, '🌾')
                st.markdown(f"""
                    <div class="commodity-card">
                        <div class="commodity-icon">{icon}</div>
                        <h4>{commodity}</h4>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Data visualization
        st.markdown("### 📈 Price Trends Analysis")
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("#### Average Food Prices Over Time")
            fig, ax = plt.subplots(figsize=(10, 5))
            food_df['date'] = pd.to_datetime(food_df['date'])
            daily_avg = food_df.groupby(food_df['date'].dt.to_period('M'))['price'].mean()
            ax.fill_between(range(len(daily_avg)), daily_avg.values, alpha=0.3, color='#667eea')
            ax.plot(range(len(daily_avg)), daily_avg.values, marker='o', linewidth=2.5, 
                   color='#667eea', markersize=4)
            ax.set_title("Price Trajectory", fontsize=12, fontweight='bold')
            ax.set_xlabel("Time Period")
            ax.set_ylabel("Price (KES)")
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xticks([])
            st.pyplot(fig)
        
        with chart_col2:
            st.markdown("#### Average Price by Commodity")
            fig, ax = plt.subplots(figsize=(10, 5))
            commodity_avg = food_df.groupby('commodity')['price'].mean().sort_values(ascending=False).head(10)
            colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#ffa502']
            colors = (colors * (len(commodity_avg) // len(colors) + 1))[:len(commodity_avg)]
            commodity_avg.plot(kind='barh', ax=ax, color=colors)
            ax.set_title("Top Commodities by Price", fontsize=12, fontweight='bold')
            ax.set_xlabel("Average Price (KES)")
            ax.grid(True, alpha=0.3, axis='x', linestyle='--')
            st.pyplot(fig)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
        st.markdown("## 🤖 Machine Learning Model Training", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("Train multiple models and compare their performance")
        with col2:
            train_button = st.button("🚀 Train Models", key="train_btn", use_container_width=True)
        
        if train_button:
            with st.spinner("🔄 Training models... This may take a few minutes"):
                progress_bar = st.progress(0)
                # Preprocess data
                X_train, X_test, y_train, y_test, scaler, encoder, merged_df = preprocess_and_train_models(
                    food_df, rainfall_df, fuel_df
                )
                progress_bar.progress(20)
                
                if X_train is not None:
                    st.session_state.model_cache = {}
                    
                    # Train models
                    models_to_train = {
                        "Linear Regression": train_linear_regression,
                        "Random Forest": train_random_forest,
                    }
                    
                    if XGBOOST_AVAILABLE:
                        models_to_train["XGBoost"] = train_xgboost
                    
                    model_count = len(models_to_train)
                    for idx, (model_name, train_func) in enumerate(models_to_train.items()):
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
                        progress_bar.progress(20 + ((idx + 1) / model_count) * 60)
                    
                    st.session_state.data_cache = {
                        'X_train': X_train,
                        'X_test': X_test,
                        'y_train': y_train,
                        'y_test': y_test,
                        'scaler': scaler,
                        'encoder': encoder,
                        'merged_df': merged_df
                    }
                    
                    progress_bar.progress(100)
                    st.success("✅ All models trained successfully!")
        
        # Display model comparison
        if st.session_state.model_cache:
            st.markdown("---")
            st.markdown("### 📊 Model Performance Comparison")
            
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
            
            # Display comparison table with better styling
            st.markdown("#### Performance Metrics Table")
            comparison_styled = comparison_df.style.format({
                'MAE': '{:.2f}',
                'RMSE': '{:.2f}',
                'R² Score': '{:.4f}'
            }).highlight_max(axis=0, color='#90EE90').highlight_min(axis=0, color='#FFB6C6')
            st.dataframe(comparison_styled, use_container_width=True)
            
            st.markdown("---")
            st.markdown("#### Performance Visualizations")
            
            # Visualize comparison with better styling
            vis_col1, vis_col2, vis_col3 = st.columns(3)
            
            with vis_col1:
                fig, ax = plt.subplots(figsize=(8, 5))
                colors = ['#667eea', '#764ba2', '#f093fb'][:len(comparison_df)]
                ax.barh(comparison_df['Model'], comparison_df['MAE'], color=colors)
                ax.set_title("Mean Absolute Error", fontsize=13, fontweight='bold', pad=20)
                ax.set_xlabel("MAE (KES)")
                ax.grid(True, alpha=0.3, axis='x', linestyle='--')
                for i, v in enumerate(comparison_df['MAE']):
                    ax.text(v + 0.5, i, f'{v:.2f}', va='center', fontweight='bold')
                st.pyplot(fig)
            
            with vis_col2:
                fig, ax = plt.subplots(figsize=(8, 5))
                colors = ['#667eea', '#764ba2', '#f093fb'][:len(comparison_df)]
                ax.barh(comparison_df['Model'], comparison_df['RMSE'], color=colors)
                ax.set_title("Root Mean Squared Error", fontsize=13, fontweight='bold', pad=20)
                ax.set_xlabel("RMSE (KES)")
                ax.grid(True, alpha=0.3, axis='x', linestyle='--')
                for i, v in enumerate(comparison_df['RMSE']):
                    ax.text(v + 0.5, i, f'{v:.2f}', va='center', fontweight='bold')
                st.pyplot(fig)
            
            with vis_col3:
                fig, ax = plt.subplots(figsize=(8, 5))
                colors = ['#667eea', '#764ba2', '#f093fb'][:len(comparison_df)]
                ax.barh(comparison_df['Model'], comparison_df['R² Score'], color=colors)
                ax.set_title("R² Score", fontsize=13, fontweight='bold', pad=20)
                ax.set_xlabel("R² Score")
                ax.grid(True, alpha=0.3, axis='x', linestyle='--')
                for i, v in enumerate(comparison_df['R² Score']):
                    ax.text(v + 0.01, i, f'{v:.4f}', va='center', fontweight='bold')
                st.pyplot(fig)
            
            st.markdown("---")
            st.markdown("#### Actual vs Predicted Prices")
            
            for model_name in comparison_df['Model']:
                with st.expander(f"📈 {model_name} - Prediction Analysis"):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    y_test = st.session_state.data_cache['y_test']
                    y_pred = st.session_state.model_cache[model_name]['y_pred']
                    
                    ax.scatter(y_test, y_pred, alpha=0.6, s=50, color='#667eea', edgecolors='#764ba2')
                    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                           'r--', lw=2.5, label='Perfect Prediction')
                    ax.set_xlabel("Actual Prices (KES)", fontsize=11, fontweight='bold')
                    ax.set_ylabel("Predicted Prices (KES)", fontsize=11, fontweight='bold')
                    ax.set_title(f"{model_name}: Actual vs Predicted Prices", fontsize=13, fontweight='bold', pad=20)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    ax.legend(fontsize=10)
                    st.pyplot(fig)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
        st.markdown("## 🔮 Price Predictions", unsafe_allow_html=True)
        
        if st.session_state.model_cache:
            # Create comparison dataframe for model info
            comparison_data = []
            for model_name in st.session_state.model_cache.keys():
                comparison_data.append({'Model': model_name})
            comparison_df = pd.DataFrame(comparison_data)
            
            if submit:
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
                
                st.markdown("---")
                st.markdown(f"""
                    <div class="prediction-result">
                        <h2>🎯 Prediction Results</h2>
                        <p style="font-size: 1.1em;">
                            <strong>{commodity}</strong> in <strong>{location}</strong>
                        </p>
                        <p style="font-size: 0.95em; opacity: 0.9;">
                            📍 {price_type} | 📅 {month}/{year}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("### Predictions from Different Models")
                
                # Display predictions from different models
                prediction_cols = st.columns(len(comparison_df))
                
                prediction_results = {}
                for idx, model_name in enumerate(comparison_df['Model']):
                    try:
                        model = st.session_state.model_cache[model_name]['model']
                        scaler = st.session_state.data_cache['scaler']
                        encoder = st.session_state.data_cache['encoder']
                        
                        # Prepare prediction input - match training pipeline
                        numeric_cols = ['year', 'month', 'rfh', 'r3q_lag_3', 'diesel_price_lag_1']
                        cat_cols = ['admin2', 'commodity', 'pricetype']
                        
                        # Scale numeric features
                        pred_numeric = scaler.transform(prediction_data[numeric_cols])
                        pred_numeric_df = pd.DataFrame(pred_numeric, columns=numeric_cols)
                        
                        # Encode categorical features
                        pred_categorical = encoder.transform(prediction_data[cat_cols])
                        pred_categorical_df = pd.DataFrame(
                            pred_categorical,
                            columns=encoder.get_feature_names_out(cat_cols)
                        )
                        
                        # Concatenate in same order as training
                        pred_input = pd.concat([pred_numeric_df, pred_categorical_df], axis=1)
                        
                        price_pred = model.predict(pred_input.values)[0]
                        prediction_results[model_name] = price_pred
                        
                        with prediction_cols[idx]:
                            st.markdown(f"""
                                <div class="metric-card">
                                    <h3>{model_name}</h3>
                                    <div class="metric-value">KES {price_pred:.2f}</div>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    except Exception as e:
                        with prediction_cols[idx]:
                            st.warning(f"⚠️ {model_name}: Error - {str(e)[:80]}")
                
                # Average prediction with beautiful styling
                if prediction_results:
                    st.markdown("---")
                    avg_pred = np.mean(list(prediction_results.values()))
                    
                    st.markdown(f"""
                        <div class="prediction-result" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                            <h2>💰 Ensemble Prediction</h2>
                            <p style="font-size: 0.95em; margin-bottom: 20px;">Average of all models</p>
                            <div class="prediction-value">KES {avg_pred:.2f}</div>
                            <p style="font-size: 0.9em; margin-top: 15px; opacity: 0.9;">
                                Based on {len(prediction_results)} model{'s' if len(prediction_results) > 1 else ''}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Additional insights
                    st.markdown("### 📊 Prediction Insights")
                    insight_col1, insight_col2, insight_col3 = st.columns(3)
                    
                    with insight_col1:
                        min_pred = min(prediction_results.values())
                        st.markdown(f"""
                            <div class="stat-box">
                                <h4>🔽 Lowest Estimate</h4>
                                <div class="stat-number">KES {min_pred:.2f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with insight_col2:
                        max_pred = max(prediction_results.values())
                        st.markdown(f"""
                            <div class="stat-box">
                                <h4>🔝 Highest Estimate</h4>
                                <div class="stat-number">KES {max_pred:.2f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with insight_col3:
                        price_range = max_pred - min_pred
                        st.markdown(f"""
                            <div class="stat-box">
                                <h4>📏 Price Range</h4>
                                <div class="stat-number">KES {price_range:.2f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Confidence level
                    confidence = 100 - (price_range / avg_pred * 100) if avg_pred > 0 else 0
                    confidence = max(0, min(100, confidence))
                    
                    st.markdown("---")
                    st.markdown("### 🎯 Prediction Confidence")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.progress(confidence / 100)
                    with col2:
                        st.markdown(f"<h3 style='text-align: center;'>{confidence:.1f}%</h3>", unsafe_allow_html=True)
                    
                    if confidence > 80:
                        st.success("✅ High confidence prediction - Models are in good agreement")
                    elif confidence > 60:
                        st.info("ℹ️ Moderate confidence - Consider market factors")
                    else:
                        st.warning("⚠️ Lower confidence - High model variance, use with caution")
        
        else:
            st.markdown("""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                           padding: 40px; border-radius: 15px; color: white; text-align: center;
                           box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                    <h2>⚠️ No Models Trained Yet</h2>
                    <p style="font-size: 1.1em;">Please train models first in the <strong>Model Training</strong> tab!</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.error("Unable to load data. Please check if all CSV files exist in the 'data' folder.")

# Footer
st.markdown("---")
st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
               padding: 40px; border-radius: 15px; color: white; margin: 30px 0;">
        
        <h2 style="text-align: center; margin-bottom: 20px;">📚 About This Application</h2>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin: 20px 0;">
            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; text-align: center;">
                <h4>📊 Data Sources</h4>
                <p style="font-size: 0.9em; line-height: 1.6;">
                    • WFP Food Prices<br>
                    • Kenya Rainfall Data<br>
                    • EPRA Pump Prices
                </p>
            </div>
            
            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; text-align: center;">
                <h4>🤖 ML Models</h4>
                <p style="font-size: 0.9em; line-height: 1.6;">
                    • Linear Regression<br>
                    • Random Forest<br>
                    • XGBoost
                </p>
            </div>
            
            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; text-align: center;">
                <h4>🎯 Features</h4>
                <p style="font-size: 0.9em; line-height: 1.6;">
                    • Real-time Predictions<br>
                    • Model Comparison<br>
                    • Price Trends
                </p>
            </div>
        </div>
        
        <div style="text-align: center; border-top: 1px solid rgba(255,255,255,0.2); padding-top: 20px; margin-top: 20px;">
            <p style="margin: 5px 0; font-size: 0.95em;">
                <strong>🌾 Food Price Stability - Predictive Analytics</strong>
            </p>
            <p style="margin: 5px 0; font-size: 0.85em; opacity: 0.9;">
                Kenya Agricultural Commodities | Machine Learning Powered | 2024
            </p>
            <p style="margin: 10px 0; font-size: 0.8em; opacity: 0.8;">
                Built with ❤️ for price stability and food security
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
