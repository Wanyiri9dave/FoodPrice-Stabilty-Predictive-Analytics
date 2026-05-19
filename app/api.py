"""
Flask API for Food Price Prediction
Provides REST endpoints for price predictions and model management
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import warnings

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Global state for models and preprocessing objects
app.config['models'] = {}
app.config['scaler'] = None
app.config['encoder'] = None
app.config['models_trained'] = False


def load_data():
    """Load and preprocess data from CSV files"""
    try:
        food_df = pd.read_csv('data/wfp_food_prices_ken.csv')
        rainfall_df = pd.read_csv('data/ken-rainfall-subnat-full.csv')
        fuel_df = pd.read_csv('data/Pump Prices Energy and Petroleum Regulatory Authority.csv')
        return food_df, rainfall_df, fuel_df
    except Exception as e:
        return None, None, None


def preprocess_and_train_models(food_df, rainfall_df, fuel_df):
    """Preprocess data and train models"""
    try:
        # Process food prices - create a clean copy
        food_df = food_df.copy()
        food_df['date'] = pd.to_datetime(food_df['date'])
        food_df['year'] = food_df['date'].dt.year
        food_df['month'] = food_df['date'].dt.month
        food_df = food_df.rename(columns={'admin1': 'admin2'})
        food_df = food_df.reset_index(drop=True)
        
        # Process rainfall - create a clean copy
        rainfall_df = rainfall_df.copy()
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
        rainfall_df = rainfall_df.sort_values(['admin2', 'date']).reset_index(drop=True)
        rainfall_df['r3q_lag_3'] = rainfall_df.groupby('admin2')['r3q'].shift(3)
        rainfall_df = rainfall_df.dropna(subset=['r3q_lag_3']).reset_index(drop=True)
        
        # Select only needed columns from rainfall
        rainfall_clean = rainfall_df[['date', 'admin2', 'rfh', 'r3q_lag_3']].copy()
        
        # Process fuel prices - create a clean copy
        fuel_df = fuel_df.copy()
        fuel_df['Date'] = pd.to_datetime(fuel_df['Date'])
        fuel_df['Date'] = fuel_df['Date'].dt.to_period('M').dt.to_timestamp()
        fuel_df = fuel_df.rename(columns={'Date': 'date', 'Diesel (AGO)': 'diesel_price'})
        fuel_df = fuel_df[['date', 'diesel_price']].drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
        fuel_df['diesel_price_lag_1'] = fuel_df['diesel_price'].shift(1)
        fuel_clean = fuel_df[['date', 'diesel_price_lag_1']].copy()
        
        # Merge datasets step by step
        # First merge: food_df with rainfall_clean
        merged_df = food_df.merge(rainfall_clean, on=['date', 'admin2'], how='left')
        
        # Remove any duplicate columns that might have been created
        if merged_df.columns.duplicated().any():
            merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
        
        # Second merge: merged_df with fuel_clean
        merged_df = merged_df.merge(fuel_clean, on='date', how='left')
        
        # Remove any duplicate columns again
        if merged_df.columns.duplicated().any():
            merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
        
        # Clean data
        merged_df = merged_df.drop_duplicates(subset=['date', 'admin2'], keep='first').reset_index(drop=True)
        merged_df = merged_df.dropna(subset=['rfh', 'r3q_lag_3', 'diesel_price_lag_1']).reset_index(drop=True)
        
        # Select required columns
        required_cols = ['admin2', 'commodity', 'pricetype', 'year', 'month', 'rfh', 'r3q_lag_3', 'diesel_price_lag_1', 'price']
        missing_cols = [col for col in required_cols if col not in merged_df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing columns after preprocessing: {missing_cols}")
        
        # Final check for duplicate columns
        if merged_df.columns.duplicated().any():
            merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
        
        X = merged_df[['admin2', 'commodity', 'pricetype', 'year', 'month', 'rfh', 'r3q_lag_3', 'diesel_price_lag_1']].copy()
        y = merged_df['price'].copy()
        
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
        print(f"Error preprocessing data: {e}")
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


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_trained': app.config['models_trained'],
        'available_models': list(app.config['models'].keys())
    }), 200


@app.route('/api/train', methods=['POST'])
def train_models():
    """Train all models endpoint"""
    try:
        food_df, rainfall_df, fuel_df = load_data()
        
        if food_df is None:
            return jsonify({'error': 'Failed to load data'}), 400
        
        X_train, X_test, y_train, y_test, scaler, encoder, merged_df = preprocess_and_train_models(
            food_df, rainfall_df, fuel_df
        )
        
        if X_train is None:
            return jsonify({'error': 'Failed to preprocess data'}), 400
        
        # Store scaler and encoder
        app.config['scaler'] = scaler
        app.config['encoder'] = encoder
        
        # Train models
        models_to_train = {
            "Linear Regression": train_linear_regression,
            "Random Forest": train_random_forest,
        }
        
        if XGBOOST_AVAILABLE:
            models_to_train["XGBoost"] = train_xgboost
        
        training_results = {}
        for model_name, train_func in models_to_train.items():
            model = train_func(X_train, y_train)
            if model is not None:
                y_pred, mae, rmse, r2 = evaluate_model(model, X_test, y_test)
                app.config['models'][model_name] = model
                training_results[model_name] = {
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'r2': float(r2)
                }
        
        app.config['models_trained'] = True
        
        return jsonify({
            'status': 'success',
            'message': 'Models trained successfully',
            'models': training_results
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict food price endpoint"""
    try:
        if not app.config['models_trained']:
            return jsonify({'error': 'Models not trained yet. Please train models first.'}), 400
        
        data = request.get_json()
        
        # Validate input
        required_fields = ['location', 'commodity', 'price_type', 'month', 'year']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Create prediction dataframe
        prediction_data = pd.DataFrame({
            'admin2': [data['location']],
            'commodity': [data['commodity']],
            'pricetype': [data['price_type']],
            'year': [data['year']],
            'month': [data['month']],
            'rfh': [data.get('rfh', 50.0)],
            'r3q_lag_3': [data.get('r3q_lag_3', 50.0)],
            'diesel_price_lag_1': [data.get('diesel_price', 100.0)]
        })
        
        scaler = app.config['scaler']
        encoder = app.config['encoder']
        
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
        
        # Get predictions from all models
        predictions = {}
        for model_name, model in app.config['models'].items():
            price_pred = model.predict(pred_input.values)[0]
            predictions[model_name] = float(price_pred)
        
        # Calculate ensemble prediction
        ensemble_prediction = np.mean(list(predictions.values()))
        
        return jsonify({
            'status': 'success',
            'input': {
                'location': data['location'],
                'commodity': data['commodity'],
                'price_type': data['price_type'],
                'month': data['month'],
                'year': data['year']
            },
            'predictions': predictions,
            'ensemble_prediction': float(ensemble_prediction),
            'average_prediction': float(ensemble_prediction),
            'min_prediction': float(min(predictions.values())),
            'max_prediction': float(max(predictions.values())),
            'price_range': float(max(predictions.values()) - min(predictions.values()))
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/status', methods=['GET'])
def models_status():
    """Get models training status and metrics"""
    try:
        if not app.config['models_trained']:
            return jsonify({
                'status': 'untrained',
                'message': 'No models trained yet'
            }), 200
        
        return jsonify({
            'status': 'trained',
            'available_models': list(app.config['models'].keys()),
            'model_count': len(app.config['models'])
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/data/info', methods=['GET'])
def data_info():
    """Get information about available data"""
    try:
        food_df, rainfall_df, fuel_df = load_data()
        
        if food_df is None:
            return jsonify({'error': 'Failed to load data'}), 400
        
        return jsonify({
            'status': 'success',
            'food_prices': {
                'records': int(len(food_df)),
                'commodities': list(food_df['commodity'].unique())[:20],
                'locations': list(food_df['admin1'].unique())[:20]
            },
            'rainfall_data': {
                'records': int(len(rainfall_df))
            },
            'fuel_prices': {
                'records': int(len(fuel_df))
            }
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/commodities', methods=['GET'])
def get_commodities():
    """Get list of available commodities"""
    try:
        food_df, _, _ = load_data()
        if food_df is None:
            return jsonify({'error': 'Failed to load data'}), 400
        
        commodities = list(food_df['commodity'].unique())
        return jsonify({
            'status': 'success',
            'commodities': commodities,
            'count': len(commodities)
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/locations', methods=['GET'])
def get_locations():
    """Get list of available locations"""
    try:
        food_df, _, _ = load_data()
        if food_df is None:
            return jsonify({'error': 'Failed to load data'}), 400
        
        locations = list(food_df['admin1'].unique())
        return jsonify({
            'status': 'success',
            'locations': locations,
            'count': len(locations)
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
