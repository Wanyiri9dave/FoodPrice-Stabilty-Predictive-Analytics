# Food Price Prediction APIs

## Overview

The API provides REST endpoints for training machine learning models and predicting food prices in Kenya. The backend is built with Flask and exposes multiple endpoints for model management and price prediction.

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Running the API

```bash
python api.py
```

The API will start on `http://localhost:5000`

## API Endpoints

### 1. Health Check

Check if the API is running and see the current status of models.

**Endpoint:** `GET /api/health`

**Response:**

```json
{
  "status": "healthy",
  "models_trained": false,
  "available_models": []
}
```

---

### 2. Train Models

Train all available machine learning models (Linear Regression, Random Forest, XGBoost).

**Endpoint:** `POST /api/train`

**Request Body:**

```json
{}
```

**Response:**

```json
{
  "status": "success",
  "message": "Models trained successfully",
  "models": {
    "Linear Regression": {
      "mae": 45.32,
      "rmse": 62.18,
      "r2": 0.8234
    },
    "Random Forest": {
      "mae": 38.12,
      "rmse": 51.45,
      "r2": 0.8892
    },
    "XGBoost": {
      "mae": 35.67,
      "rmse": 48.92,
      "r2": 0.9023
    }
  }
}
```

---

### 3. Make a Prediction

Predict food prices for a specific commodity, location, and time period.

**Endpoint:** `POST /api/predict`

**Request Body:**

```json
{
  "location": "Nairobi",
  "commodity": "Maize",
  "price_type": "Retail",
  "month": 5,
  "year": 2024,
  "rfh": 50.0,
  "r3q_lag_3": 50.0,
  "diesel_price": 100.0
}
```

**Parameters:**

- `location` (required): County/Location name (e.g., "Nairobi", "Mombasa", "Kisumu")
- `commodity` (required): Commodity name (e.g., "Maize", "Rice", "Beans")
- `price_type` (required): "Retail" or "Wholesale"
- `month` (required): Month (1-12)
- `year` (required): Year (2020-2026)
- `rfh` (optional): Rainfall high forecast (default: 50.0)
- `r3q_lag_3` (optional): 3-month rainfall lag (default: 50.0)
- `diesel_price` (optional): Diesel price lag (default: 100.0)

**Response:**

```json
{
  "status": "success",
  "input": {
    "location": "Nairobi",
    "commodity": "Maize",
    "price_type": "Retail",
    "month": 5,
    "year": 2024
  },
  "predictions": {
    "Linear Regression": 85.42,
    "Random Forest": 82.15,
    "XGBoost": 81.92
  },
  "ensemble_prediction": 83.16,
  "average_prediction": 83.16,
  "min_prediction": 81.92,
  "max_prediction": 85.42,
  "price_range": 3.5
}
```

---

### 4. Models Status

Get the current status of trained models.

**Endpoint:** `GET /api/models/status`

**Response:**

```json
{
  "status": "trained",
  "available_models": ["Linear Regression", "Random Forest", "XGBoost"],
  "model_count": 3
}
```

---

### 5. Data Information

Get information about available data sources.

**Endpoint:** `GET /api/data/info`

**Response:**

```json
{
  "status": "success",
  "food_prices": {
    "records": 156834,
    "commodities": ["Maize", "Rice", "Beans", "Wheat", ...],
    "locations": ["Nairobi", "Mombasa", "Kisumu", ...]
  },
  "rainfall_data": {
    "records": 45632
  },
  "fuel_prices": {
    "records": 1203
  }
}
```

---

### 6. Get Commodities

Get list of all available commodities in the dataset.

**Endpoint:** `GET /api/commodities`

**Response:**

```json
{
  "status": "success",
  "commodities": ["Maize", "Rice", "Beans", "Sorghum", "Millet", ...],
  "count": 45
}
```

---

### 7. Get Locations

Get list of all available locations in the dataset.

**Endpoint:** `GET /api/locations`

**Response:**

```json
{
  "status": "success",
  "locations": ["Nairobi", "Mombasa", "Kisumu", "Nakuru", ...],
  "count": 47
}
```

---

## Example Usage

### Python Example

```python
import requests
import json

BASE_URL = "http://localhost:5000"

# 1. Train models
print("Training models...")
response = requests.post(f"{BASE_URL}/api/train")
print(response.json())

# 2. Make a prediction
print("\nMaking prediction...")
prediction_data = {
    "location": "Nairobi",
    "commodity": "Maize",
    "price_type": "Retail",
    "month": 5,
    "year": 2024
}
response = requests.post(f"{BASE_URL}/api/predict", json=prediction_data)
result = response.json()
print(f"Ensemble Prediction: KES {result['ensemble_prediction']:.2f}")
```

### cURL Example

```bash
# Train models
curl -X POST http://localhost:5000/api/train

# Make prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "location": "Nairobi",
    "commodity": "Maize",
    "price_type": "Retail",
    "month": 5,
    "year": 2024
  }'

# Get commodities
curl http://localhost:5000/api/commodities

# Get locations
curl http://localhost:5000/api/locations
```

### JavaScript/Fetch Example

```javascript
const BASE_URL = "http://localhost:5000";

// Train models
async function trainModels() {
  const response = await fetch(`${BASE_URL}/api/train`, {
    method: "POST",
  });
  const data = await response.json();
  console.log(data);
}

// Make prediction
async function predictPrice() {
  const predictionData = {
    location: "Nairobi",
    commodity: "Maize",
    price_type: "Retail",
    month: 5,
    year: 2024,
  };

  const response = await fetch(`${BASE_URL}/api/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(predictionData),
  });

  const result = await response.json();
  console.log(
    `Ensemble Prediction: KES ${result.ensemble_prediction.toFixed(2)}`,
  );
}

trainModels();
predictPrice();
```

---

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- `200 OK`: Successful request
- `400 Bad Request`: Missing required fields or invalid data
- `500 Internal Server Error`: Server-side error

**Error Response Example:**

```json
{
  "error": "Models not trained yet. Please train models first."
}
```

---

## Running Multiple Servers

You can run both the Streamlit app and Flask API simultaneously:

**Terminal 1 - Streamlit App:**

```bash
streamlit run app.py
```

**Terminal 2 - Flask API:**

```bash
python api.py
```

The Streamlit app will be available at `http://localhost:8501`
The Flask API will be available at `http://localhost:5000`

---

## Performance Notes

- **Model Training**: Takes 2-5 minutes depending on hardware
- **Predictions**: Returns results in < 100ms after models are trained
- **Memory Usage**: Approximately 500MB for all models and data

---

## Future Enhancements

- [ ] Database integration for result persistence
- [ ] User authentication and API keys
- [ ] Rate limiting
- [ ] Model versioning and history
- [ ] Batch prediction endpoint
- [ ] WebSocket support for real-time predictions
- [ ] Docker containerization
