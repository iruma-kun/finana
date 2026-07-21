import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import toml

# Assuming these are in the same src directory
from data_generator import FinancialDataGenerator
from preprocessor import FinancialPreprocessor
from models import VolatilityPredictor

def load_config():
    """Loads configuration from config.toml or returns defaults."""
    loaded_config = {}
    if os.path.exists("config.toml"):
        try:
            loaded_config = toml.load("config.toml")
        except Exception as e:
            print(f"Error loading config.toml: {e}. Using defaults for missing sections.")

    # Merge loaded config with defaults to ensure all keys exist
    default_config = {
        "cloud_settings": {"enabled": False, "cache_file": "data/llm_features.csv", "provider": "openai"},
        "model_settings": {"model_type": "random_forest"},
        "live_settings": {"model_path": "models/best_volatility_predictor.pkl", "host": "127.0.0.1", "port": 5000}
    }

    config = {}
    for section, defaults in default_config.items():
        config[section] = {**defaults, **loaded_config.get(section, {})}

    return config

def get_next_day_prediction(model_path: str = None, market: str = "Global") -> tuple[float, int]:
    """
    Loads pre-trained model and pre-fitted preprocessor, then generates a prediction.
    Accepts a 'market' parameter for future multi-market data routing.
    """
    config = load_config()
    if model_path is None:
        model_path = config["live_settings"].get("model_path", "models/best_volatility_predictor.pkl")

    if not os.path.exists(model_path):
        print(f"Error: No model found at {model_path}.")
        return 0.5, 0

    print(f"[Live Predictor] Fetching prediction for {market}...")
    save_data = joblib.load(model_path)
    predictor = save_data['predictor']
    preprocessor = save_data['preprocessor']
    train_metadata = save_data['metadata']

    # Simulate 'today's' market and news for the specific market requested
    # In a real-world scenario, the generator or an API would fetch data for 'market'
    start_date = datetime.now() - timedelta(days=30)
    generator = FinancialDataGenerator(start_date=start_date.strftime("%Y-%m-%d"), 
                                       end_date=datetime.now().strftime("%Y-%m-%d"),
                                       random_state=None)
    recent_raw_df = generator.generate()

    if recent_raw_df.empty:
        return 0.5, 0
    
    # 1. Engineer features locally using the same rules (lags/rolling windows)
    current_features_df = preprocessor.engineer_features(recent_raw_df)
    current_features_df['clean_txt'] = current_features_df['headlines'].apply(preprocessor.clean_text)
    
    # Use the last available row for tomorrow's prediction
    today_row = current_features_df.tail(1)

    # 2. Transform numeric and text features using the PRE-FITTED preprocessor
    # This ensures feature names and scaling exactly match the training set.
    X_num = preprocessor.scaler.transform(today_row[preprocessor.numeric_features])
    X_txt = preprocessor.tfidf_vectorizer.transform(today_row['clean_txt']).toarray()
    
    # 3. Combine and Predict
    X_live = pd.DataFrame(np.hstack([X_num, X_txt]), columns=train_metadata['feature_names'])
    
    prediction_prob = predictor.predict_proba(X_live, metadata=train_metadata)[0]
    prediction_class = int(prediction_prob > 0.5)

    print(f"[Live Predictor] Prediction for tomorrow: Prob={prediction_prob:.2f}, Class={prediction_class}")
    return float(prediction_prob), prediction_class

if __name__ == "__main__":
    # To run this directly, first train and save a model:
    # python3 src/main.py --save-model models/best_volatility_predictor.pkl
    prob, cls = get_next_day_prediction()
    print(f"Direct Run - Predicted Probability: {prob:.2f}, Class: {cls}")