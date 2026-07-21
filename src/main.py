import argparse
import pandas as pd
import numpy as np
import toml
import os
import joblib
from data_generator import FinancialDataGenerator
from preprocessor import FinancialPreprocessor
from models import VolatilityPredictor
from evaluator import VolatilityEvaluator
from cloud_processor import CloudBatchProcessor

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

def run_pipeline():
    config = load_config()
    
    # Priority: Command line args > Config file > Default
    parser = argparse.ArgumentParser(description="Financial Volatility Prediction.")
    parser.add_argument("--model", type=str, default=config["model_settings"].get("model_type", "random_forest"),
                        help="Base model to train.")
    parser.add_argument("--cloud", action="store_true", default=config.get("cloud_settings", {}).get("enabled", False),
                        help="Enable cloud sentiment features.")
    parser.add_argument("--save-model", type=str, default=None,
                        help="Path to save the best trained model (e.g., models/best_model.pkl).")
    args = parser.parse_args()

    model_type = args.model
    use_cloud = args.cloud
    save_model_path = args.save_model

    print("=" * 70)
    print(f"FINANCIAL MARKET VOLATILITY PREDICTION PIPELINE")
    print(f"Model: {model_type.upper()} | Cloud Features: {use_cloud}")
    print("=" * 70)
    
    # 1. Generate Synthetic Financial Data
    print("\n[Step 1] Generating synthetic financial and news dataset...")
    generator = FinancialDataGenerator(start_date="2021-01-01", end_date="2025-12-31", random_state=42)
    raw_data = generator.generate()
    
    # 2. Cloud LLM Feature Augmentation
    if use_cloud:
        print("\n[Step 2] Augmenting with Cloud LLM features...")
        cache_file = config["cloud_settings"].get("cache_file", "data/llm_features.csv")
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        processor = CloudBatchProcessor()
        if not os.path.exists(cache_file):
            print(f"  No cache found. Processing headlines via Cloud API...")
            processor.process_headlines(raw_data, "headlines", cache_file)
        
        llm_features = pd.read_csv(cache_file)
        raw_data = raw_data.merge(llm_features, on="date")
        print(f"  Cloud features loaded from {cache_file}")

    # 3. Robust Preprocessing and Splitting
    print("\n[Step 3] Preprocessing and temporal splitting...")
    # If cloud is enabled, we have an extra feature: 'llm_sentiment'
    preprocessor = FinancialPreprocessor(max_features=500, test_size=0.15, val_size=0.15)
    processed = preprocessor.fit_transform(raw_data)
    
    X_train, y_train = processed['X_train'], processed['y_train']
    X_val, y_val = processed['X_val'], processed['y_val']
    X_test, y_test = processed['X_test'], processed['y_test']
    metadata = processed['metadata']
    
    # 4. Model Training & Validation
    print("\n[Step 4] Training model configurations...")
    configs = ["time_series", "text", "all"]
    config_labels = {
        "time_series": "Time-Series Only Baseline",
        "text": "News Headlines NLP Only Baseline",
        "all": "Multi-Modal Combined"
    }
    
    val_results = {}
    test_results = {}
    trained_predictors = {}
    
    for cfg in configs:
        print(f"  Training {config_labels[cfg]}...")
        predictor = VolatilityPredictor(model_type=model_type, random_state=42)
        predictor.fit(X_train, y_train, feature_subset=cfg, metadata=metadata)
        
        val_probs = predictor.predict_proba(X_val, metadata=metadata)
        val_metrics = VolatilityEvaluator.calculate_metrics(y_val, (val_probs > 0.5).astype(int), val_probs)
        val_results[config_labels[cfg]] = val_metrics
        
        test_probs = predictor.predict_proba(X_test, metadata=metadata)
        test_metrics = VolatilityEvaluator.calculate_metrics(y_test, (test_probs > 0.5).astype(int), test_probs)
        test_results[config_labels[cfg]] = test_metrics
        
        trained_predictors[cfg] = predictor
    
    # 5. Display Comparison
    print("\n--- PERFORMANCE COMPARISON (TEST SET) ---")
    test_comparison = VolatilityEvaluator.compare_models(test_results)
    print(test_comparison.to_string())
    
    # 6. ROC Curve and Risk Backtest for Best Model
    print("\n[Step 5] Best Model ROC Curve (Out-of-Sample):")
    best_predictor = trained_predictors["all"]
    best_test_probs = best_predictor.predict_proba(X_test, metadata=metadata)
    VolatilityEvaluator.print_ascii_roc_curve(y_test, best_test_probs)

    if save_model_path:
        print(f"\n[Step 7] Saving best model and preprocessor...")
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        
        # Save both the predictor and the fitted preprocessor
        save_data = {
            'predictor': best_predictor,
            'preprocessor': preprocessor,
            'metadata': metadata
        }
        joblib.dump(save_data, save_model_path)
        print(f"  Exported to {save_model_path}")
    
    print("\n[Step 6] Risk Management Backtest...")
    best_test_preds = (best_test_probs > 0.5).astype(int)
    test_indices = raw_data.index[len(raw_data) - len(y_test):]
    test_realized_vol = raw_data.loc[test_indices, 'realized_volatility'].values
    
    backtest = VolatilityEvaluator.simulate_risk_management(y_test, best_test_preds, test_realized_vol)
    
    print(f"\nVolatility-Managed Portfolio:")
    print(f"  - Sharpe Ratio improvement: {backtest['managed_sharpe'] - backtest['static_sharpe']:.3f}")
    print(f"  - Max Drawdown reduction: {backtest['static_max_drawdown'] - backtest['managed_max_drawdown']:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()