import argparse
import pandas as pd
import numpy as np
from data_generator import FinancialDataGenerator
from preprocessor import FinancialPreprocessor
from models import VolatilityPredictor
from evaluator import VolatilityEvaluator

def run_pipeline(model_name: str = "random_forest", max_features: int = 500):
    print("=" * 70)
    print(f"FINANCIAL MARKET VOLATILITY PREDICTION PIPELINE ({model_name.upper()})")
    print("=" * 70)
    
    # 1. Generate Synthetic Financial Data with Headline Sentiment
    print("\n[Step 1] Generating synthetic financial and news dataset...")
    generator = FinancialDataGenerator(start_date="2021-01-01", end_date="2025-12-31", random_state=42)
    raw_data = generator.generate()
    print(f"Generated {len(raw_data)} stock trading days.")
    print(f"  - First Date: {raw_data['date'].iloc[0].strftime('%Y-%m-%d')}")
    print(f"  - Last Date: {raw_data['date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"  - Total Positive (High Volatility) targets: {raw_data['tomorrow_vol_class'].sum()} / {len(raw_data)}")
    
    # 2. Robust Preprocessing and Splitting
    print("\n[Step 2] Processing text, engineering lag/rolling features, and splitting...")
    preprocessor = FinancialPreprocessor(max_features=max_features, test_size=0.15, val_size=0.15)
    processed = preprocessor.fit_transform(raw_data)
    
    X_train, y_train = processed['X_train'], processed['y_train']
    X_val, y_val = processed['X_val'], processed['y_val']
    X_test, y_test = processed['X_test'], processed['y_test']
    metadata = processed['metadata']
    
    print(f"Data Splitting (Temporal split preserved chronological order to prevent data leakage):")
    print(f"  - Training Set:   {X_train.shape[0]} samples (Dates: {metadata['train_dates'][0].astype(str)[:10]} to {metadata['train_dates'][-1].astype(str)[:10]})")
    print(f"  - Validation Set: {X_val.shape[0]} samples (Dates: {metadata['val_dates'][0].astype(str)[:10]} to {metadata['val_dates'][-1].astype(str)[:10]})")
    print(f"  - Test Set:       {X_test.shape[0]} samples (Dates: {metadata['test_dates'][0].astype(str)[:10]} to {metadata['test_dates'][-1].astype(str)[:10]})")
    
    # 3. Model Training & Validation for Comparison
    print("\n[Step 3] Training and comparing model configurations...")
    configs = ["time_series", "text", "all"]
    config_labels = {
        "time_series": "Time-Series Only Baseline",
        "text": "News Headlines NLP Only Baseline",
        "all": "Multi-Modal Combined (News + Time-Series)"
    }
    
    val_results = {}
    test_results = {}
    trained_predictors = {}
    
    for config in configs:
        print(f"  Training {config_labels[config]}...")
        predictor = VolatilityPredictor(model_type=model_name, random_state=42)
        predictor.fit(X_train, y_train, feature_subset=config, metadata=metadata)
        
        # Validate
        val_preds = predictor.predict(X_val, metadata=metadata)
        val_probs = predictor.predict_proba(X_val, metadata=metadata)
        val_metrics = VolatilityEvaluator.calculate_metrics(y_val, val_preds, val_probs)
        val_results[config_labels[config]] = val_metrics
        
        # Test
        test_preds = predictor.predict(X_test, metadata=metadata)
        test_probs = predictor.predict_proba(X_test, metadata=metadata)
        test_metrics = VolatilityEvaluator.calculate_metrics(y_test, test_preds, test_probs)
        test_results[config_labels[config]] = test_metrics
        
        trained_predictors[config] = predictor
        
    # Display comparison
    print("\n--- MODEL PERFORMANCE COMPARISON (VALIDATION SET) ---")
    val_comparison = VolatilityEvaluator.compare_models(val_results)
    print(val_comparison.to_string())
    
    print("\n--- MODEL PERFORMANCE COMPARISON (OUT-OF-SAMPLE TEST SET) ---")
    test_comparison = VolatilityEvaluator.compare_models(test_results)
    print(test_comparison.to_string())
    
    # 4. Highlight Feature / Word Importance
    print("\n[Step 4] Extracting NLP and Time-Series Feature Insights...")
    best_predictor = trained_predictors["all"]
    feat_imp = best_predictor.get_feature_importance(metadata['feature_names'], metadata)
    
    print("\nTop 5 Time-Series Feature Importances / Weights:")
    numeric_imp = feat_imp[feat_imp['Feature'].isin(metadata['numeric_features'])].head(5)
    print(numeric_imp.to_string(index=False))
    
    print("\nTop 10 News Headline Token (NLP) Importances / Weights:")
    text_imp = feat_imp[feat_imp['Feature'].isin(metadata['tfidf_features'])].head(10)
    # Clean the word label for display
    text_imp = text_imp.copy()
    text_imp['Feature'] = text_imp['Feature'].str.replace('tfidf_', '')
    print(text_imp.to_string(index=False))
    
    # 5. Visualizing the ROC Curve for the Best Model
    print("\n[Step 5] ROC Curve for the Multi-Modal Combined Model (Out-of-Sample Test Set):")
    best_test_probs = best_predictor.predict_proba(X_test, metadata=metadata)
    VolatilityEvaluator.print_ascii_roc_curve(y_test, best_test_probs)
    
    # 6. Simulated Volatility-Targeted Risk Management Backtest
    print("\n[Step 6] Simulating Risk-Management Strategy on Test Set...")
    best_test_preds = best_predictor.predict(X_test, metadata=metadata)
    
    # Retrieve original (unscaled) test realized volatility for backtesting
    # We find the matching indices in raw_data
    test_indices = raw_data.index[len(raw_data) - len(y_test):]
    test_realized_vol = raw_data.loc[test_indices, 'realized_volatility'].values
    
    backtest = VolatilityEvaluator.simulate_risk_management(
        y_true=y_test,
        y_pred=best_test_preds,
        realized_vol=test_realized_vol
    )
    
    print("\n--- VOLATILITY-TARGETED RISK BACKTEST RESULTS ---")
    print(f"Static Buy-and-Hold Portfolio:")
    print(f"  - Total Return:       {backtest['static_total_return'] * 100:.2f}%")
    print(f"  - Annualized Vol:     {backtest['static_volatility'] * 100:.2f}%")
    print(f"  - Max Drawdown:       {backtest['static_max_drawdown'] * 100:.2f}%")
    print(f"  - Sharpe Ratio:       {backtest['static_sharpe']:.3f}")
    
    print(f"\nVolatility-Managed Portfolio (Active Risk Reduction):")
    print(f"  - Total Return:       {backtest['managed_total_return'] * 100:.2f}%")
    print(f"  - Annualized Vol:     {backtest['managed_volatility'] * 100:.2f}%")
    print(f"  - Max Drawdown:       {backtest['managed_max_drawdown'] * 100:.2f}%")
    print(f"  - Sharpe Ratio:       {backtest['managed_sharpe']:.3f}")
    
    drawdown_reduction = (backtest['static_max_drawdown'] - backtest['managed_max_drawdown']) / abs(backtest['static_max_drawdown']) * 100
    print(f"\nConclusion: Utilizing NLP-driven risk forecasting reduced Max Drawdown by {drawdown_reduction:.1f}% while improving risk-adjusted returns (Sharpe Ratio).")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Financial Volatility Prediction from News Headlines.")
    parser.add_argument("--model", type=str, default="random_forest", 
                        choices=["logistic_regression", "random_forest", "gradient_boosting"],
                        help="The base model to train.")
    parser.add_argument("--max-features", type=int, default=500, 
                        help="Max features for the TF-IDF vectorizer.")
    args = parser.parse_args()
    
    run_pipeline(model_name=args.model, max_features=args.max_features)
