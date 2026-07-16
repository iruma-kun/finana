import unittest
import numpy as np
import pandas as pd
# Import classes to test
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
import pandas as pd
from data_generator import FinancialDataGenerator
from preprocessor import FinancialPreprocessor
from models import VolatilityPredictor
from evaluator import VolatilityEvaluator

class TestFinancialVolatilityPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Generate a small dataset for testing purposes
        cls.generator = FinancialDataGenerator(start_date="2024-01-01", end_date="2024-06-30", random_state=42)
        cls.raw_df = cls.generator.generate()
        
        cls.preprocessor = FinancialPreprocessor(max_features=100, test_size=0.2, val_size=0.2)
        cls.processed = cls.preprocessor.fit_transform(cls.raw_df)

    def test_data_generation(self):
        """Test that data generation works correctly and returns the correct schema."""
        self.assertIsInstance(self.raw_df, pd.DataFrame)
        self.assertFalse(self.raw_df.empty)
        
        # Check required columns
        required_cols = ['date', 'headlines', 'open', 'high', 'low', 'close', 'volume', 'realized_volatility', 'tomorrow_vol_class']
        for col in required_cols:
            self.assertIn(col, self.raw_df.columns)
            
        # Ensure target is binary
        unique_targets = self.raw_df['tomorrow_vol_class'].unique()
        self.assertTrue(all(val in [0, 1] for val in unique_targets))
        
        # Ensure only weekday dates are present
        for date_val in self.raw_df['date']:
            self.assertLess(date_val.weekday(), 5) # 0-4 are Mon-Fri

    def test_text_cleaning(self):
        """Test the text preprocessing clean function."""
        preprocessor = FinancialPreprocessor()
        dirty_text = "SURPRISE! Market crash (99%) looms...  earnings shock!!  "
        clean_text = preprocessor.clean_text(dirty_text)
        
        # Check lowercase, special characters, and multiple spaces removed
        self.assertEqual(clean_text, "surprise market crash looms earnings shock")

    def test_temporal_splitting_and_leakage_prevention(self):
        """Test that splitting preserves chronological order (no future data in train)."""
        train_dates = self.processed['metadata']['train_dates']
        val_dates = self.processed['metadata']['val_dates']
        test_dates = self.processed['metadata']['test_dates']
        
        # Ensure chronological ordering: max(train) < min(val) and max(val) < min(test)
        self.assertLess(train_dates.max(), val_dates.min())
        self.assertLess(val_dates.max(), test_dates.min())
        
        # Ensure feature matrix sizes match targets
        self.assertEqual(self.processed['X_train'].shape[0], len(self.processed['y_train']))
        self.assertEqual(self.processed['X_val'].shape[0], len(self.processed['y_val']))
        self.assertEqual(self.processed['X_test'].shape[0], len(self.processed['y_test']))

    def test_outlier_handling_no_leakage(self):
        """Test that outlier bounds are computed on train and applied correctly without leaking."""
        preprocessor = FinancialPreprocessor()
        
        # Handle outlier in 'high' to ensure volatility variation
        # Construct synthetic dataframe with sufficient variety to avoid NaN
        np.random.seed(42)
        test_df = pd.DataFrame({
            'date': pd.date_range(start="2024-01-01", periods=50, freq="D"),
            'close': np.random.uniform(90, 110, 50),
            'high': np.random.uniform(105, 120, 50),
            'low': np.random.uniform(80, 95, 50),
            'volume': np.random.randint(1000, 5000, 50),
            'tomorrow_vol_class': np.random.randint(0, 2, 50),
            'headlines': ["normal headline"] * 50
        })
        # Add extreme outlier
        test_df.iloc[45:, 2] = 500.0 
        
        test_df['realized_volatility'] = (test_df['high'] - test_df['low']) / test_df['close']
        
        # Engineer features
        df_engineered = preprocessor.engineer_features(test_df)
        df_engineered['cleaned_headlines'] = df_engineered['headlines']
        
        # Separate train (rows 10-30) and validation (rows 35-45)
        train_df = df_engineered.iloc[10:30].copy()
        val_df = df_engineered.iloc[35:45].copy()
        
        # Volatility index check before clipping
        self.assertIn('realized_volatility', preprocessor.numeric_features)
        
        # Fit outlier bounds on training only
        preprocessor.handle_outliers(train_df, fit=True)
        val_processed = preprocessor.handle_outliers(val_df, fit=False)
        
        # Ensure validation outlier was capped based on training boundaries
        train_max_vol = train_df['realized_volatility'].quantile(0.99)
        self.assertLessEqual(val_processed['realized_volatility'].max(), train_max_vol)

    def test_custom_model_injection(self):
        """Test that injecting a custom sklearn model works within the predictor wrapper."""
        from sklearn.svm import SVC
        
        # 1. Initialize custom model
        my_custom_model = SVC(probability=True, kernel='linear')
        
        # 2. Inject into wrapper
        predictor = VolatilityPredictor(model=my_custom_model)
        
        # 3. Fit and predict
        predictor.fit(self.processed['X_train'], self.processed['y_train'], feature_subset="time_series", metadata=self.processed['metadata'])
        preds = predictor.predict(self.processed['X_val'], metadata=self.processed['metadata'])
        
        # 4. Verify functionality
        self.assertEqual(len(preds), len(self.processed['X_val']))
        self.assertIsInstance(predictor.model, SVC)

    def test_evaluator_metrics_and_backtest(self):
        """Test that metric calculations, backtesting, and visualization execute successfully."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 0, 1])
        y_pred = np.array([1, 0, 0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.9, 0.1, 0.4, 0.2, 0.8, 0.7, 0.3, 0.95])
        
        # Check metrics calculation
        metrics = VolatilityEvaluator.calculate_metrics(y_true, y_pred, y_prob)
        self.assertIn('precision', metrics)
        self.assertIn('roc_auc', metrics)
        self.assertGreaterEqual(metrics['precision'], 0.0)
        self.assertLessEqual(metrics['precision'], 1.0)
        self.assertGreaterEqual(metrics['roc_auc'], 0.5) # Since predictions are fairly accurate
        
        # Smoke test visual print ROC
        # This checks that the visual representation function runs without throwing exceptions
        try:
            VolatilityEvaluator.print_ascii_roc_curve(y_true, y_prob, width=20, height=8)
            success = True
        except Exception as e:
            success = False
            print(f"ASCII ROC print failed: {e}")
        self.assertTrue(success)
        
        # Check Risk Management Backtest simulation
        realized_vol = np.random.uniform(0.01, 0.05, len(y_pred))
        backtest = VolatilityEvaluator.simulate_risk_management(y_true, y_pred, realized_vol)
        
        self.assertIn('static_final_value', backtest)
        self.assertIn('managed_final_value', backtest)
        self.assertIn('static_max_drawdown', backtest)
        self.assertIn('managed_max_drawdown', backtest)
        self.assertLessEqual(backtest['static_max_drawdown'], 0.0)
        self.assertLessEqual(backtest['managed_max_drawdown'], 0.0)

if __name__ == '__main__':
    unittest.main()
