from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List

class VolatilityPredictor:
    """
    Trains and handles machine learning models to forecast next-day financial market volatility.
    Supports Time-Series only, Text only, and Combined multi-modal modeling configurations.
    """
    def __init__(self, model_type: str = "random_forest", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        
        # Initialize selected base model
        if model_type == "logistic_regression":
            self.model = LogisticRegression(
                max_iter=1000, 
                class_weight='balanced', 
                random_state=random_state,
                C=0.1  # Stronger L2 regularization to prevent overfitting on TF-IDF features
            )
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=150,
                max_depth=6,
                min_samples_split=5,
                class_weight='balanced',
                random_state=random_state,
                n_jobs=-1
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Choose from 'logistic_regression', 'random_forest', 'gradient_boosting'.")

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, feature_subset: str = "all", metadata: Dict[str, Any] = None):
        """
        Fits the model on the specified subset of features:
        - "all": Both time series and TF-IDF features.
        - "time_series": Only numeric time series features.
        - "text": Only TF-IDF features.
        """
        self.feature_subset = feature_subset
        X_train_filtered = self._filter_features(X_train, feature_subset, metadata)
        self.model.fit(X_train_filtered, y_train)
        return self

    def predict(self, X: pd.DataFrame, metadata: Dict[str, Any] = None) -> np.ndarray:
        """
        Predicts binary outcomes (1 for high volatility, 0 for normal).
        """
        X_filtered = self._filter_features(X, self.feature_subset, metadata)
        return self.model.predict(X_filtered)

    def predict_proba(self, X: pd.DataFrame, metadata: Dict[str, Any] = None) -> np.ndarray:
        """
        Predicts probabilities for class membership.
        """
        X_filtered = self._filter_features(X, self.feature_subset, metadata)
        return self.model.predict_proba(X_filtered)[:, 1]

    def _filter_features(self, X: pd.DataFrame, feature_subset: str, metadata: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Filters input dataframe according to the designated subset mode.
        """
        if feature_subset == "all":
            return X
        
        if metadata is None:
            raise ValueError("Metadata is required when filtering for specific feature subsets ('time_series' or 'text').")
            
        if feature_subset == "time_series":
            cols = [col for col in X.columns if col in metadata['numeric_features']]
            return X[cols]
            
        if feature_subset == "text":
            cols = [col for col in X.columns if col in metadata['tfidf_features']]
            return X[cols]
            
        raise ValueError(f"Invalid feature_subset: {feature_subset}. Choose 'all', 'time_series', or 'text'.")

    def get_feature_importance(self, feature_names: List[str], metadata: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Extracts coefficients or feature importances and returns a sorted DataFrame.
        """
        # Get actual features used
        if self.feature_subset == "time_series":
            cols = [col for col in feature_names if col in metadata['numeric_features']]
        elif self.feature_subset == "text":
            cols = [col for col in feature_names if col in metadata['tfidf_features']]
        else:
            cols = feature_names
            
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            metric_name = "Importance"
        elif hasattr(self.model, "coef_"):
            importances = self.model.coef_[0]
            metric_name = "Coefficient"
        else:
            return pd.DataFrame() # No importances available
            
        df = pd.DataFrame({
            'Feature': cols,
            metric_name: importances
        })
        
        # Absolute sorting for coefficients to get top impactors
        if metric_name == "Coefficient":
            df['Abs_Coefficient'] = df[metric_name].abs()
            df = df.sort_values(by='Abs_Coefficient', ascending=False).drop(columns=['Abs_Coefficient'])
        else:
            df = df.sort_values(by=metric_name, ascending=False)
            
        return df.reset_index(drop=True)

if __name__ == "__main__":
    # Smoke test
    from data_generator import FinancialDataGenerator
    from preprocessor import FinancialPreprocessor
    
    gen = FinancialDataGenerator()
    data = gen.generate()
    
    prep = FinancialPreprocessor()
    split_data = prep.fit_transform(data)
    
    predictor = VolatilityPredictor(model_type="random_forest")
    predictor.fit(split_data['X_train'], split_data['y_train'], feature_subset="all")
    preds = predictor.predict(split_data['X_val'])
    
    print("Model fit successfully.")
    print(f"Num predictions: {len(preds)}, unique classes predicted: {np.unique(preds)}")
    
    # Feature importance
    feat_imp = predictor.get_feature_importance(split_data['metadata']['feature_names'], split_data['metadata'])
    print("\nTop 10 Feature Importances:")
    print(feat_imp.head(10))
