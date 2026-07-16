from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
import numpy as np
from typing import Dict, Any, List

class VolatilityPredictor:
    """
    Wrapper for financial forecasting models. 
    Supports local sklearn models and custom model injection.
    """
    def __init__(self, model=None, model_type: str = "random_forest", random_state: int = 42):
        if model:
            self.model = model
        else:
            params = {'random_state': random_state, 'class_weight': 'balanced'}
            models = {
                "logistic_regression": LogisticRegression(max_iter=1000, C=0.1, **params),
                "random_forest": RandomForestClassifier(n_estimators=100, max_depth=6, n_jobs=-1, **params),
                "gradient_boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=random_state)
            }
            self.model = models.get(model_type, models["random_forest"])
        self.feature_subset = "all"

    def fit(self, X: pd.DataFrame, y: np.ndarray, feature_subset: str = "all", metadata: Dict = None):
        self.feature_subset = feature_subset
        X_f = self._filter(X, feature_subset, metadata)
        self.model.fit(X_f, y)
        return self

    def predict(self, X: pd.DataFrame, metadata: Dict = None) -> np.ndarray:
        X_f = self._filter(X, self.feature_subset, metadata)
        return self.model.predict(X_f)

    def predict_proba(self, X: pd.DataFrame, metadata: Dict = None) -> np.ndarray:
        X_f = self._filter(X, self.feature_subset, metadata)
        return self.model.predict_proba(X_f)[:, 1]

    def _filter(self, X: pd.DataFrame, subset: str, meta: Dict) -> pd.DataFrame:
        if subset == "all": return X
        cols = meta['numeric_features'] if subset == "time_series" else meta['tfidf_features']
        return X[[c for c in X.columns if c in cols]]

    def get_feature_importance(self, meta: Dict) -> pd.DataFrame:
        if hasattr(self.model, "feature_importances_"):
            imp = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            imp = self.model.coef_[0]
        else: return pd.DataFrame()
        
        return pd.DataFrame({'Feature': meta['feature_names'], 'Importance': imp}).sort_values('Importance', ascending=False)
