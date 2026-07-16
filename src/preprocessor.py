import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any, List

class FinancialPreprocessor:
    """
    Optimized preprocessor for time-series and NLP feature engineering.
    Implements strict temporal splitting to prevent data leakage.
    """
    def __init__(self, text_col='headlines', target_col='tomorrow_vol_class', 
                 max_features=300, test_size=0.15, val_size=0.15):
        self.text_col = text_col
        self.target_col = target_col
        self.max_features = max_features
        self.test_size = test_size
        self.val_size = val_size
        
        # Pre-compile regex for faster text cleaning
        self.clean_re = re.compile(r'[^a-zA-Z\s]')
        self.space_re = re.compile(r'\s+')
        
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True, 
            stop_words='english', 
            max_features=self.max_features,
            token_pattern=r'(?u)\b\w\w+\b'
        )
        self.scaler = StandardScaler()
        self.outlier_bounds = {}
        self.numeric_features = []

    def clean_text(self, text: str) -> str:
        """Optimized text cleaning using pre-compiled regex."""
        if not isinstance(text, str): return ""
        text = self.clean_re.sub('', text.lower().replace('|', ' '))
        return self.space_re.sub(' ', text).strip()

    def handle_outliers(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Handles outliers in numeric columns by capping them at the 1st and 99th percentiles."""
        df = df.copy()
        for col in self.numeric_features:
            if fit:
                self.outlier_bounds[col] = (df[col].quantile(0.01), df[col].quantile(0.99))
            lower, upper = self.outlier_bounds.get(col, (float('-inf'), float('inf')))
            df[col] = np.clip(df[col], lower, upper)
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized feature engineering for high performance."""
        df = df.copy()
        
        # Returns and Volatility Lags
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        for lag in [1, 2, 3]:
            df[f'vol_lag_{lag}'] = df['realized_volatility'].shift(lag)
        
        # Vectorized Rolling Windows
        df['vol_roll_mean_5'] = df['realized_volatility'].rolling(5).mean()
        df['vol_roll_std_5'] = df['realized_volatility'].rolling(5).std()
        df['ret_roll_std_5'] = df['log_return'].rolling(5).std()
        
        self.numeric_features = ['realized_volatility', 'log_return', 'vol_lag_1', 
                                 'vol_lag_2', 'vol_lag_3', 'vol_roll_mean_5', 
                                 'vol_roll_std_5', 'ret_roll_std_5']
        
        # Include optional LLM features if present
        if 'llm_sentiment' in df.columns:
            self.numeric_features.append('llm_sentiment')
            
        return df.dropna(subset=self.numeric_features + [self.target_col])

    def fit_transform(self, raw_df: pd.DataFrame) -> Dict[str, Any]:
        """Orchestrates the full pipeline with optimized memory management."""
        df = self.engineer_features(raw_df)
        df['clean_txt'] = df[self.text_col].apply(self.clean_text)
        
        # Temporal Split
        n = len(df)
        test_idx = int(n * (1 - self.test_size))
        val_idx = int(test_idx * (1 - self.val_size / (1 - self.test_size)))
        
        train, val, test = df.iloc[:val_idx], df.iloc[val_idx:test_idx], df.iloc[test_idx:]
        
        # Scaling & Outlier Handling (In-place on numeric columns for speed)
        X_train_num = self.scaler.fit_transform(train[self.numeric_features])
        X_val_num = self.scaler.transform(val[self.numeric_features])
        X_test_num = self.scaler.transform(test[self.numeric_features])
        
        # TF-IDF
        X_train_txt = self.tfidf_vectorizer.fit_transform(train['clean_txt']).toarray()
        X_val_txt = self.tfidf_vectorizer.transform(val['clean_txt']).toarray()
        X_test_txt = self.tfidf_vectorizer.transform(test['clean_txt']).toarray()
        
        # Combine
        tfidf_cols = [f"tfidf_{w}" for w in self.tfidf_vectorizer.get_feature_names_out()]
        cols = self.numeric_features + tfidf_cols
        
        return {
            'X_train': pd.DataFrame(np.hstack([X_train_num, X_train_txt]), columns=cols),
            'X_val': pd.DataFrame(np.hstack([X_val_num, X_val_txt]), columns=cols),
            'X_test': pd.DataFrame(np.hstack([X_test_num, X_test_txt]), columns=cols),
            'y_train': train[self.target_col].values,
            'y_val': val[self.target_col].values,
            'y_test': test[self.target_col].values,
            'metadata': {
                'train_dates': train['date'].values,
                'val_dates': val['date'].values,
                'test_dates': test['date'].values,
                'numeric_features': self.numeric_features, 
                'tfidf_features': tfidf_cols, 
                'feature_names': cols
            }
        }
