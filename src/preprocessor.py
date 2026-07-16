import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any, List

class FinancialPreprocessor:
    """
    Handles robust time-series and NLP feature engineering, temporal splitting,
    outlier handling, scaling, and TF-IDF vectorization to avoid data leakage.
    """
    def __init__(self, text_col='headlines', target_col='tomorrow_vol_class', 
                 max_features=500, test_size=0.2, val_size=0.2):
        self.text_col = text_col
        self.target_col = target_col
        self.max_features = max_features
        self.test_size = test_size
        self.val_size = val_size
        
        # Objects to be fit on train and applied to validation/test
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True, 
            stop_words='english', 
            max_features=self.max_features,
            token_pattern=r'(?u)\b\w\w+\b'
        )
        self.scaler = StandardScaler()
        self.outlier_bounds = {}  # {feature_name: (lower_bound, upper_bound)}
        
        # Numeric features list
        self.numeric_features = []

    def clean_text(self, text: str) -> str:
        """
        Cleans headlines by removing special characters, numbers, and multiple spaces.
        """
        if not isinstance(text, str):
            return ""
        # Lowercase and remove non-alphabetic chars
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s|]', '', text)  # Keep the pipe '|' as it separates headlines if needed, but we'll remove it too
        text = text.replace('|', ' ')
        # Remove single characters and extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def engineer_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates time series features like log returns, lag realized volatility, 
        and rolling standard deviations.
        """
        df = df.copy()
        
        # 1. Log returns of closing price
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # 2. Lagged realized volatility (1, 2, 3 day lags)
        df['realized_vol_lag_1'] = df['realized_volatility'].shift(1)
        df['realized_vol_lag_2'] = df['realized_volatility'].shift(2)
        df['realized_vol_lag_3'] = df['realized_volatility'].shift(3)
        
        # 3. Rolling realized volatility (5-day and 10-day moving averages and std)
        df['realized_vol_roll_mean_5'] = df['realized_volatility'].rolling(window=5).mean()
        df['realized_vol_roll_mean_10'] = df['realized_volatility'].rolling(window=10).mean()
        df['realized_vol_roll_std_5'] = df['realized_volatility'].rolling(window=5).std()
        
        # 4. Rolling stock returns volatility (historical volatility)
        df['return_roll_std_5'] = df['log_return'].rolling(window=5).std()
        df['return_roll_std_10'] = df['log_return'].rolling(window=10).std()
        
        # Define the numeric features list
        self.numeric_features = [
            'realized_volatility', 'log_return', 
            'realized_vol_lag_1', 'realized_vol_lag_2', 'realized_vol_lag_3',
            'realized_vol_roll_mean_5', 'realized_vol_roll_mean_10', 'realized_vol_roll_std_5',
            'return_roll_std_5', 'return_roll_std_10'
        ]
        
        # Clean up any NaNs created by lagging/rolling operations
        df = df.dropna(subset=self.numeric_features + [self.target_col])
        return df

    def temporal_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Performs a temporal split (train, validation, test) to preserve chronological order
        and prevent data leakage.
        """
        # Ensure df is sorted chronologically
        df = df.sort_values('date').reset_index(drop=True)
        
        n = len(df)
        test_start_idx = int(n * (1 - self.test_size))
        val_start_idx = int(test_start_idx * (1 - self.val_size / (1 - self.test_size)))
        
        train_df = df.iloc[:val_start_idx].copy()
        val_df = df.iloc[val_start_idx:test_start_idx].copy()
        test_df = df.iloc[test_start_idx:].copy()
        
        return train_df, val_df, test_df

    def handle_outliers(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Handles outliers in numeric columns by capping them at the 1st and 99th percentiles
        derived strictly from the training dataset.
        """
        df = df.copy()
        for col in self.numeric_features:
            if fit:
                # Calculate percentiles only on training data
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                self.outlier_bounds[col] = (lower, upper)
            
            # Retrieve bounds (fall back to -inf, inf if not fit)
            lower, upper = self.outlier_bounds.get(col, (float('-inf'), float('inf')))
            df[col] = np.clip(df[col], lower, upper)
            
        return df

    def fit_transform(self, raw_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Orchestrates full feature engineering, split, outlier handling, scaling,
        and TF-IDF vectorization. Returns preprocessed features and targets.
        """
        # 1. Feature Engineering (lags, rollings, logs)
        df_engineered = self.engineer_time_series_features(raw_df)
        
        # 2. Clean headlines
        df_engineered['cleaned_headlines'] = df_engineered[self.text_col].apply(self.clean_text)
        
        # 3. Temporal Split
        train_df, val_df, test_df = self.temporal_split(df_engineered)
        
        # 4. Outlier Handling (Fit on train, apply to all)
        train_df = self.handle_outliers(train_df, fit=True)
        val_df = self.handle_outliers(val_df, fit=False)
        test_df = self.handle_outliers(test_df, fit=False)
        
        # 5. Fit & Transform Scaling for Numeric Features
        train_numeric = self.scaler.fit_transform(train_df[self.numeric_features])
        val_numeric = self.scaler.transform(val_df[self.numeric_features])
        test_numeric = self.scaler.transform(test_df[self.numeric_features])
        
        # Convert scale matrices to DataFrames for easy combining
        train_numeric_df = pd.DataFrame(train_numeric, columns=self.numeric_features, index=train_df.index)
        val_numeric_df = pd.DataFrame(val_numeric, columns=self.numeric_features, index=val_df.index)
        test_numeric_df = pd.DataFrame(test_numeric, columns=self.numeric_features, index=test_df.index)
        
        # 6. Fit & Transform TF-IDF for Text Features
        train_text_sparse = self.tfidf_vectorizer.fit_transform(train_df['cleaned_headlines'])
        val_text_sparse = self.tfidf_vectorizer.transform(val_df['cleaned_headlines'])
        test_text_sparse = self.tfidf_vectorizer.transform(test_df['cleaned_headlines'])
        
        # Convert sparse text vectors to dense DataFrames
        tfidf_cols = [f"tfidf_{w}" for w in self.tfidf_vectorizer.get_feature_names_out()]
        train_text_df = pd.DataFrame(train_text_sparse.toarray(), columns=tfidf_cols, index=train_df.index)
        val_text_df = pd.DataFrame(val_text_sparse.toarray(), columns=tfidf_cols, index=val_df.index)
        test_text_df = pd.DataFrame(test_text_sparse.toarray(), columns=tfidf_cols, index=test_df.index)
        
        # 7. Concatenate Numeric & Text Features
        X_train = pd.concat([train_numeric_df, train_text_df], axis=1)
        X_val = pd.concat([val_numeric_df, val_text_df], axis=1)
        X_test = pd.concat([test_numeric_df, test_text_df], axis=1)
        
        # Extract Targets
        y_train = train_df[self.target_col].values
        y_val = val_df[self.target_col].values
        y_test = test_df[self.target_col].values
        
        # Keep track of dates and original data for backtesting / evaluation
        metadata = {
            'train_dates': train_df['date'].values,
            'val_dates': val_df['date'].values,
            'test_dates': test_df['date'].values,
            'feature_names': list(X_train.columns),
            'tfidf_features': tfidf_cols,
            'numeric_features': self.numeric_features
        }
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'metadata': metadata
        }

if __name__ == "__main__":
    from data_generator import FinancialDataGenerator
    generator = FinancialDataGenerator()
    raw_data = generator.generate()
    
    preprocessor = FinancialPreprocessor()
    processed = preprocessor.fit_transform(raw_data)
    
    print("Preprocessed Data Summary:")
    print(f"X_train shape: {processed['X_train'].shape}, y_train shape: {processed['y_train'].shape}")
    print(f"X_val shape: {processed['X_val'].shape}, y_val shape: {processed['y_val'].shape}")
    print(f"X_test shape: {processed['X_test'].shape}, y_test shape: {processed['y_test'].shape}")
    print("First few columns of X_train:")
    print(processed['X_train'].columns[:15].tolist())
