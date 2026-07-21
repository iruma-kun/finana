# Finana: Financial Volatility Forecasting
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Flask](https://img.shields.io/badge/framework-flask-000000.svg)
![scikit-learn](https://img.shields.io/badge/library-scikit_learn-F7931E.svg)
![GitHub issues](https://img.shields.io/github/issues/iruma-kun/finana)
![GitHub stars](https://img.shields.io/github/stars/iruma-kun/finana?style=social)



A professional-grade machine learning pipeline for predicting next-day financial market volatility by combining quantitative time-series data with news sentiment analysis.

This project features an interactive **live dashboard** designed with a **Zen / Japandi dark aesthetic**—providing a clean, minimalist, and clutter-free interface inspired by the Zen Browser.

## Key Features
- **Multi-Modal Learning**: Seamlessly blends stock price dynamics (OHLCV) with NLP features (TF-IDF or lexicon-based sentiment).
- **Zen / Japandi UI**: Zero-gradient, high-contrast minimalist interface tailored for real-time risk visualization.
- **Market Selection & Search**: Quick-toggle buttons for major indices (S&P 500, NASDAQ, Dow Jones, DAX) and a lookup search bar for any custom stock ticker or market asset.
- **Configuration-First**: Manage models and settings via `config.toml`.
- **Leakage-Free**: Strict temporal splitting ensures no look-ahead bias during training.
- **Custom Model Injection**: Support for any scikit-learn compatible estimator.
- **Backtesting Suite**: Simulates risk-managed portfolio performance based on volatility signals.

## Installation
```bash
git clone https://github.com/iruma-kun/finana.git
cd finana
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Architecture
```mermaid
graph TD
    A[Data Generator] --> B[Preprocessor]
    B --> C{Temporal Split}
    C -->|Training Set| D[Model Training]
    D --> E[Saved Model & Preprocessor]
    E --> F[Predict Live Server]
    F --> G[Web UI (Frontend)]
    C -->|Validation/Test Set| H[Model Evaluation]
    H --> I[Risk Management Backtest]
    I --> J[Performance Metrics]
```

## Usage Guide

### 1. Training and Saving Your Model
Before running the live dashboard, you need to train a model and save it. This will also run the full pipeline and print performance metrics.

```bash
# Run the main pipeline and save the best combined model
# The saved file includes the trained predictor and preprocessor
python3 src/main.py --save-model models/best_volatility_predictor.pkl
```

### 2. Configuring Cloud LLM Features (Optional)
To integrate advanced sentiment features from cloud LLMs (e.g., OpenAI, Anthropic, Google), edit `config.toml`:

```toml
# config.toml
[cloud_settings]
enabled = true
api_key = "your_api_key_here" # Replace with your actual API key
# provider = "openai" # Options: openai, anthropic, google
cache_file = "data/llm_features.csv"
```

Note: Running `src/main.py` with `cloud_settings.enabled = true` will trigger API calls and cache results. Ensure your API key is configured.

### 3. Running the Live Volatility Dashboard
Once your model is saved, you can launch the interactive web dashboard:

```bash
# In your terminal, activate venv and run:
python3 src/app.py
```

After running the command, open your web browser and navigate to the address shown in the terminal (e.g., `http://127.0.0.1:5000`).

### 4. Custom Model Integration
You can plug in any scikit-learn compatible model to compare against the defaults. Simply instantiate your model and pass it to the `VolatilityPredictor`:

```python
from sklearn.svm import SVC
from src.models import VolatilityPredictor

# Instantiate your custom model
my_custom_model = SVC(probability=True, kernel='linear')

# Then, within your script (e.g., main.py for a custom run):
predictor = VolatilityPredictor(model=my_custom_model)
predictor.fit(X_train, y_train, feature_subset="all", metadata=metadata)
```

## Configuration (`config.toml`)

```toml
# Configuration for Finana Volatility Pipeline

[model_settings]
# Options: random_forest, gradient_boosting, logistic_regression, or 'custom'
model_type = "random_forest"

[live_settings]
# Path where the trained model is saved for the web dashboard
model_path = "models/best_volatility_predictor.pkl"
# Host and Port for the local Flask server
host = "127.0.0.1"
port = 5000

[cloud_settings]
# Set to true to use cloud sentiment features
enabled = false
# Options: openai, anthropic, google
provider = "openai"
api_key = "your_api_key_here"
# Cache file to save API results. Ensure 'data/' directory exists.
cache_file = "data/llm_features.csv"
```

## Developer Notes
- **Outlier Handling**: Robust scaling and percentile capping prevent extreme market moves from biasing the model.
- **Memory Optimization**: Vectorized pandas operations and pre-compiled regex ensure fast runtimes.
- **Feature Consistency**: The `predict_live.py` script ensures that live predictions use the exact same preprocessing steps and feature set as the training phase, preventing feature mismatch errors.

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/awesome-feature`).
3. Commit your changes (`git commit -am 'Add awesome feature'`).
4. Push to the branch (`git push origin feature/awesome-feature`).
5. Open a Pull Request.

Please follow the existing code style (PEP 8) and add tests for new functionality.