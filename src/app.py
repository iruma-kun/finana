from flask import Flask, render_template, jsonify
import os
import toml
from datetime import datetime

# Import live prediction logic
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from predict_live import get_next_day_prediction, load_config

app = Flask(__name__, 
            template_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '../templates')),
            static_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '../static')))

@app.route('/')
def index():
    """Serves the main web dashboard."""
    return render_template('index.html')

@app.route('/api/volatility')
def get_volatility():
    """Exposes real-time next-day volatility prediction as JSON."""
    from flask import request
    market = request.args.get('market', 'Global')
    
    try:
        config = load_config()
        model_path = config["live_settings"].get("model_path", "models/best_volatility_predictor.pkl")
        
        # Check if the model is trained yet
        if not os.path.exists(model_path):
            return jsonify({
                'status': 'error',
                'message': f"No trained model found at '{model_path}'. Please run 'python3 src/main.py --save-model {model_path}' first."
            }), 404
            
        prob, cls = get_next_day_prediction(model_path, market=market)
        
        # Simulate market-specific news headlines
        market_headlines = {
            'SPX': [
                "S&P 500 faces resistance at key technical levels",
                "Broad market sentiment remains cautious ahead of retail data",
                "Energy sector pull-back weighs on index performance"
            ],
            'NDX': [
                "Tech giants show resilience amid regulatory scrutiny",
                "Semiconductor demand forecast boosts NASDAQ outlook",
                "Innovation index tracks steady growth in AI investments"
            ],
            'Global': [
                "Central bank hints at surprise economic policy adjustments",
                "Tech earnings beat analyst forecasts but raise outlook concerns",
                "Geopolitical uncertainty prompts steady market volume"
            ]
        }
        
        headlines = market_headlines.get(market, [
            f"Analyzing risk factors for {market}...",
            f"Institutional volume in {market} remains within normal range",
            f"Market participants monitor {market} for volatility triggers"
        ])
        
        return jsonify({
            'status': 'success',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'market': market,
            'next_day_prediction_prob': prob,
            'next_day_prediction_class': cls,
            'volatility_status': "HIGH RISK" if cls == 1 else "NORMAL/STABLE",
            'used_headlines': headlines
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    config = load_config()
    host = config.get("live_settings", {}).get("host", "127.0.0.1")
    port = config.get("live_settings", {}).get("port", 5000)
    
    print("=" * 70)
    print(f"Starting Finana Live Dashboard on http://{host}:{port}")
    print("=" * 70)
    app.run(host=host, port=port, debug=True)
