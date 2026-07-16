import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class FinancialDataGenerator:
    """
    Generates synthetic stock market data and corresponding news headlines
    with embedded signals to simulate predicting volatility from news & time series.
    """
    def __init__(self, start_date="2022-01-01", end_date="2025-12-31", random_state=42):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Vocabularies that trigger high or low volatility
        self.high_vol_phrases = [
            "unexpected market collapse looms",
            "regulatory probe sparks heavy selloff",
            "earnings shock triggers massive panic",
            "central bank announces surprise rate hike",
            "geopolitical tensions escalate rapidly",
            "ceo suddenly resigns amid investigation",
            "supply chain crisis halts production",
            "bankruptcy rumors surrounding major bank",
            "debt default risk rises sharply",
            "inflation surges to record highs"
        ]
        
        self.low_vol_phrases = [
            "steady growth expected this quarter",
            "stable outlook maintained by ratings agency",
            "earnings in line with analyst consensus",
            "routine annual shareholder meeting begins",
            "market stabilizes after gradual recovery",
            "business as usual reports major tech firm",
            "interest rates remain unchanged",
            "solid performance from consumer sector",
            "inflation metrics meet expectations",
            "merger integration proceeds smoothly"
        ]
        
        self.neutral_phrases = [
            "stock trading volume remains average",
            "analysts update quarterly target prices",
            "new product line announced for next year",
            "weather conditions impact local shipping",
            "company schedules upcoming conference call",
            "minor organizational changes announced",
            "industry conference starts next week",
            "executive speaks at technology forum",
            "weekly inventory report released",
            "investors monitor global currency fluctuations"
        ]

    def generate(self) -> pd.DataFrame:
        """
        Generates daily stock prices and synthetic news headlines.
        """
        # Generate date range
        date_list = []
        curr = self.start_date
        while curr <= self.end_date:
            # Only stock trading days (Mon-Fri)
            if curr.weekday() < 5:
                date_list.append(curr)
            curr += timedelta(days=1)
            
        n_days = len(date_list)
        
        # Simulate base stock price (Geometric Brownian Motion)
        initial_price = 100.0
        mu = 0.05 / 252  # daily drift
        sigma_base = 0.15 / np.sqrt(252)  # daily base volatility
        
        prices = [initial_price]
        for i in range(1, n_days):
            # Price change with drift and base volatility
            ret = np.random.normal(mu, sigma_base)
            prices.append(prices[-1] * np.exp(ret))
            
        prices = np.array(prices)
        
        # Create stock DataFrame
        df = pd.DataFrame(index=date_list)
        df['close'] = prices
        df['open'] = prices * np.random.normal(1.0, 0.002, n_days)
        
        # Volatility state: 0 = Low/Normal, 1 = High
        # Volatility state transitions as a Markov chain with news influence
        vol_state = np.zeros(n_days)
        headlines = []
        
        state = 0  # initial state: normal
        for i in range(n_days):
            # Decide news sentiment for today
            # Headline content affects state transitions or state triggers volatility
            headline_pool = []
            
            # Transition probability dependent on current state
            if state == 0:
                p_transition = 0.15  # 15% chance to go to high volatility
            else:
                p_transition = 0.40  # 40% chance to return to normal volatility
                
            if np.random.rand() < p_transition:
                state = 1 - state
                
            vol_state[i] = state
            
            # Form headlines matching the volatility state
            num_headlines = np.random.randint(1, 4)
            chosen_headlines = []
            for _ in range(num_headlines):
                rand_val = np.random.rand()
                if state == 1:
                    # High volatility days: 70% chance of high vol headlines, 30% neutral
                    if rand_val < 0.7:
                        chosen_headlines.append(np.random.choice(self.high_vol_phrases))
                    else:
                        chosen_headlines.append(np.random.choice(self.neutral_phrases))
                else:
                    # Low volatility days: 70% chance of low vol headlines, 30% neutral
                    if rand_val < 0.7:
                        chosen_headlines.append(np.random.choice(self.low_vol_phrases))
                    else:
                        chosen_headlines.append(np.random.choice(self.neutral_phrases))
                        
            headlines.append(" | ".join(chosen_headlines))
            
        df['vol_state'] = vol_state
        df['headlines'] = headlines
        
        # Calculate high/low matching the volatility state
        # High volatility days have larger high-low spreads
        spread_multiplier = np.where(df['vol_state'] == 1, np.random.uniform(2.5, 5.0, n_days), np.random.uniform(1.0, 2.0, n_days))
        daily_vol = sigma_base * spread_multiplier
        
        df['high'] = df[['open', 'close']].max(axis=1) * (1.0 + np.abs(np.random.normal(daily_vol, daily_vol * 0.1)))
        df['low'] = df[['open', 'close']].min(axis=1) * (1.0 - np.abs(np.random.normal(daily_vol, daily_vol * 0.1)))
        df['volume'] = np.random.randint(50000, 2000000, n_days) * (1.0 + df['vol_state'] * np.random.uniform(0.5, 1.5, n_days))
        
        # Target variable: High Volatility TOMORROW (1 if daily range > median daily range, 0 otherwise)
        # Realized volatility today can be measured by Parkingson Volatility or simple Normalized Range: (High - Low) / Close
        df['realized_volatility'] = (df['high'] - df['low']) / df['close']
        
        # Shift target to represent TOMORROW's volatility class (binary classification)
        # We classify next day's realized volatility into High (1) or Low (0) based on median threshold
        median_vol = df['realized_volatility'].median()
        df['tomorrow_vol_class'] = (df['realized_volatility'].shift(-1) > median_vol).astype(int)
        
        # Drop the last row since tomorrow_vol_class is NaN
        df = df.dropna()
        
        df = df.reset_index().rename(columns={'index': 'date'})
        return df

if __name__ == "__main__":
    generator = FinancialDataGenerator()
    data = generator.generate()
    print(f"Generated {len(data)} rows of financial news and market data.")
    print(data[['date', 'headlines', 'close', 'realized_volatility', 'tomorrow_vol_class']].head())
