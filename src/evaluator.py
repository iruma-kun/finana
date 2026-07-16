from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve, confusion_matrix
import numpy as np
import pandas as pd
from typing import Dict, Any, List

class VolatilityEvaluator:
    """
    Evaluates classification performance (precision, ROC-AUC) and simulates
    financial risk management backtests based on volatility predictions.
    """
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """
        Calculates key classification metrics including precision and ROC-AUC.
        """
        return {
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'roc_auc': float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
        }

    @staticmethod
    def print_ascii_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, width: int = 40, height: int = 15):
        """
        Renders a stunning ASCII ROC Curve in the terminal to visualize ROC-AUC.
        """
        if len(np.unique(y_true)) < 2:
            print("Cannot plot ROC curve: only one class present in target.")
            return

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        
        # Grid initialization (spaces)
        grid = [[" " for _ in range(width)] for _ in range(height)]
        
        # Plot diagonal random line (y = x)
        for col in range(width):
            row = int((1 - (col / (width - 1))) * (height - 1))
            grid[row][col] = "."
            
        # Plot ROC curve points
        # For each column in the grid, map it to an FPR value, interpolate TPR, and plot
        for col in range(width):
            curr_fpr = col / (width - 1)
            # Interpolate TPR for this FPR
            curr_tpr = np.interp(curr_fpr, fpr, tpr)
            row = int((1 - curr_tpr) * (height - 1))
            # Ensure row bounds
            row = max(0, min(height - 1, row))
            grid[row][col] = "█"

        # Print the grid
        print("\n" + " " * 8 + "┌" + "─" * width + "┐")
        for i, row in enumerate(grid):
            # Y-axis labels (TPR)
            tpr_val = 1.0 - (i / (height - 1))
            label = f"{tpr_val:4.1f} ───┤" if i % 3 == 0 else "        │"
            print(label + "".join(row) + "│")
            
        print(" " * 8 + "└" + "─" * width + "┘")
        
        # X-axis labels (FPR)
        xaxis_ticks = " " * 9
        for col in range(0, width, width // 4):
            val = col / (width - 1)
            xaxis_ticks += f"{val:4.1f}" + " " * (width // 4 - 4)
        print(xaxis_ticks)
        print(" " * 18 + "False Positive Rate (FPR)")
        print(" " * 22 + "(Diagonal '.' = Random; '█' = Model ROC)")

    @staticmethod
    def simulate_risk_management(y_true: np.ndarray, y_pred: np.ndarray, realized_vol: np.ndarray, 
                                  initial_capital: float = 1000000.0) -> Dict[str, Any]:
        """
        Simulates a Volatility-Targeted Risk Management strategy.
        When High Volatility is predicted for tomorrow, we reduce portfolio exposure to 20%.
        When Normal Volatility is predicted, we maintain 100% exposure.
        
        This demonstrates the concrete business value of forecasting financial risk.
        """
        n_days = len(y_pred)
        capital_static = [initial_capital]
        capital_managed = [initial_capital]
        
        # Simulate returns using realized volatility with random direction to represent daily stock returns
        # For evaluation, we assume a benchmark return series. Let's simulate a standard distribution of returns
        # but scaling volatility.
        np.random.seed(42)
        daily_bench_returns = np.random.normal(0.0002, 0.01, n_days)
        
        # Adjust returns for actual volatility to keep it tied to the actual environment
        # Realized vol is daily (high - low) / close, so returns standard deviation is proportional to realized vol
        daily_returns = daily_bench_returns * (realized_vol / np.median(realized_vol))
        
        for i in range(n_days):
            ret = daily_returns[i]
            
            # Static strategy (Always 100% exposed)
            capital_static.append(capital_static[-1] * (1 + ret))
            
            # Managed strategy (Volatility target: if predicted high volatility tomorrow, reduce risk today)
            # The model predicts volatility for tomorrow based on today's features.
            # So if y_pred[i] (which is prediction for day i+1) is 1, we reduce exposure for day i+1
            exposure = 0.2 if y_pred[i] == 1 else 1.0
            managed_ret = ret * exposure
            capital_managed.append(capital_managed[-1] * (1 + managed_ret))
            
        capital_static = np.array(capital_static)
        capital_managed = np.array(capital_managed)
        
        # Calculate Drawdowns
        def calculate_max_drawdown(nav_series):
            peaks = np.maximum.accumulate(nav_series)
            drawdowns = (nav_series - peaks) / peaks
            return float(np.min(drawdowns))
            
        static_mdd = calculate_max_drawdown(capital_static)
        managed_mdd = calculate_max_drawdown(capital_managed)
        
        # Total Returns
        static_total_return = float((capital_static[-1] - initial_capital) / initial_capital)
        managed_total_return = float((capital_managed[-1] - initial_capital) / initial_capital)
        
        # Volatility of portfolio
        static_vol = float(np.std(np.diff(capital_static)/capital_static[:-1]) * np.sqrt(252))
        managed_vol = float(np.std(np.diff(capital_managed)/capital_managed[:-1]) * np.sqrt(252))
        
        # Sharpe Ratio (assuming risk free rate = 0 for simplicity)
        static_sharpe = (static_total_return / static_vol) if static_vol > 0 else 0
        managed_sharpe = (managed_total_return / managed_vol) if managed_vol > 0 else 0
        
        return {
            'static_final_value': float(capital_static[-1]),
            'managed_final_value': float(capital_managed[-1]),
            'static_total_return': static_total_return,
            'managed_total_return': managed_total_return,
            'static_max_drawdown': static_mdd,
            'managed_max_drawdown': managed_mdd,
            'static_volatility': static_vol,
            'managed_volatility': managed_vol,
            'static_sharpe': static_sharpe,
            'managed_sharpe': managed_sharpe
        }

    @classmethod
    def compare_models(cls, results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Organizes performance comparisons into a structured table.
        """
        df = pd.DataFrame(results).T
        # Round columns for clean display
        return df.round(4)
