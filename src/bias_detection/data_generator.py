import numpy as np
import pandas as pd
import os

def generate_behavioral_data(num_samples=1000, random_state=42):
    np.random.seed(random_state)
    
    # Target biases:
    # 0: Rational (No Bias)
    # 1: Loss Aversion
    # 2: FOMO / Herd Mentality
    # 3: Overconfidence
    # 4: Recency Bias
    
    # We will generate base features and then apply class-specific patterns
    data = []
    
    for i in range(num_samples):
        # Determine label
        label = np.random.choice([0, 1, 2, 3, 4], p=[0.25, 0.20, 0.20, 0.18, 0.17])
        
        # Base features
        trade_frequency = np.random.randint(1, 50) # trades per month
        avg_holding_period = np.random.uniform(1, 365) # days
        win_loss_ratio = np.random.uniform(0.3, 1.5)
        # Average days holding a losing position / average days holding a winning position
        loss_win_hold_ratio = np.random.uniform(0.8, 1.5)
        # Portfolio concentration (HHI index: 0 to 1, higher means less diversified)
        portfolio_concentration = np.random.uniform(0.1, 0.8)
        # Correlation of trades with 5-day market trend (0 to 1, higher means following herd)
        trend_following_index = np.random.uniform(0.1, 0.9)
        # Percentage of trades with stop loss set
        stop_loss_pct = np.random.uniform(0.1, 1.0)
        # Average position size as % of portfolio
        avg_position_size = np.random.uniform(0.01, 0.25)
        # Churn rate (portfolio turnover per month)
        portfolio_churn = np.random.uniform(0.05, 0.8)
        # Performance metrics (annualized return, max drawdown)
        annual_return = np.random.uniform(-0.15, 0.25)
        max_drawdown = np.random.uniform(0.05, 0.40)
        
        # Apply bias-specific signatures
        if label == 0: # Rational
            trade_frequency = np.random.randint(1, 10)
            avg_holding_period = np.random.uniform(60, 365)
            win_loss_ratio = np.random.uniform(0.8, 1.6)
            loss_win_hold_ratio = np.random.uniform(0.6, 0.95) # sells losers fast, lets winners run
            portfolio_concentration = np.random.uniform(0.1, 0.35) # diversified
            trend_following_index = np.random.uniform(0.2, 0.5)
            stop_loss_pct = np.random.uniform(0.7, 1.0)
            avg_position_size = np.random.uniform(0.02, 0.08)
            portfolio_churn = np.random.uniform(0.05, 0.20)
            annual_return = np.random.uniform(0.08, 0.22)
            max_drawdown = np.random.uniform(0.08, 0.18)
            
        elif label == 1: # Loss Aversion
            # Holds losers much longer than winners
            loss_win_hold_ratio = np.random.uniform(1.8, 4.0)
            win_loss_ratio = np.random.uniform(0.3, 0.7) # low win/loss ratio because of holding losers
            avg_holding_period = np.random.uniform(90, 300)
            stop_loss_pct = np.random.uniform(0.05, 0.4) # rarely sets stop losses
            annual_return = np.random.uniform(-0.10, 0.05)
            max_drawdown = np.random.uniform(0.20, 0.45)
            
        elif label == 2: # FOMO / Herd Mentality
            # Highly correlated with trends, higher concentration
            trend_following_index = np.random.uniform(0.75, 0.99)
            portfolio_concentration = np.random.uniform(0.4, 0.8)
            trade_frequency = np.random.randint(15, 40)
            portfolio_churn = np.random.uniform(0.4, 0.8)
            stop_loss_pct = np.random.uniform(0.2, 0.6)
            annual_return = np.random.uniform(-0.25, 0.12)
            max_drawdown = np.random.uniform(0.25, 0.50)
            
        elif label == 3: # Overconfidence
            # Extremely high frequency, large positions, low diversification, low stop losses
            trade_frequency = np.random.randint(35, 120)
            portfolio_concentration = np.random.uniform(0.6, 0.95)
            avg_position_size = np.random.uniform(0.18, 0.45)
            stop_loss_pct = np.random.uniform(0.0, 0.3)
            avg_holding_period = np.random.uniform(0.5, 15) # very short holding
            annual_return = np.random.uniform(-0.40, 0.30) # high variance
            max_drawdown = np.random.uniform(0.30, 0.65)
            
        elif label == 4: # Recency Bias
            # Churns portfolio rapidly based on recent news or market moves
            portfolio_churn = np.random.uniform(0.65, 0.98)
            trade_frequency = np.random.randint(20, 60)
            avg_holding_period = np.random.uniform(5, 30)
            win_loss_ratio = np.random.uniform(0.4, 0.9)
            annual_return = np.random.uniform(-0.18, 0.08)
            max_drawdown = np.random.uniform(0.20, 0.40)
            
        # Add random noise to make it realistic
        noise = np.random.normal(0, 0.02, 11)
        
        sample = {
            'trade_frequency': max(1, int(trade_frequency)),
            'avg_holding_period_days': max(0.5, avg_holding_period + noise[0]*30),
            'win_loss_ratio': max(0.1, win_loss_ratio + noise[1]),
            'loss_win_hold_ratio': max(0.1, loss_win_hold_ratio + noise[2]),
            'portfolio_concentration_hhi': clip(portfolio_concentration + noise[3], 0.01, 1.0),
            'trend_following_index': clip(trend_following_index + noise[4], 0.0, 1.0),
            'stop_loss_pct': clip(stop_loss_pct + noise[5], 0.0, 1.0),
            'avg_position_size': clip(avg_position_size + noise[6], 0.005, 1.0),
            'portfolio_churn_rate': clip(portfolio_churn + noise[7], 0.01, 1.0),
            'annual_return': annual_return + noise[8]*0.1,
            'max_drawdown': clip(max_drawdown + noise[9]*0.1, 0.01, 0.95),
            'label': label
        }
        data.append(sample)
        
    df = pd.DataFrame(data)
    return df

def clip(val, min_val, max_val):
    return min(max(val, min_val), max_val)

if __name__ == '__main__':
    print("Generating behavioral finance datasets...")
    df = generate_behavioral_data(num_samples=2500)
    
    # Ensure data directory exists
    os.makedirs('C:/Users/Sahil Sharma/Projects/InvestIQ/data', exist_ok=True)
    
    df.to_csv('C:/Users/Sahil Sharma/Projects/InvestIQ/data/behavioral_profiles.csv', index=False)
    print(f"Generated {len(df)} samples saved to 'data/behavioral_profiles.csv'.")
    print("\nClass distribution:")
    print(df['label'].value_counts().rename({
        0: 'Rational (0)',
        1: 'Loss Aversion (1)',
        2: 'FOMO/Herd (2)',
        3: 'Overconfidence (3)',
        4: 'Recency Bias (4)'
    }))
