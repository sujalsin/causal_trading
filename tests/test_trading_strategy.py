import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from causal_trading.models.trading_strategy import CausalTradingStrategy

def create_test_data():
    """Create synthetic data for testing."""
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    symbols = ['AAPL', 'GOOGL']
    data = []
    
    # Generate more realistic test data
    for symbol in symbols:
        price = 100.0
        volume = 1000000
        for date in dates:
            # Generate price movement
            returns = np.random.normal(0.0001, 0.02)  # Mean slightly positive with realistic volatility
            price *= (1 + returns)
            
            # Generate volume with autocorrelation
            volume = 0.7 * volume + 0.3 * (1000000 + np.random.normal(0, 100000))
            
            # Calculate technical indicators
            ma50 = price * (1 + np.random.normal(0, 0.01))
            ma200 = price * (1 + np.random.normal(0, 0.01))
            
            data.append({
                'Date': date,
                'Symbol': symbol,
                'Close': price,
                'Volume': volume,
                'MA50': ma50,
                'MA200': ma200,
                'Daily_Return': returns,
                'Volume_Ratio': volume / 1000000,
                'Rolling_Volatility': abs(returns) * np.sqrt(252),
                'RSI': 50 + np.random.normal(0, 10),
                'Treatment': int(ma50 > ma200)
            })
    
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def test_trading_strategy_initialization():
    """Test trading strategy initialization."""
    symbols = ['AAPL', 'GOOGL']
    strategy = CausalTradingStrategy(symbols=symbols)
    
    assert strategy.symbols == symbols
    assert strategy.lookback_period == 252
    assert strategy.confidence_threshold == 0.95
    assert all(pos == 0 for pos in strategy.positions.values())

def test_signal_generation():
    """Test trading signal generation."""
    data = create_test_data()
    symbols = ['AAPL', 'GOOGL']
    strategy = CausalTradingStrategy(symbols=symbols)
    
    features = ['Volume_Ratio', 'Rolling_Volatility', 'RSI']
    signals = strategy.identify_trading_signals(data, features)
    
    assert isinstance(signals, dict)
    assert all(symbol in signals for symbol in symbols)
    assert all(isinstance(signal, (int, float)) for signal in signals.values())
    assert all(-1 <= signal <= 1 for signal in signals.values())

def test_position_sizing():
    """Test position size calculation."""
    symbols = ['AAPL', 'GOOGL']
    strategy = CausalTradingStrategy(symbols=symbols)
    
    # Test with zero signals
    signals = {symbol: 0 for symbol in symbols}
    positions = strategy.calculate_position_sizes(signals, portfolio_value=1000000)
    assert all(pos == 0 for pos in positions.values())
    
    # Test with non-zero signals
    signals = {'AAPL': 1, 'GOOGL': -0.5}
    positions = strategy.calculate_position_sizes(
        signals,
        portfolio_value=1000000,
        max_position_size=0.2
    )
    assert all(abs(pos) <= 200000 for pos in positions.values())  # 20% of 1M

def test_backtest():
    """Test backtesting functionality."""
    data = create_test_data()
    symbols = ['AAPL', 'GOOGL']
    strategy = CausalTradingStrategy(symbols=symbols)
    
    features = ['Volume_Ratio', 'Rolling_Volatility', 'RSI']
    results = strategy.backtest(data, features, initial_capital=1000000)
    
    assert isinstance(results, pd.DataFrame)
    assert 'Portfolio_Value' in results.columns
    assert 'Daily_Return' in results.columns
    assert len(results) > 0
    assert all(results['Portfolio_Value'] > 0)  # No bankruptcy

def test_performance_report():
    """Test performance reporting."""
    data = create_test_data()
    symbols = ['AAPL', 'GOOGL']
    strategy = CausalTradingStrategy(symbols=symbols)
    
    features = ['Volume_Ratio', 'Rolling_Volatility', 'RSI']
    backtest_results = strategy.backtest(data, features, initial_capital=1000000)
    performance = strategy.generate_trading_report(backtest_results)
    
    assert isinstance(performance, dict)
    assert 'Total_Return' in performance
    assert 'Annual_Return' in performance
    assert 'Sharpe_Ratio' in performance
    assert 'Max_Drawdown' in performance
    assert performance['Max_Drawdown'] <= 0  # Drawdown should be negative or zero

if __name__ == '__main__':
    pytest.main([__file__])
