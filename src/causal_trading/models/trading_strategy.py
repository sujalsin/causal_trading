import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from .causal_model import CausalModel
import warnings

class CausalTradingStrategy:
    """Trading strategy based on causal inference."""
    
    def __init__(
        self,
        symbols: List[str],
        lookback_period: int = 252,
        position_size: float = 0.1,
        stop_loss: float = 0.02,
        take_profit: float = 0.05,
        confidence_threshold: float = 0.95,
        random_state: int = 42
    ):
        """Initialize the trading strategy.
        
        Args:
            symbols: List of stock symbols
            lookback_period: Number of days to look back for analysis
            position_size: Maximum position size as fraction of portfolio
            stop_loss: Stop loss threshold
            take_profit: Take profit threshold
            confidence_threshold: Confidence level for signal generation
            random_state: Random seed for reproducibility
        """
        self.symbols = symbols
        self.lookback_period = lookback_period
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.confidence_threshold = confidence_threshold
        self.causal_model = CausalModel(random_state=random_state)
        self.positions = {symbol: 0 for symbol in symbols}  # Initialize positions
        
    def calculate_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators as features.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with additional technical features
        """
        df = data.copy()
        
        # Price-based features
        df['Returns'] = df.groupby('Symbol')['Close'].pct_change()
        df['Log_Returns'] = np.log1p(df['Returns'])
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'MA_{window}'] = df.groupby('Symbol')['Close'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'MA_{window}_Slope'] = df.groupby('Symbol')[f'MA_{window}'].transform(
                lambda x: (x - x.shift(1)) / x.shift(1)
            )
        
        # Volatility
        df['Volatility'] = df.groupby('Symbol')['Returns'].transform(
            lambda x: x.rolling(window=20, min_periods=1).std()
        )
        
        # Volume features
        df['Volume_MA_5'] = df.groupby('Symbol')['Volume'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        df['Volume_MA_20'] = df.groupby('Symbol')['Volume'].transform(
            lambda x: x.rolling(window=20, min_periods=1).mean()
        )
        df['Relative_Volume'] = df['Volume'] / df['Volume_MA_20']
        
        # RSI
        def calculate_rsi(prices, periods=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        df['RSI'] = df.groupby('Symbol')['Close'].transform(
            lambda x: calculate_rsi(x)
        )
        
        return df.fillna(0)
        
    def identify_trading_signals(
        self,
        data: pd.DataFrame,
        features: List[str]
    ) -> Dict[str, float]:
        """Generate trading signals using causal inference.
        
        Args:
            data: DataFrame with price and feature data
            features: List of feature names to use
            
        Returns:
            Dictionary of symbols and their signal strengths
        """
        # Calculate technical features first
        df = self.calculate_technical_features(data)
        signals = {}
        
        for symbol in self.symbols:
            symbol_data = df[df['Symbol'] == symbol].copy()
            
            if len(symbol_data) < 20:  # Minimum required for statistical validity
                signals[symbol] = 0
                continue
                
            # Define treatment and outcome
            symbol_data['Treatment'] = (
                symbol_data['MA_5'] > symbol_data['MA_20']
            ).astype(int)
            symbol_data['Outcome'] = symbol_data['Returns'].shift(-1)
            
            # Estimate causal effect
            effect, lower, upper = self.causal_model.estimate_causal_effect(
                data=symbol_data,
                treatment='Treatment',
                outcome='Outcome',
                features=features
            )
            
            # Generate signal based on effect size and confidence
            signal_strength = effect
            if lower > 0:  # Strong positive signal
                signal_strength *= 1.5
            elif upper < 0:  # Strong negative signal
                signal_strength *= 1.5
            
            signals[symbol] = signal_strength
            
        return signals
        
    def calculate_position_sizes(
        self,
        signals: Dict[str, float],
        portfolio_value: float,
        max_position_size: Optional[float] = None
    ) -> Dict[str, float]:
        """Calculate position sizes based on signals.
        
        Args:
            signals: Dictionary of symbols and their signal strengths
            portfolio_value: Current portfolio value
            max_position_size: Maximum position size as fraction of portfolio
            
        Returns:
            Dictionary of symbols and their position sizes
        """
        # Use provided max_position_size or default to self.position_size
        position_size_limit = max_position_size if max_position_size is not None else self.position_size
        
        # Normalize signals to sum to 1
        total_signal = sum(abs(s) for s in signals.values())
        if total_signal == 0:
            return {symbol: 0 for symbol in signals}
            
        normalized_signals = {
            symbol: signal / total_signal
            for symbol, signal in signals.items()
        }
        
        # Calculate position sizes with risk management
        positions = {}
        for symbol, signal in normalized_signals.items():
            position = signal * position_size_limit * portfolio_value
            
            # Apply stop loss and take profit
            if abs(signal) < self.stop_loss:
                position = 0
            elif abs(signal) > self.take_profit:
                position = np.sign(signal) * self.take_profit * portfolio_value
                
            positions[symbol] = position
            
        return positions
        
    def backtest(
        self,
        data: pd.DataFrame,
        features: List[str],
        initial_capital: float = 1000000
    ) -> pd.DataFrame:
        """Run backtest of the trading strategy.
        
        Args:
            data: DataFrame with price data
            features: List of features to use for signal generation
            initial_capital: Initial capital for backtesting
            
        Returns:
            DataFrame with backtest results
        """
        # Prepare data
        df = self.calculate_technical_features(data)
        
        portfolio_value = initial_capital
        results = []
        
        # Convert data to datetime index and sort
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Initialize lookback buffer
        lookback_buffer = []
        
        # Group data by date
        for date, group in df.groupby('Date'):
            # Update lookback buffer
            lookback_buffer.extend(group.to_dict('records'))
            if len(lookback_buffer) > self.lookback_period:
                lookback_buffer = lookback_buffer[-self.lookback_period:]
                
            # Convert buffer to DataFrame
            historical_data = pd.DataFrame(lookback_buffer)
            
            if len(historical_data) >= 20:  # Minimum required for statistical validity
                # Generate trading signals
                signals = self.identify_trading_signals(
                    data=historical_data,
                    features=features
                )
                
                # Calculate position sizes
                positions = self.calculate_position_sizes(
                    signals=signals,
                    portfolio_value=portfolio_value
                )
            else:
                positions = {symbol: 0 for symbol in self.symbols}
            
            # Calculate daily returns
            daily_returns = {}
            for symbol in self.symbols:
                symbol_data = group[group['Symbol'] == symbol]
                if not symbol_data.empty:
                    daily_returns[symbol] = symbol_data['Returns'].iloc[0]
                else:
                    daily_returns[symbol] = 0.0
            
            # Update portfolio value
            portfolio_return = sum(
                positions[symbol] * daily_returns[symbol]
                for symbol in self.symbols
            )
            portfolio_value *= (1 + portfolio_return)
            
            # Store results
            results.append({
                'Date': date,
                'Portfolio_Value': portfolio_value,
                'Daily_Return': portfolio_return,
                'Positions': positions.copy()
            })
            
        return pd.DataFrame(results)
        
    def generate_trading_report(self, backtest_results: pd.DataFrame) -> Dict[str, float]:
        """Generate performance metrics from backtest results.
        
        Args:
            backtest_results: DataFrame with backtest results
            
        Returns:
            Dictionary of performance metrics
        """
        returns = backtest_results['Daily_Return'].values
        portfolio_values = backtest_results['Portfolio_Value'].values
        
        # Basic return metrics
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # Risk metrics
        daily_vol = np.std(returns)
        annual_vol = daily_vol * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol != 0 else 0
        
        # Drawdown analysis
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = np.min(drawdowns)
        
        # Win rate analysis
        winning_days = np.sum(returns > 0)
        total_days = len(returns)
        win_rate = winning_days / total_days if total_days > 0 else 0
        
        return {
            'Total_Return': total_return,
            'Annual_Return': annual_return,
            'Annual_Volatility': annual_vol,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown,
            'Win_Rate': win_rate
        }
