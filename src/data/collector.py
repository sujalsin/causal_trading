import yfinance as yf
import pandas as pd
from typing import List, Optional
from datetime import datetime, timedelta

class FinancialDataCollector:
    """Collects and preprocesses financial data from various sources."""
    
    def __init__(self, cache_dir: str = "../../data"):
        """Initialize the data collector.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir
        
    def fetch_stock_data(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch historical stock data for given symbols.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for data collection
            end_date: End date for data collection
            interval: Data frequency ('1d', '1h', etc.)
            
        Returns:
            DataFrame with historical stock data
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()
            
        dfs = []
        for symbol in symbols:
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date, interval=interval)
            df['Symbol'] = symbol
            dfs.append(df)
            
        return pd.concat(dfs)
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various return metrics.
        
        Args:
            df: DataFrame with stock price data
            
        Returns:
            DataFrame with additional return columns
        """
        df['Daily_Return'] = df.groupby('Symbol')['Close'].pct_change()
        df['Log_Return'] = df.groupby('Symbol')['Close'].apply(lambda x: np.log(x/x.shift(1)))
        
        # Calculate rolling metrics
        df['Rolling_Volatility'] = df.groupby('Symbol')['Daily_Return'].rolling(window=20).std()
        df['Rolling_Mean'] = df.groupby('Symbol')['Daily_Return'].rolling(window=20).mean()
        
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators as potential causal factors.
        
        Args:
            df: DataFrame with stock price data
            
        Returns:
            DataFrame with additional technical indicators
        """
        # Moving averages
        df['MA50'] = df.groupby('Symbol')['Close'].rolling(window=50).mean()
        df['MA200'] = df.groupby('Symbol')['Close'].rolling(window=200).mean()
        
        # Volume indicators
        df['Volume_MA20'] = df.groupby('Symbol')['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']
        
        # Momentum indicators
        df['RSI'] = df.groupby('Symbol')['Daily_Return'].rolling(window=14).apply(
            lambda x: 100 - (100 / (1 + (x[x > 0].mean() / -x[x < 0].mean())))
        )
        
        return df
    
    def save_data(self, df: pd.DataFrame, filename: str):
        """Save data to cache directory.
        
        Args:
            df: DataFrame to save
            filename: Name of the file
        """
        df.to_parquet(f"{self.cache_dir}/{filename}.parquet")
        
    def load_data(self, filename: str) -> pd.DataFrame:
        """Load data from cache directory.
        
        Args:
            filename: Name of the file
            
        Returns:
            Loaded DataFrame
        """
        return pd.read_parquet(f"{self.cache_dir}/{filename}.parquet")
