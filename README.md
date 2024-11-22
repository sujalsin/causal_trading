# Causal Trading Strategy

This project implements advanced causal inference methods for algorithmic trading, aiming to distinguish true causal relationships from mere correlations in financial data.

## Project Structure

```
causal_trading/
├── data/              # Financial data storage
├── notebooks/         # Jupyter notebooks for analysis
├── src/              # Source code
│   ├── data/         # Data processing modules
│   ├── models/       # Causal inference models
│   └── utils/        # Utility functions
└── tests/            # Unit and integration tests
```

## Features

- Causal inference techniques:
  - Directed Acyclic Graphs (DAGs)
  - Propensity Score Matching
  - Instrumental Variables
  - Regression Discontinuity Design
- Machine learning integration
- Real-time data processing
- Interactive visualizations

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Data Collection:
   - Use `src/data/collector.py` to fetch financial data
   - Store data in the `data/` directory

2. Causal Analysis:
   - Explore notebooks in `notebooks/` for examples
   - Use modules in `src/models/` for causal inference

3. Running Tests:
   ```bash
   pytest tests/
   ```

## Implementation Details

Our implementation leverages advanced causal inference techniques:

- **Double Machine Learning**: Uses CausalForestDML from econml for robust estimation
- **Technical Features**: 
  - Moving averages (5, 10, 20, 50 day windows)
  - Volatility tracking
  - Volume analysis
  - RSI (Relative Strength Index)
  - Log returns
- **Risk Management**:
  - Dynamic position sizing based on signal strength
  - Stop loss and take profit mechanisms
  - Confidence threshold-based filtering

## Performance Metrics

The strategy evaluates performance using:
- Total and annual returns
- Sharpe ratio
- Maximum drawdown
- Win rate
- Risk-adjusted metrics

## License

MIT License
