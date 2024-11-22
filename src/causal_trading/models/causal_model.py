import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import networkx as nx
from dowhy import CausalModel
from sklearn.preprocessing import StandardScaler
from econml.dml import CausalForestDML, DML
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
import warnings
from econml.sklearn_extensions.linear_model import WeightedLasso

class CausalInferenceModel:
    """Implements causal inference methods for financial data analysis."""
    
    def __init__(self):
        """Initialize the causal inference model."""
        self.scaler = StandardScaler()
        self.causal_model = None
        self.treatment_effects = {}
        
    def create_causal_graph(self, variables: List[str], edges: List[tuple]) -> nx.DiGraph:
        """Create a Directed Acyclic Graph (DAG) representing causal relationships.
        
        Args:
            variables: List of variable names
            edges: List of tuples representing directed edges (cause, effect)
            
        Returns:
            NetworkX DiGraph object
        """
        G = nx.DiGraph()
        G.add_nodes_from(variables)
        G.add_edges_from(edges)
        
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("Graph contains cycles and is not a DAG")
            
        return G
    
    def estimate_causal_effect(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        confounders: List[str],
        graph: Optional[nx.DiGraph] = None
    ) -> Dict:
        """Estimate causal effect using DoWhy framework.
        
        Args:
            data: DataFrame containing variables
            treatment: Treatment variable
            outcome: Outcome variable
            confounders: List of confounding variables
            graph: Optional causal graph
            
        Returns:
            Dictionary with causal effect estimates
        """
        # Ensure we have enough data points
        if len(data) < 20:  # Minimum required for reliable statistical tests
            return {
                'effect_estimate': 0.0,
                'p_value': 1.0
            }
            
        try:
            # Scale the features
            X = self.scaler.fit_transform(data[confounders])
            T = data[treatment].values
            Y = data[outcome].values
            
            # Use econml's Double Machine Learning
            est = DML(
                model_y=RandomForestRegressor(n_estimators=100, random_state=42),
                model_t=LassoCV(random_state=42),
                random_state=42
            )
            
            est.fit(Y, T, X=X)
            effect_value = float(est.effect(X).mean())
            
            # Calculate p-value using bootstrap
            n_bootstrap = 100
            bootstrap_effects = []
            for _ in range(n_bootstrap):
                idx = np.random.randint(0, len(data), len(data))
                X_boot = X[idx]
                T_boot = T[idx]
                Y_boot = Y[idx]
                
                est_boot = DML(
                    model_y=RandomForestRegressor(n_estimators=100, random_state=42),
                    model_t=LassoCV(random_state=42),
                    random_state=42
                )
                est_boot.fit(Y_boot, T_boot, X=X_boot)
                bootstrap_effects.append(float(est_boot.effect(X_boot).mean()))
            
            # Calculate p-value as proportion of bootstrap samples where effect has opposite sign
            p_value = np.mean(np.sign(bootstrap_effects) != np.sign(effect_value))
            
        except Exception as e:
            print(f"Warning: DML estimation failed with error {str(e)}. Falling back to simple estimation.")
            effect_value = float(data[data[treatment] == 1][outcome].mean() - 
                              data[data[treatment] == 0][outcome].mean())
            p_value = 0.5  # Conservative p-value for fallback method
        
        return {
            'effect_estimate': effect_value,
            'p_value': p_value
        }
    
    def estimate_heterogeneous_effects(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        features: List[str],
        n_estimators: int = 100
    ) -> np.ndarray:
        """Estimate heterogeneous treatment effects using Causal Forest.
        
        Args:
            data: DataFrame containing variables
            treatment: Treatment variable
            outcome: Outcome variable
            features: List of feature variables
            n_estimators: Number of trees in the forest
            
        Returns:
            Array of treatment effect estimates
        """
        # Prepare data
        X = data[features].values
        T = data[treatment].values
        Y = data[outcome].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit causal forest
        cf = CausalForestDML(
            n_estimators=n_estimators,
            random_state=42
        )
        cf.fit(X_scaled, T, Y)
        
        # Estimate treatment effects
        treatment_effects = cf.effect(X_scaled)
        
        return treatment_effects
    
    def get_feature_importance(self, model) -> pd.Series:
        """Get feature importance scores from the causal forest.
        
        Args:
            model: Fitted causal forest model
            
        Returns:
            Series with feature importance scores
        """
        return pd.Series(
            model.feature_importances_,
            index=self.features,
            name='importance'
        ).sort_values(ascending=False)
    
    def validate_assumptions(self, data: pd.DataFrame, treatment: str, confounders: List[str]) -> Dict:
        """Validate causal inference assumptions.
        
        Args:
            data: DataFrame containing variables
            treatment: Treatment variable
            confounders: List of confounding variables
            
        Returns:
            Dictionary with validation results
        """
        results = {}
        
        # Check balance of confounders
        for confounder in confounders:
            treated = data[data[treatment] == 1][confounder]
            control = data[data[treatment] == 0][confounder]
            
            # Standardized mean difference
            smd = (treated.mean() - control.mean()) / \
                  np.sqrt((treated.var() + control.var()) / 2)
            
            results[f'{confounder}_smd'] = smd
            
        return results

class CausalModel:
    def __init__(self, random_state: int = 42):
        """Initialize the causal model with advanced ML estimators.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        # Initialize more sophisticated ML models
        self.model = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=random_state),
            model_t=WeightedLasso(alpha=0.01, random_state=random_state),
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=10,
            random_state=random_state
        )
        
    def estimate_causal_effect(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        features: List[str],
        min_samples: int = 100
    ) -> Tuple[float, float, float]:
        """Estimate the causal effect using Double Machine Learning with Causal Forests.
        
        Args:
            data: DataFrame containing the data
            treatment: Name of treatment variable
            outcome: Name of outcome variable
            features: List of feature names
            min_samples: Minimum number of samples required
            
        Returns:
            Tuple of (effect estimate, lower bound, upper bound)
        """
        if len(data) < min_samples:
            warnings.warn(f"Sample size ({len(data)}) is less than minimum required ({min_samples})")
            return 0.0, 0.0, 0.0
            
        try:
            # Scale features
            X = self.scaler.fit_transform(data[features])
            T = data[treatment].values
            Y = data[outcome].values
            
            # Fit the causal model
            self.model.fit(X, T, Y)
            
            # Get treatment effect estimate and confidence intervals
            effects = self.model.effect(X)
            mean_effect = np.mean(effects)
            
            # Calculate confidence intervals
            lower, upper = self.model.effect_interval(X)
            lower_bound = np.mean(lower)
            upper_bound = np.mean(upper)
            
            return mean_effect, lower_bound, upper_bound
            
        except Exception as e:
            warnings.warn(f"Error in causal effect estimation: {str(e)}")
            return 0.0, 0.0, 0.0
            
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores from the causal forest.
        
        Returns:
            Dictionary of feature names and their importance scores
        """
        try:
            importance = self.model.feature_importances_
            return {f"Feature_{i}": imp for i, imp in enumerate(importance)}
        except:
            return {}
