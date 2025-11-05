"""
Statistical Significance Testing

This module contains functions for statistical significance analysis:
- Calculate t-tests, p-values, and confidence intervals
- Create confidence interval summary tables
"""

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from typing import Dict, List


def calculate_statistical_significance(trials: List[Dict]) -> Dict[str, Dict]:
    """
    Calculate statistical significance and confidence intervals for key metrics

    Returns:
        Dict with metrics and their statistical properties:
        - mean, std, confidence_interval, t_statistic, p_value, is_significant
    """
    results = {}

    # Key metrics to analyze
    metrics = {
        'total_return_pct': 'Return %',
        'sharpe_ratio': 'Sharpe Ratio',
        'sortino_ratio': 'Sortino Ratio',
        'win_rate': 'Win Rate %'
    }

    for metric_key, metric_name in metrics.items():
        values = [t[metric_key] for t in trials]
        n = len(values)

        if n < 2:
            continue

        # Calculate statistics
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)  # Sample standard deviation
        se = std_val / np.sqrt(n)  # Standard error

        # 95% confidence interval using t-distribution
        confidence_level = 0.95
        alpha = 1 - confidence_level
        t_critical = scipy_stats.t.ppf(1 - alpha/2, df=n-1)
        margin_of_error = t_critical * se
        ci_lower = mean_val - margin_of_error
        ci_upper = mean_val + margin_of_error

        # One-sample t-test against zero (null hypothesis: mean = 0)
        t_statistic, p_value = scipy_stats.ttest_1samp(values, 0)

        # Determine significance (p < 0.05)
        is_significant = p_value < 0.05

        # Probability of positive outcome (for return-like metrics)
        prob_positive = sum(1 for v in values if v > 0) / n * 100

        results[metric_name] = {
            'mean': mean_val,
            'std': std_val,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            't_statistic': t_statistic,
            'p_value': p_value,
            'is_significant': is_significant,
            'prob_positive': prob_positive,
            'n_trials': n
        }

    return results


def create_confidence_interval_table(trials: List[Dict]) -> pd.DataFrame:
    """
    Create table showing confidence intervals and statistical significance.
    Returns numeric values for proper sorting - formatting should be done in display.
    """
    stats = calculate_statistical_significance(trials)

    table_data = []

    for metric_name, data in stats.items():
        # Store numeric values, CI as separate columns for sorting
        table_data.append({
            'Metric': metric_name,
            'Mean': data['mean'],
            'CI Lower': data['ci_lower'],
            'CI Upper': data['ci_upper'],
            'Std Dev': data['std'],
            'P-Value': data['p_value'],
            'Significant?': "✓ Yes" if data['is_significant'] else "✗ No",
            'Prob > 0': data['prob_positive']
        })

    return pd.DataFrame(table_data)
