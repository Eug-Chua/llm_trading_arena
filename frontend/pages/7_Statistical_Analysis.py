"""
Statistical Analysis Page - Intra-Model Variance

Shows non-determinism effects within the same model configuration:
- Overlay equity curves from multiple trials (same model+temp)
- Visualize distribution of outcomes from LLM variance
- Compare low-temp (0.1) vs high-temp (0.7) consistency
- Statistical metrics for each configuration independently
"""

import streamlit as st
from pathlib import Path
import sys
from typing import Dict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import logger
from src.utils.logger import setup_logger
logger = setup_logger(__name__)

# Import utilities from refactored modules
from frontend.utils.checkpoint_utils import load_checkpoint
from frontend.utils.statistical_metrics import extract_metrics, group_by_model_config
from frontend.utils.capture_ratios import calculate_coin_capture_ratios
# Import from refactored chart modules
from frontend.utils.charts.performance_charts import (
    create_trial_performance_chart,
    create_trade_pnl_kde_overlay,
    create_return_distribution_chart,
    create_risk_return_scatter,
)
from frontend.utils.charts.risk_charts import (
    create_upside_downside_comparison,
    create_volatility_scatter,
    create_distribution_shape_chart,
    create_risk_metrics_summary,
)
from frontend.utils.charts.statistical_analysis import (
    create_confidence_interval_table,
    calculate_statistical_significance,
)
from frontend.utils.charts.capture_ratio_charts import create_capture_ratios_chart
from frontend.utils.charts.summary_tables import (
    create_overall_stats_table,
    create_advanced_stats_table,
)
from frontend.utils.charts.equity_charts import create_equity_curve_overlay

# Page config
st.set_page_config(
    page_title="Statistical Analysis - LLM Trading Arena",
    page_icon="📊",
    layout="wide"
)


# Wrap checkpoint loading with Streamlit cache
@st.cache_data
def load_checkpoint_cached(checkpoint_path: str):
    """Load checkpoint data with caching"""
    return load_checkpoint(checkpoint_path)


def main():
    st.title("Statistical Analysis - Intra-Model Variance")
    st.markdown("**Measuring non-determinism: How much do results vary across multiple trials of the same model configuration?**")

    # Load checkpoints (recursively scan results directory)
    results_dir = project_root / "results"

    # Natural sort to handle trial1, trial2, ..., trial10 correctly (recursively)
    from frontend.utils.checkpoint_utils import natural_sort_key
    available_checkpoints = sorted([f for f in results_dir.rglob("*.pkl")], key=natural_sort_key)

    if not available_checkpoints:
        st.error("❌ No checkpoint files found in results/")
        st.info("Run backtests first to generate statistical data: `python scripts/run_backtest.py --model anthropic --temperature 0.7 --run-id 1`")
        return

    # Load all checkpoints
    checkpoints_data = {}
    for checkpoint_file in available_checkpoints:
        try:
            checkpoint = load_checkpoint_cached(str(checkpoint_file))
            checkpoints_data[checkpoint_file.stem] = checkpoint
        except Exception as e:
            st.sidebar.warning(f"Failed to load {checkpoint_file.name}: {e}")

    if not checkpoints_data:
        st.warning("⚠️ No checkpoints found. Run backtests first.")
        return

    # Store checkpoints in session state for equity curve building
    if 'loaded_checkpoints' not in st.session_state:
        st.session_state['loaded_checkpoints'] = {}
    st.session_state['loaded_checkpoints'] = checkpoints_data

    # Group by model configuration
    grouped_data = group_by_model_config(checkpoints_data)

    # Sidebar: Model and Temperature selectors (like Home page)
    st.sidebar.header("Configuration")

    # Extract unique models and temperature folders from checkpoint paths
    temp_folders = {}  # {model: [temp_folders]}
    for checkpoint_file in available_checkpoints:
        # Parse path: results/{model}/{temp_folder}/{filename}
        parts = checkpoint_file.parts
        if len(parts) >= 3:
            model = parts[-3]  # e.g., 'anthropic'
            temp_folder = parts[-2]  # e.g., 'temp01' or 'temp07'

            if model not in temp_folders:
                temp_folders[model] = set()
            temp_folders[model].add(temp_folder)

    if not temp_folders:
        st.error("❌ No valid checkpoint structure found")
        return

    # Organize by temperature sections
    models_temp01 = [m for m, temps in temp_folders.items() if 'temp01' in temps]
    models_temp07 = [m for m, temps in temp_folders.items() if 'temp07' in temps]

    # Temperature section selector
    available_sections = []
    if models_temp01:
        available_sections.append("0.1")
    if models_temp07:
        available_sections.append("0.7")

    if not available_sections:
        st.error("❌ No temperature data found")
        return

    temp_section = st.sidebar.radio(
        "Temperature",
        options=available_sections,
        index=0
    )

    # Determine available models and temp folder for selected section
    if temp_section == "0.1":
        available_models = sorted(models_temp01)
        temp_folder = "temp01"
        temp_value = 0.1
    else:
        available_models = sorted(models_temp07)
        temp_folder = "temp07"
        temp_value = 0.7

    if not available_models:
        st.sidebar.warning(f"⚠️ No models available for this temperature")
        return

    # Model selector
    selected_model = st.sidebar.selectbox(
        "Model",
        options=available_models,
        index=0,
        format_func=lambda x: x.title()
    )

    # Build config key using the actual temperature value from metadata
    # The config key format is: {model}_temp{actual_temp_value}
    config_key = f"{selected_model}_temp{temp_value}"
    selected_temp = temp_value  # For backwards compatibility with rest of code

    if config_key not in grouped_data:
        st.error(f"❌ No trials found for {selected_model.title()} at temperature {selected_temp}")
        st.info(f"Available configurations: {', '.join(grouped_data.keys())}")
        return

    trials = grouped_data[config_key]
    num_trials = len(trials)

    # Show configuration summary
    st.header(f"{selected_model.title()} (Temperature: {selected_temp})")
    st.markdown(f"**Number of trials:** {num_trials}")

    if num_trials < 2:
        st.warning(f"⚠️ Need at least 2 trials to analyze variance. Currently have {num_trials} trial(s).")
        st.info(f"Run more trials with: `python scripts/run_backtest.py --model {selected_model} --temperature {selected_temp} --run-id <N>`")
        return

    st.markdown("---")

    # Determine color based on model
    config_color = '#D97757' if 'anthropic' in selected_model else '#ECF4E8'

    # === MAIN FEATURE: Equity Curve Overlays ===
    st.header("Equity Curve Overlays")

    equity_fig = create_equity_curve_overlay(grouped_data, config_key)
    if equity_fig:
        st.plotly_chart(equity_fig, use_container_width=True)

    st.markdown("---")

    # Summary Statistics Tables (Overall & Advanced Stats - Per Trial + Aggregate)
    st.header("Model Performance")

    # Create tabs for Overall Stats and Advanced Stats (like nof1.ai)
    stats_tab1, stats_tab2 = st.tabs(["Overall Stats", "Advanced Stats"])

    with stats_tab1:
        overall_stats_df = create_overall_stats_table(trials)

        # Style the dataframe with formatters while keeping numeric data
        import math
        styled_df = overall_stats_df.style.format({
            'Trial': lambda x: x if isinstance(x, str) else ("" if (isinstance(x, float) and math.isnan(x)) else ("AVERAGE" if x == 999999 else f"{int(x)}")),
            'Acc. Value': '${:,.0f}',
            'Return %': '{:.2f}%',
            'Std Dev %': '{:.2f}%',
            'Total PnL': '${:,.0f}',
            'Fees': '${:,.0f}',
            'Win Rate': '{:.1f}%',
            'Biggest Win': '${:,.0f}',
            'Biggest Loss': '${:,.0f}',
            'Sharpe': '{:.2f}',
            'No of Trades': '{:.0f}'
        })

        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True
        )

    with stats_tab2:
        advanced_stats_df = create_advanced_stats_table(trials)

        # Style the dataframe with formatters while keeping numeric data
        styled_df = advanced_stats_df.style.format({
            'Trial': lambda x: x if isinstance(x, str) else ("" if (isinstance(x, float) and math.isnan(x)) else ("AVERAGE" if x == 999999 else f"{int(x)}")),
            'Acc. Value': '${:,.0f}',
            'Avg Trade Size': '${:,.0f}',
            'Median Trade Size': '${:,.0f}',
            'Avg Hold (hrs)': '{:.1f}',
            'Median Hold (hrs)': '{:.1f}',
            '% Long': '{:.1f}%',
            'Expected Value': '${:,.0f}',
            'Median Leverage': '{:.1f}x',
            'Avg Leverage': '{:.1f}x',
            'Avg Confidence': '{:.3f}',
            'Median Confidence': '{:.3f}'
        })

        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")

    # Visualize return distribution with standard deviation bands
    st.markdown("### Return Distribution & Standard Deviation")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Return Distribution**")
        st.caption("Each dot represents a trial. Bands show expected variance (±1σ and ±2σ).")
        return_dist_fig = create_return_distribution_chart(trials)
        if return_dist_fig:
            st.plotly_chart(return_dist_fig, use_container_width=True)

    with col2:
        st.markdown("**Risk-Return Profile**")
        st.caption("Scatter plot showing the tradeoff between risk (std dev) and return.")
        risk_return_fig = create_risk_return_scatter(trials)
        if risk_return_fig:
            st.plotly_chart(risk_return_fig, use_container_width=True)

    st.markdown("---")

    # Trade P&L Distribution (KDE overlays)
    st.header("Trade P&L Distributions")
    st.markdown("""
    Each curve shows the probability distribution of individual trade P&Ls for one trial:
    - **Tight, tall curve** = Consistent trade outcomes
    - **Wide, flat curve** = Highly varied trade results
    - **Multiple peaks** = Distinct win/loss clusters
    """)

    trade_kde_fig = create_trade_pnl_kde_overlay(trials, f'{selected_model.title()} Temp {selected_temp}')
    if trade_kde_fig:
        st.plotly_chart(trade_kde_fig, use_container_width=True)
    else:
        st.info("Need at least 2 trades per trial to show distributions")

    st.markdown("---")

    # Risk Asymmetry & Distribution Analysis
    st.header("Risk Asymmetry & Distribution Shape Analysis")
    st.markdown("""
    **Understanding the risk profile and distribution characteristics of returns across trials:**

    **Metric Tracked:**
    - **Sortino Ratio**: Return / downside risk
    - **Std Dev**: Overall standard deviation (total volatility of all trade returns)
    - **Upside Deviation**: Volatility of winning trades only
    - **Downside Deviation**: Volatility of losing trades only
    - **Skewness**: Negative = left-skewed (more big losses), Positive = right-skewed (more big wins)
    - **Kurtosis**: Positive = fat tails (extreme outcomes), Negative = thin tails (consistent)
    """)

    # Summary metrics table
    st.subheader("Risk Metrics Summary")
    risk_summary_df = create_risk_metrics_summary(trials)

    # Style the dataframe with formatters while keeping numeric data
    styled_df = risk_summary_df.style.format({
        'Trial': lambda x: x if isinstance(x, str) else ("" if (isinstance(x, float) and math.isnan(x)) else ("AVERAGE" if x == 999999 else f"{int(x)}")),
        'Sortino Ratio': '{:.2f}',
        'Std Dev': '${:,.0f}',
        'Upside Dev': '${:,.0f}',
        'Downside Dev': '${:,.0f}',
        'Skewness': '{:.3f}',
        'Kurtosis': '{:.3f}'
    })

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # Upside vs Downside Volatility
    st.subheader("Upside vs Downside Volatility")

    col1, col2 = st.columns(2)

    with col1:
        updown_fig = create_upside_downside_comparison(trials)
        if updown_fig:
            st.plotly_chart(updown_fig, use_container_width=True)

    with col2:
        # Scatter plot showing asymmetry
        scatter_fig = create_volatility_scatter(trials)
        if scatter_fig:
            st.plotly_chart(scatter_fig, use_container_width=True)

    st.markdown("---")

    # Capture Ratios
    st.header("Capture Ratios - Decision Making")
    st.markdown("""
    **How well does the model capture each coin's price movements when it decides to trade that coin?**

    For each coin, we compare the model's portfolio returns vs the coin's price movements during periods when the model had an active position in that coin.

    **Interpretation:**
    - **Upside Capture > 100%** = Model captures more than the coin's gains (excellent timing, leverage, or compounding)
    - **Upside Capture < 100%** = Model captures less than the coin's gains (suboptimal timing or sizing)
    - **Downside Capture < 100%** = Model protects capital in down-markets (good risk management!)
    - **Downside Capture > 100%** = Model loses more than the coin's decline (poor timing or over-leveraged)

    **Ideal Profile:** High upside capture (>100%) + Low downside capture (<100%) = Asymmetric return profile
    """)

    # Get coins list from metadata
    coins = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE']

    capture_fig = create_capture_ratios_chart(trials, coins)
    if capture_fig:
        st.plotly_chart(capture_fig, use_container_width=True)
    else:
        st.info("Capture ratios require historical market data. Ensure data is available in data/historical/")

    st.markdown("---")

    # # Statistical Significance & Confidence Intervals
    # st.header("Statistical Significance & Performance Confidence")
    # st.markdown("""
    # **Is this model's performance statistically significant or just luck?**

    # Using {}-trial sample to determine:
    # - **95% Confidence Intervals**: Range where true mean likely falls
    # - **P-Value**: Probability results are due to chance (p < 0.05 = significant)
    # - **Probability > 0**: Likelihood of positive outcome on next trial

    # **Interpretation:**
    # - **Significant = Yes (p < 0.05)**: Performance is statistically meaningful, not random luck
    # - **Significant = No (p ≥ 0.05)**: Need more trials to confirm if performance is real
    # - **Narrow CI**: Consistent, predictable performance
    # - **Wide CI**: High variance, outcomes less predictable
    # """.format(len(trials)))

    # # Display confidence interval table
    # ci_table = create_confidence_interval_table(trials)

    # # Style the dataframe with formatters while keeping numeric data
    # styled_df = ci_table.style.format({
    #     'Mean': '{:.2f}',
    #     'CI Lower': '{:.2f}',
    #     'CI Upper': '{:.2f}',
    #     'Std Dev': '{:.2f}',
    #     'P-Value': '{:.4f}',
    #     'Prob > 0': '{:.0f}%'
    # })

    # st.dataframe(
    #     styled_df,
    #     use_container_width=True,
    #     hide_index=True
    # )

    # # Add interpretation summary
    # stats = calculate_statistical_significance(trials)
    # return_stats = stats.get('Return %')

    # if return_stats:
    #     st.markdown("### Summary")
    #     if return_stats['is_significant']:
    #         st.success(f"""
    #         **✓ Statistically Significant Performance**
    #         - Mean return: **{return_stats['mean']:.2f}%** (95% CI: {return_stats['ci_lower']:.2f}% to {return_stats['ci_upper']:.2f}%)
    #         - Probability of positive return on next trial: **{return_stats['prob_positive']:.0f}%**
    #         - This performance is **unlikely due to random chance** (p={return_stats['p_value']:.4f})
    #         """)
    #     else:
    #         st.warning(f"""
    #         **⚠ Not Yet Statistically Significant**
    #         - Mean return: **{return_stats['mean']:.2f}%** (95% CI: {return_stats['ci_lower']:.2f}% to {return_stats['ci_upper']:.2f}%)
    #         - P-value: {return_stats['p_value']:.4f} (need p < 0.05)
    #         - **Recommendation**: Run more trials to confirm if performance is real or luck
    #         """)


if __name__ == "__main__":
    main()
