"""
Example: Performance Analysis for Target Definition

This script demonstrates how to analyze loan performance to define
the target variable (Good=0, Bad=1) for credit risk modeling.

Inputs:
- Loan performance data with payment history
- Origination dates and loan identifiers

Outputs:
- Target variable based on delinquency
- Vintage curves showing bad rate evolution
- Roll rate transition matrices
"""

import pandas as pd
import numpy as np
from vintage_analysis import VintageAnalysis
from roll_rate_analysis import RollRateAnalysis
import matplotlib.pyplot as plt
import seaborn as sns

# Example 1: Define target based on delinquency status
def define_target_variable(performance_df, bad_definition='90DPD', 
                         observation_window=12, performance_window=12):
    """
    Define binary target variable based on loan performance
    
    Args:
        performance_df: DataFrame with loan payment history
        bad_definition: Definition of bad ('90DPD', '60DPD', 'Charge-off')
        observation_window: Months to observe after origination
        performance_window: Months to track performance
        
    Returns:
        target_df: DataFrame with loan_id and target (0=Good, 1=Bad)
    """
    # Map delinquency status to days past due
    dpd_mapping = {
        'CURRENT': 0,
        '1-29DPD': 15,
        '30-59DPD': 45,
        '60-89DPD': 75,
        '90-119DPD': 105,
        '120+DPD': 150,
        'CHARGE_OFF': 999
    }
    
    # Define bad thresholds
    bad_thresholds = {
        '30DPD': 30,
        '60DPD': 60,
        '90DPD': 90,
        'Charge-off': 999
    }
    
    threshold = bad_thresholds.get(bad_definition, 90)
    
    # Calculate maximum delinquency for each loan
    results = []
    
    for loan_id in performance_df['loan_id'].unique():
        loan_data = performance_df[performance_df['loan_id'] == loan_id].copy()
        loan_data = loan_data.sort_values('month_on_book')
        
        # Filter to performance window
        loan_data = loan_data[
            (loan_data['month_on_book'] >= observation_window) & 
            (loan_data['month_on_book'] < observation_window + performance_window)
        ]
        
        if len(loan_data) == 0:
            continue
            
        # Get maximum DPD
        loan_data['dpd'] = loan_data['status'].map(dpd_mapping)
        max_dpd = loan_data['dpd'].max()
        
        # Define target
        target = 1 if max_dpd >= threshold else 0
        
        results.append({
            'loan_id': loan_id,
            'target': target,
            'max_dpd': max_dpd,
            'max_status': loan_data[loan_data['dpd'] == max_dpd]['status'].iloc[0]
        })
    
    target_df = pd.DataFrame(results)
    
    print(f"Target Definition Summary:")
    print(f"- Bad definition: {bad_definition}")
    print(f"- Observation window: {observation_window} months")
    print(f"- Performance window: {performance_window} months")
    print(f"- Total loans: {len(target_df)}")
    print(f"- Bad rate: {target_df['target'].mean():.2%}")
    
    return target_df


# Example 2: Vintage Analysis
def perform_vintage_analysis(loan_df, performance_df, segment_by=None):
    """
    Perform vintage analysis to understand cohort performance
    
    Args:
        loan_df: DataFrame with loan origination info
        performance_df: DataFrame with monthly performance
        segment_by: Column to segment analysis (e.g., 'product_type')
        
    Returns:
        vintage_curves: DataFrame with bad rates by vintage and month
    """
    # Initialize vintage analysis
    va = VintageAnalysis()
    
    # Prepare data
    analysis_df = performance_df.merge(
        loan_df[['loan_id', 'origination_date', 'origination_amount']],
        on='loan_id'
    )
    
    # Define bad as 90+ DPD
    analysis_df['is_bad'] = analysis_df['status'].isin(['90-119DPD', '120+DPD', 'CHARGE_OFF'])
    
    # Calculate vintage curves
    if segment_by:
        segments = loan_df[segment_by].unique()
        vintage_results = {}
        
        for segment in segments:
            segment_loans = loan_df[loan_df[segment_by] == segment]['loan_id']
            segment_data = analysis_df[analysis_df['loan_id'].isin(segment_loans)]
            
            vintage_results[segment] = va.calculate_vintage_curves(
                segment_data,
                id_col='loan_id',
                date_col='origination_date',
                performance_col='is_bad',
                weight_col='origination_amount'
            )
    else:
        vintage_results = va.calculate_vintage_curves(
            analysis_df,
            id_col='loan_id',
            date_col='origination_date',
            performance_col='is_bad',
            weight_col='origination_amount'
        )
    
    # Plot vintage curves
    va.plot_vintage_curves(vintage_results, title='Vintage Analysis - Cumulative Bad Rate')
    
    return vintage_results


# Example 3: Roll Rate Analysis
def perform_roll_rate_analysis(performance_df, from_month=3, to_month=6):
    """
    Analyze transitions between delinquency states
    
    Args:
        performance_df: DataFrame with monthly performance
        from_month: Starting month for transition
        to_month: Ending month for transition
        
    Returns:
        transition_matrix: DataFrame showing transition probabilities
    """
    # Initialize roll rate analysis
    rra = RollRateAnalysis()
    
    # Prepare data for specific months
    from_data = performance_df[performance_df['month_on_book'] == from_month][['loan_id', 'status']]
    from_data = from_data.rename(columns={'status': 'from_status'})
    
    to_data = performance_df[performance_df['month_on_book'] == to_month][['loan_id', 'status']]
    to_data = to_data.rename(columns={'status': 'to_status'})
    
    # Merge to get transitions
    transitions = from_data.merge(to_data, on='loan_id')
    
    # Calculate transition matrix
    transition_matrix = pd.crosstab(
        transitions['from_status'],
        transitions['to_status'],
        normalize='index'
    )
    
    # Order states logically
    state_order = ['CURRENT', '1-29DPD', '30-59DPD', '60-89DPD', '90-119DPD', '120+DPD', 'CHARGE_OFF']
    state_order = [s for s in state_order if s in transition_matrix.index]
    transition_matrix = transition_matrix.reindex(index=state_order, columns=state_order, fill_value=0)
    
    # Visualize transition matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(transition_matrix, annot=True, fmt='.1%', cmap='YlOrRd', cbar_kws={'label': 'Transition Probability'})
    plt.title(f'Roll Rate Matrix: Month {from_month} to Month {to_month}')
    plt.xlabel('To Status')
    plt.ylabel('From Status')
    plt.tight_layout()
    plt.show()
    
    return transition_matrix


# Example 4: Performance window analysis
def analyze_performance_windows(performance_df):
    """
    Analyze different performance windows to optimize target definition
    
    Args:
        performance_df: DataFrame with monthly performance
        
    Returns:
        window_analysis: DataFrame comparing different window definitions
    """
    results = []
    
    # Test different observation and performance windows
    for obs_window in [6, 12, 18]:
        for perf_window in [6, 12, 18, 24]:
            
            # Skip if total window exceeds available data
            if obs_window + perf_window > performance_df['month_on_book'].max():
                continue
                
            # Define target for this window
            target_df = define_target_variable(
                performance_df,
                bad_definition='90DPD',
                observation_window=obs_window,
                performance_window=perf_window
            )
            
            results.append({
                'observation_window': obs_window,
                'performance_window': perf_window,
                'total_window': obs_window + perf_window,
                'bad_rate': target_df['target'].mean(),
                'sample_size': len(target_df)
            })
    
    window_analysis = pd.DataFrame(results)
    
    # Visualize results
    pivot = window_analysis.pivot(
        index='observation_window',
        columns='performance_window',
        values='bad_rate'
    )
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.1%', cmap='viridis')
    plt.title('Bad Rate by Performance Window Definition')
    plt.xlabel('Performance Window (months)')
    plt.ylabel('Observation Window (months)')
    plt.tight_layout()
    plt.show()
    
    return window_analysis


# Example usage
if __name__ == "__main__":
    # Create sample performance data
    np.random.seed(42)
    n_loans = 1000
    
    # Generate sample loan data
    loan_df = pd.DataFrame({
        'loan_id': range(n_loans),
        'origination_date': pd.date_range('2020-01-01', periods=12, freq='M').repeat(n_loans//12 + 1)[:n_loans],
        'origination_amount': np.random.lognormal(10, 1, n_loans),
        'product_type': np.random.choice(['Personal', 'Auto', 'Student'], n_loans)
    })
    
    # Generate sample performance data
    performance_records = []
    statuses = ['CURRENT', '1-29DPD', '30-59DPD', '60-89DPD', '90-119DPD', '120+DPD', 'CHARGE_OFF']
    
    for loan_id in range(n_loans):
        # Simulate 24 months of performance
        current_status = 'CURRENT'
        
        for month in range(1, 25):
            # Simple markov chain simulation
            if current_status == 'CURRENT':
                next_status = np.random.choice(
                    ['CURRENT', '1-29DPD'],
                    p=[0.95, 0.05]
                )
            elif current_status == 'CHARGE_OFF':
                next_status = 'CHARGE_OFF'
            else:
                # Probability of improvement, staying same, or worsening
                current_idx = statuses.index(current_status)
                probs = [0.3, 0.4, 0.3]  # improve, same, worsen
                change = np.random.choice([-1, 0, 1], p=probs)
                new_idx = max(0, min(len(statuses)-1, current_idx + change))
                next_status = statuses[new_idx]
            
            performance_records.append({
                'loan_id': loan_id,
                'month_on_book': month,
                'status': current_status
            })
            
            current_status = next_status
    
    performance_df = pd.DataFrame(performance_records)
    
    print("=== Performance Analysis Examples ===\n")
    
    # Example 1: Define target variable
    print("1. Defining Target Variable")
    target_df = define_target_variable(performance_df)
    print(f"\nTarget distribution:")
    print(target_df['target'].value_counts(normalize=True))
    
    # Example 2: Analyze performance windows
    print("\n2. Analyzing Performance Windows")
    window_analysis = analyze_performance_windows(performance_df)
    
    # Example 3: Roll rate analysis
    print("\n3. Roll Rate Analysis")
    transition_matrix = perform_roll_rate_analysis(performance_df)
    
    print("\nAnalysis complete!")