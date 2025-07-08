"""
Example: Data Preparation for Credit Risk Modeling

This script demonstrates how to prepare raw credit data for modeling using
the ml4risk data preparation utilities.

Inputs:
- Raw credit bureau data (CSV/Parquet)
- Data dictionary defining valid ranges and types

Outputs:
- Cleaned DataFrame ready for feature engineering
- Data quality report
"""

import pandas as pd
import numpy as np
from pathlib import Path
from data_dictionary import ExperianDataDict

# Example 1: Load and validate data using Experian data dictionary
def prepare_experian_data(data_path, dict_path):
    """
    Prepare Experian credit bureau data using data dictionary
    
    Args:
        data_path: Path to raw Experian data file
        dict_path: Path to Experian data dictionary
        
    Returns:
        cleaned_df: DataFrame with validated and cleaned data
        validation_report: Dictionary with validation results
    """
    # Initialize data dictionary
    data_dict = ExperianDataDict()
    data_dict.load(dict_path)
    
    # Load raw data
    raw_df = pd.read_csv(data_path)
    print(f"Loaded {len(raw_df)} records with {len(raw_df.columns)} features")
    
    # Validate data types
    type_report = data_dict.validate_types(raw_df)
    
    # Validate ranges
    range_report = data_dict.validate_ranges(raw_df)
    
    # Apply exclusions
    cleaned_df = data_dict.apply_exclusions(raw_df)
    
    # Create validation report
    validation_report = {
        'type_violations': type_report,
        'range_violations': range_report,
        'records_excluded': len(raw_df) - len(cleaned_df)
    }
    
    return cleaned_df, validation_report


# Example 2: Basic data preparation without data dictionary
def prepare_generic_data(data_path):
    """
    Basic data preparation for generic credit data
    
    Args:
        data_path: Path to credit data file
        
    Returns:
        prepared_df: DataFrame with basic cleaning applied
    """
    # Load data
    if str(data_path).endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    
    print(f"Initial shape: {df.shape}")
    
    # Basic cleaning steps
    # 1. Remove duplicate records
    df = df.drop_duplicates()
    print(f"After removing duplicates: {df.shape}")
    
    # 2. Handle column names (remove special characters)
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)
    
    # 3. Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    
    # 4. Basic data quality checks
    missing_report = df.isnull().sum() / len(df)
    high_missing = missing_report[missing_report > 0.95]
    
    if len(high_missing) > 0:
        print(f"\nColumns with >95% missing values:")
        print(high_missing)
        # Remove columns with excessive missing values
        df = df.drop(columns=high_missing.index)
    
    # 5. Check for constant columns
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        print(f"\nRemoving {len(constant_cols)} constant columns")
        df = df.drop(columns=constant_cols)
    
    return df


# Example 3: Create data quality report
def create_data_quality_report(df):
    """
    Generate comprehensive data quality report
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        report: Dictionary with data quality metrics
    """
    report = {}
    
    # Basic statistics
    report['n_records'] = len(df)
    report['n_features'] = len(df.columns)
    
    # Missing value analysis
    missing_stats = df.isnull().sum()
    report['missing_values'] = {
        'total_missing': missing_stats.sum(),
        'columns_with_missing': (missing_stats > 0).sum(),
        'avg_missing_per_column': missing_stats.mean(),
        'max_missing_column': missing_stats.idxmax(),
        'max_missing_pct': missing_stats.max() / len(df)
    }
    
    # Data type summary
    report['data_types'] = df.dtypes.value_counts().to_dict()
    
    # Numeric column statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        numeric_stats = df[numeric_cols].describe()
        report['numeric_summary'] = {
            'n_numeric_features': len(numeric_cols),
            'features_with_negatives': (df[numeric_cols] < 0).any().sum(),
            'features_with_zeros': (df[numeric_cols] == 0).any().sum()
        }
    
    # Categorical column statistics
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        report['categorical_summary'] = {
            'n_categorical_features': len(cat_cols),
            'avg_cardinality': df[cat_cols].nunique().mean(),
            'max_cardinality': df[cat_cols].nunique().max(),
            'high_cardinality_features': df[cat_cols].nunique()[df[cat_cols].nunique() > 100].index.tolist()
        }
    
    return report


# Example usage
if __name__ == "__main__":
    # Example 1: Using generic preparation
    data_path = "../../data/home-credit-default-risk/train_df.parquet"
    
    if Path(data_path).exists():
        # Prepare data
        prepared_df = prepare_generic_data(data_path)
        
        # Generate quality report
        quality_report = create_data_quality_report(prepared_df)
        
        print("\n=== Data Quality Report ===")
        for key, value in quality_report.items():
            print(f"\n{key}:")
            if isinstance(value, dict):
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"  {value}")
        
        # Save prepared data
        output_path = "prepared_data.parquet"
        prepared_df.to_parquet(output_path)
        print(f"\nPrepared data saved to: {output_path}")
    else:
        print(f"Data file not found at: {data_path}")
        print("Please update the path to your data file")