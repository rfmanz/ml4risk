# Credit Risk Modeling Workflow Guide

This guide provides a complete, step-by-step workflow for building production-ready credit risk models using the ml4risk library. The workflow is organized in sequential steps, each building on the previous ones.

## Overview

Building a credit risk model involves several critical steps:

1. **Data Preparation**: Clean and standardize raw data
2. **Performance Analysis**: Define what constitutes a "bad" loan
3. **Reject Inference**: Handle selection bias from rejected applications
4. **Feature Engineering**: Transform features using Weight of Evidence (WOE)
5. **Model Development**: Build and train the scorecard model
6. **Score Alignment**: Ensure consistent scoring across models
7. **Monitoring**: Track model performance in production
8. **Complete Example**: End-to-end implementation

## Workflow Steps

### Step 1: Data Preparation (`01_data_preparation/`)

**Purpose**: Standardize and prepare raw credit data for modeling

**Key Files**:
- `data_dictionary.py`: Handles data dictionary management (from ml4risk)
- `example_data_prep.py`: Example usage showing data preparation

**Inputs**:
- Raw credit bureau data (CSV/Parquet)
- Data dictionary file (defines valid ranges, types, exclusions)

**Outputs**:
- Cleaned DataFrame with standardized column names
- Data quality report
- Feature metadata

**Key Concepts**:
- Handles Experian and other credit bureau formats
- Validates data types and ranges
- Identifies and handles invalid values

### Step 2: Performance Analysis (`02_performance_analysis/`)

**Purpose**: Define target variable based on loan performance

**Key Files**:
- `vintage_analysis.py`: Analyze cohort performance over time
- `roll_rate_analysis.py`: Track transitions between delinquency states
- `example_performance_analysis.py`: Example usage

**Inputs**:
- Loan performance data with payment history
- Origination dates and amounts

**Outputs**:
- Target variable definition (0=Good, 1=Bad)
- Vintage curves showing bad rate evolution
- Roll rate transition matrices

**Key Concepts**:
- **Vintage Analysis**: Track how different origination cohorts perform
- **Roll Rates**: Understand how loans move between states (Current → 30DPD → 60DPD → 90+DPD)
- **Performance Window**: Define observation and performance periods

### Step 3: Reject Inference (`03_reject_inference/`)

**Purpose**: Correct for selection bias by inferring performance of rejected applications

**Key Files**:
- `README.md`: Comprehensive guide to reject inference techniques
- `_fuzzy_augmentation.py`: Fuzzy augmentation implementation
- `example_reject_inference.py`: Example usage

**Inputs**:
- Approved applications with known performance
- Rejected applications without performance
- External data (e.g., credit bureau scores)

**Outputs**:
- Augmented dataset including inferred reject performance
- Population weights for modeling

**Techniques Available**:
1. **Fuzzy Augmentation**: Create weighted good/bad records for rejects
2. **Performance Scoring**: Use external scores as proxy
3. **Performance Supplementation**: Use similar product performance
4. **Hard Cutoff**: Assign worst rejects as bad

### Step 4: Feature Engineering (`04_feature_engineering/`)

**Purpose**: Transform raw features into risk-predictive format using WOE

**Key Files**:
- `woe.py`: Weight of Evidence transformation
- `imputer.py`: WOE-based missing value imputation
- `example_woe_transform.py`: Example usage

**Inputs**:
- Cleaned feature data
- Target variable (Good/Bad)
- Sample weights (from reject inference)

**Outputs**:
- WOE-transformed features
- Information Value (IV) for each feature
- Binning specifications for production

**Key Concepts**:
- **WOE Transformation**: Convert features to log-odds scale
- **Information Value**: Measure predictive power
- **Monotonic Binning**: Ensure risk increases/decreases consistently
- **Smart Imputation**: Use WOE relationships for missing values

### Step 5: Model Development (`05_model_development/`)

**Purpose**: Build and train the credit risk model

**Key Files**:
- `Trainer.py`: Model training framework
- `example_scorecard_dev.py`: Scorecard development example

**Inputs**:
- WOE-transformed features
- Target variable
- Population weights

**Outputs**:
- Trained model (logistic regression or ensemble)
- Model coefficients/weights
- Performance metrics

**Model Types Supported**:
- Logistic Regression (traditional scorecard)
- LightGBM (gradient boosting)
- TensorFlow/Keras (neural networks)
- Segmented models (different models for different populations)

### Step 6: Score Alignment (`06_score_alignment/`)

**Purpose**: Ensure new model scores are compatible with existing systems

**Key Files**:
- `score_alignment.py`: Score alignment utilities
- `example_score_alignment.py`: Example usage

**Inputs**:
- Old model scores
- New model scores
- Target variable

**Outputs**:
- Score mapping table
- Aligned scores maintaining same risk ranking

**Key Concepts**:
- Maintain business rules (e.g., "approve if score > 650")
- Ensure smooth model transitions
- Preserve risk ordering

### Step 7: Monitoring (`07_monitoring/`)

**Purpose**: Track model performance and trigger retraining when needed

**Key Files**:
- `monitor.py`: Monitoring framework
- `example_monitoring.py`: Example implementation

**Inputs**:
- Production scoring data
- Actual performance outcomes
- Original development data

**Outputs**:
- Performance reports
- Population Stability Index (PSI)
- Retraining triggers

**Monitoring Metrics**:
- PSI (population drift)
- Score distribution shifts
- Approval rates
- Bad rates by score band

### Step 8: Complete Example (`08_complete_example/`)

**Purpose**: Demonstrate full end-to-end workflow

**Key Files**:
- `end_to_end_example.py`: Complete workflow implementation
- `config.yaml`: Configuration parameters

## Best Practices

1. **Data Quality**:
   - Always validate against data dictionary
   - Handle missing values appropriately
   - Check for data drift

2. **Target Definition**:
   - Use sufficient performance window (typically 12+ months)
   - Consider business definitions of "bad"
   - Validate with vintage analysis

3. **Feature Engineering**:
   - Ensure minimum bin sizes (typically 5% of population)
   - Check monotonicity of WOE relationships
   - Remove features with low IV (<0.02)

4. **Model Development**:
   - Use holdout validation
   - Check for overfitting
   - Validate on out-of-time sample

5. **Production Deployment**:
   - Save all transformations for scoring
   - Implement monitoring from day 1
   - Plan for model updates

## Common Parameters

- `min_samples_leaf`: Minimum observations per WOE bin (e.g., 500)
- `min_iv`: Minimum Information Value threshold (e.g., 0.02)
- `performance_window`: Months to observe for defining bad (e.g., 12)
- `bad_definition`: Days past due to consider bad (e.g., 90)

## Troubleshooting

**Issue**: Low IV for most features
- **Solution**: Check target definition and performance window

**Issue**: Non-monotonic WOE patterns
- **Solution**: Increase `min_samples_leaf` or use manual binning

**Issue**: Score distribution shift in production
- **Solution**: Check for data quality issues or population changes

**Issue**: Reject inference changes results dramatically
- **Solution**: Validate external data quality and consider conservative approach

## Next Steps

Start with the example in `08_complete_example/end_to_end_example.py` to see how all components work together, then customize for your specific use case.