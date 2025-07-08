# End-to-End Credit Risk Model Development Example

This example demonstrates the complete workflow for building a production-ready credit risk model using the ml4risk library.

## Overview

The `end_to_end_example.py` script provides a comprehensive demonstration of credit risk modeling from data generation through production deployment. This is a complete, runnable example that shows how all the components work together.

## 8-Step Workflow

### Step 1: Data Generation and Preparation
Creates synthetic credit application dataset with realistic features:
- **20,000 applications** with features like credit scores, income, debt-to-income ratios, employment history
- **Simulates approval process** where 60% of applications are approved based on risk factors
- **Generates missing values** to simulate real-world data quality issues
- **Creates approval decisions** using weighted risk scoring

**Key Point**: This step creates the base dataset - the approval decision is just part of data generation, not the final model target.

### Step 2: Performance Simulation
Generates loan performance data for approved loans:
- **24 months of performance data** for each approved loan
- **Simulates payment status transitions**: Current → 30DPD → 60DPD → 90+DPD
- **Defines target variable** as loans reaching 90+ days past due within 12 months
- **Creates realistic default patterns** based on borrower risk characteristics

### Step 3: Reject Inference
Corrects for selection bias using fuzzy augmentation:
- **Handles rejected applications** that have no performance data
- **Uses fuzzy augmentation** to create weighted good/bad records for rejects
- **Applies 3 iterations** of the inference process
- **Creates augmented dataset** with proper sample weights

### Step 4: Feature Engineering with WOE
Transforms raw features into risk-predictive format:
- **Weight of Evidence (WOE) transformation** converts features to log-odds scale
- **Information Value (IV) calculation** measures predictive power
- **Feature selection** keeps only features with IV > 0.02
- **Handles sample weights** from reject inference process

### Step 5: Model Development
Builds logistic regression scorecard:
- **Train/test split** with proper stratification
- **Logistic regression** with L2 regularization and balanced class weights
- **Sample weight support** for reject inference
- **Performance evaluation** using weighted AUC

### Step 6: Score Calculation and Alignment
Converts model predictions to traditional credit scores:
- **Base score**: 600 points
- **PDO (Points to Double Odds)**: 20 points
- **Score band analysis** showing bad rates across score ranges
- **Risk ordering validation** ensures higher scores = lower risk

### Step 7: Deployment Preparation
Saves artifacts for production deployment:
- **Model artifacts**: Trained model, WOE transformer, selected features
- **Scorecard parameters**: Base score, PDO, factor, offset, intercept points
- **WOE specifications**: JSON format for production scoring
- **Training metadata**: Date, performance metrics, bad rates

### Step 8: Monitoring Setup
Creates configuration for ongoing model monitoring:
- **Performance thresholds**: PSI warning/critical levels, AUC degradation limits
- **Score shift monitoring**: Alerts for significant score distribution changes
- **Expected metrics**: Baseline values for comparison
- **Monitoring frequency**: Daily checks recommended

## Key Files Generated

After running the example, you'll have:
- `credit_risk_model.pkl` - Complete model artifacts
- `woe_specifications.json` - WOE transformation specs for production
- `monitoring_config.json` - Monitoring configuration
- `model_summary.png` - Visualization summary

## Output Visualizations

The script generates a 4-panel summary visualization:
1. **Score Distribution**: Histogram of final credit scores
2. **ROC Curve**: Model discrimination performance
3. **Bad Rate by Score Band**: Risk ordering validation
4. **Feature Importance**: Top 10 features by points contribution

## Key Concepts Demonstrated

1. **Synthetic Data Generation**: Creates realistic credit data for demonstration
2. **Performance Definition**: Shows how to define "bad" loans based on payment behavior
3. **Selection Bias Correction**: Demonstrates reject inference techniques
4. **Feature Engineering**: WOE transformation and feature selection
5. **Model Development**: Proper train/test methodology with sample weights
6. **Score Calibration**: Converting probabilities to traditional credit scores
7. **Production Readiness**: Saving artifacts and monitoring setup

## Running the Example

```bash
cd /workspaces/ml4risk/credit_risk_workflow/08_complete_example
python end_to_end_example.py
```

## Expected Results

- **Model AUC**: Typically 0.65-0.75 (realistic for credit risk)
- **Score Range**: Approximately 400-800 points
- **Bad Rate**: Around 8-12% in approved population
- **Selected Features**: 8-12 features with IV > 0.02

## Understanding the Process

**Important**: The approval decision in Step 1 is just creating the initial dataset. The actual credit risk model predicts loan performance (will it default), not approval decisions. This mimics real-world scenarios where you have:
- Historical applications (approved + rejected)
- Performance data only for approved loans  
- Need to build a model predicting future performance

This example provides a complete template that can be adapted for real credit risk modeling projects.