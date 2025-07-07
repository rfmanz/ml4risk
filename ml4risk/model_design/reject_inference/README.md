# Performance Inference

* [Introduction](#Introduction)
  - [What is Performance Inference](#What-is-Performance-Inference)
  - [Objectives of Performance Inference](#Objectives-of-Performance-Inference)
  - [Value of Performance Inference](#Value-of-Performance-Inference)
* [Overview of PI Techniques](#Overview-of-PI-Techniques)
  - [Inference Inputs](#Inference-Inputs)
  - [Augmentation](#Augmentation)
  - [Extropolation](#Extropolation)
  - [Reclassification](#Reclassification)
  - [Performance Scoring](#Performance-Scoring)
  - [Proxy Performance](#Proxy-Performance)
* [Evaluation](#Evaluation)
  - [Evaluation Framework](#Evaluation-Framework)
  - [Evaluation Metrics](#Evaluation-Metrics)
  - [Analytical Tools](#Analytical-Tools)
* [Our Approach](#Our-Approach)
  - [Assumptions](#Assumptions)
  - [Key Features](#Key-Features)
  - [Inference Flow](#Inference-Flow)
* [References](#References)

## Introduction

### What is Performance Inference

Underwriting modeling relies on historical performance of past applicants to rank order potential new applicants, but the outcome performance is missing for many past applicants, either because:

(1) the application was declined (_rejects_), or
(2) the application was approved, but the credit did not taken (_unbooked_).

_Performance Inference_ is a process for assigning an inferred status (G/B) to applicants rejected or unbooked for credit. Equivalent to saying “if these applicants had been accepted or booked, this is how they would have performed”.

### Objectives of Performance Inference

* Reduce bias
  - Need statistically sound representative model development sample (selection bias)
  - Need model to be effective for applications with reject profile (prediction bias)

* Support business objectives
  - identify rejects with good performance to increase the approval rate without compromising the credit quality, or
  - identify bads among the TTD population by maintaining the same level of approval rate

### Value of Performance Inference

The benefit of performance inference varies, depending upon the past and current process (a combination of scores, policies, and judegmental decisions), population distributions, reject rate, booking rate, and the degree to which the underlying assumptions for reject inference are satisfied.
It will be greater where:

(1) TTD is a risk-hetrogeneous population
(2) reject rates are high,
(2) booking rates are low,
(3) the past/current process is effective, or
(4) there are significant differences between the data used in past decisions and what is available for the model development.


## Overview of PI Techniques

There are many generally accepted PI techniques. Each has underlying assumption and limitation.

### Inference Inputs

Performance inference makes use of the two sources of information:

- external information (credit bureau score and data)
  - performance scoring
  - Performance supplement
- internal information based on the intermediate model-based
  a) an accept/reject model for the TTD population,
  b) a booked/nonbooked model for the accepted population, and
  c) a known good/bad (KGB) model for the booked population.

>_Note_: Known good/bad models are the cornerstones of Performance Inference, and whether the others are used will depend upon the circumstances.

### Augmentation

* Concept:
  - makes use of accept/reject model and reweighting
  - apply a weight to scale up the booked population to the TTD population,
    the weight is simply - the inverse of the approval rate: w = 1/p(approve)
    so the booked population has the same score distribution as the TTD population,

* Assumption:
  the booked and reject applicants have the same performance by score

* Limitation:
  - May not work well on sample with high reject rates.
  For example, it is impossible on the reject region where the probability of approval is very small or near zero (weight = 1/p(approve) = 1/0 = DIV0!).
  - the bias due to “cherry picked” applicantions will be grossly inflated since thy are quite small in the booked population.

* When to use:
  - overlap between booked/unbooked or booked/reject


### Extropolation

* Concept:
  - makes use of known good/bad (KGB) model and parcelling
  - rely on booked data to make educated guesses about rejects and unbooked by identifying rejects or unbooked that are similar to booked.
  - Booked performance is “extrapolated” into the reject und/or unbooked space, this can be done by:

  a) _Hard cutoff_: use of a cut-off score,
  b) use of default probabilities to assign cases at random within score ranges or
  c) _Fuzzy parceling_: replicate each reject/unbooked and adjust the weights by the default probabilities (“fuzzy” parcelling).

* Assumption:
  Rejected or unbooked applicants perform similarly to the booked population within the same score band.

* Limitation:
  The accuracy of assigning bad and good performance is dependent on the predictive power of the score used for parceling.

- When to use:
  -

### Reclassification

* Concept:
  - makes use of accept/reject, KGB and reclassification
  - the worst cases of rejects are selected, reclassified as accepts, and a bad status is assigned.
  - once identified, these rejects-turned-bads (‘RTB’) are assigned a weight so that their contribution to resulting total bads (observed + inferred bads) is controlled.
    For example: control the modeling population, so that known bads/reclassified bads = 70/30

* Assumption:
  The highest-risk applicants are of such high risk that they would never be accepted under any conditions.

* Limitation:
  - will result in a small swap-set because past decisions are replicated, thereby decreasing opportunities to increase approval rates or decrease bad rates.
  - May not work well with high reject rates

* When to use:
  A more conservative approach if an agreement with past decisions is more important than maximum risk prediction.


### Performance Scoring

* Concept:
  - make use of a credit bureau (CB) score, KGB, and score calibration
  - Use Vantage or FICO score as a customer-level performance for the TTD population
  - The CB score is then calibrated to the KGB of the booked population using a regression function.
    A simple model might be:
      logOdds = B0 + B1*CB_SCORE
  - For a given reject or unbooked application, we can then compute its probability of being Good as
      p(Good) = 1 / (1 + exp{-(B0 + B1*CB_SCORE)})
  - These estimates are then used in an iterative process to infer the product specific performance for the TTD population.

* Assumption:
  the CB score contains information about their likely performance, had they been granted the credit.
  that is, the booked and reject/unbooked applications have the same performance by the CB score

* Limitation:
  - results in a small score range on the rejects/unbooked (?)

* When to use:
  as alternative PI method or validation for model-based PI methods

### Performance Supplementation

* Concept:
  - make use of a credit bureau (CB) proxy performance, KGB, and score calibration
    - use a proxy performance for rejects/unbooked
      the proxy performance is based on the similar tradelines coming from other lenders over the same observation period.

* Assumption:
  - similar lenders: the applicant’s repayment performance with another lender is commensurate with the performance of the consumer had the applicant been accepted.
  - similar products: default on other products is equivalent to default on the interested product;

* Limitation:
  - low match rates
  - time consuming for data preparation

- When to use:
  - for a more realistic approach to the market situation
  - use as validation for model-based PI methods


## Evaluation

### Evaluation Framework

* Empirical Driven
  - Use two different actual performance outcomes for the assessment:
  (1) KGB data on masked marginal booked population
    - identify booked applicants who were only marginally above the cut-off score
    - mask their KGB performance and treated them as rejects or unbooked
    - compare their true performance with their (both proxy and model-based) inferred performance

  (2) Proxy performance data on rejects and/or unbooked
    - use the proxy performance to access the inferred performance from model-based PI methods
    - compare their proxy performance with their inferred performance

* Conceptual Soundness
  - Based on the concept of score vaiability
    - The "_viable_" a of PI is tested by training a scoring model M on the TTD and separately estimating the log(Odds) of M across the known and unknown sub-populations.
    - The PI is viable if theese two log(Odds) regression lines match in slope and intercept (e.g. are aligned), indicating the reconstruction of the development data is self-consistent across the TTD population.


### Evaluation Metrics

* Conceptual soundness
    - the variability tset
    - discriminatory power (measured by KS, ROC, and Gini)
    - accuracy - measured by comparing predicted/inferred bad rate to the actual/proxy bad rate

* Discriminatory power
  - measured by KS, ROC, and Gini)

* Accuracy
  - measured by comparing predicted/inferred bad rate to the actual/proxy bad rate

* Known-to-inferred odds ratio (KI)
  - KI is a statistic used to access whether the performance inferred risk is appropriate.

    KI = (Good(known)/Bad(known))/(Good(inferred)/Bad(inferred))

    The higher the value, the greater the needs for performance inference.
    Rule of thumb: 2 <= KI <= 4+


### Analytical Tools

* Population Flow

  A tool used to illustrate the underwriting process and to show the distribution of cases different performance categories changes brought about by performance inference.

  - All applicants
    - the “Through-The-Door” (TTD) population to which the scoring system will be applied
      - rejects
      - accepts
        - unbooked
        - booked
          - good
          - bad
          - indeterminate

  - Pre- vs. Post-inference
    - the information values of features will change, and
    - reassign rejects to other groups will affect the good/bad odds

    The value of performance will depend on if this potential changes can lead to
    1) Segregating the truly good accounts from the bad accounts (ROC, GINI statistics)
    2) Accuracy with respect to predicted bad rate to final observed bad rate

* Swap-set Analysis

  Performance inference’s final outcome is a modified dataset, with non-policy rejects assigned to the other possible categories: unbooked/good/bad, etc. The outcome is often displayed as a transition matrix, with historical accepts and rejects on one axis and new accepts and rejects on the other, where cut-off was chosen to retain the same reject rate. The “swap set” is those cases that would have been accepted in the past but are now rejected, and vice versa.

* WOE Analysis



## Our Approach

### Assumptions

* Performance Inference is depends on
  - past decisions (aproval rate and booking rate)
  - inference methods used - they are valid under assumptions,
  - the quality of the intermediate models developed,
  - available internal and external data

* The reject population is more risky that the approved population.

* The unbokked population has similar risk profile as the booked population.

* ......

### Key Features

* Segment-based

  - Population with lower approval and booked rate will be more likely to be influenced by selection bias
  - Each segment will contain different degree of sample selection bias.
  - PI will be performed at the segment level, rather than running inference on the population as a whole, so that the PI process would be as accurate as possible.


* Triple Score Inference

  The central idea is to inject the historical pattern of reasonable accept/reject and booking decisions, along with observed good and bad repayment behavior, to speed inference and improve model quality.

  For the best future decisions, the underwriting models must rise above the statistical bias inherent in these historical decisions.

  Where there are other possible outcomes, they should be recognised; not just Good and Bad but also unbooked or indeterminate.

  When developing intermediate models, special attention should be given to fetures that are not only highly indicative in known performance, but also in the inference decision.

* Iterative Process

PI is an iterative process. This process continues until the odds-to-score fits converge, signalling a succesful completion of PI. Below are the steps to achieve a cridible PI:

(1) Develop intermidaite models, taking care to create bins that spread out both the knowns and unknowns
(2) Use the score(s) to assign outcome probabilities to the unknowns
(3) Identify bias and evaluate the results by analyzing multiple reports including
    - WOE reports (log odds patterns for KGB, AR, All and Inferred)
    - Approval rate report (historical vs. projected approval patterns)
    - Booking rate report (historical vs. projected booking patterns)
(3) Adjust for the bias through manipulation or rebuilding of the intermidate models
    - Explanatory variables can be chosen independent of the previous iteration.
(4) Repeat the process of adjusting the models(s) and running inference until
    - the evaluation reports indicate that the bias in the known pattern has been removed from the ALL and Inferred populations (FICO).
    - computed decline probabilities converge/stabilize where diminishing changes in model performance lends evidence of model stability (Experian).


 Convergence: Stopping criteria for model iterations.
– Probability: Changes in probabilities of individual records are aggregated When this change is sufficiently small, the iterative process stops.
– Parameter estimates: Examination of the change in parameter estimates of explanatory variables. Problematic should explanatory variables change from successive iterations.
– Performance metrics: An overall model metric, such as KS or Gini, is examined for convergence.

### Inference Flow

TBD


### References

[1] Raymond A. Anderson,  reject inference (2016)
[2] FICO, Building Powerful Predictive Scorecards (2014)
[3] Experian, Reject inference — iterative reclassification


