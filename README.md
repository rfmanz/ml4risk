# ml4risk
--- 

Machine Learning framework for credit risk modeling

### Installation
---

```
pip install -i https://repository.sofi.com/artifactory/api/pypi/pypi/simple ml4risk
```

---

to install the development environment for rdsutils, run `./env.sh`.

---

Test using `pytest` or simply `tox` in the `rdsenv`. Todo: untangle dependency and make tox fully integrated.


### Shopping List
---

Here we list the modules developed or currently on our shopping list. We further highlight some worth-mentioning components that may be useful independently (`eg` cumulative bad rates in model_selection).

- data_preparation
    - data_dictionary
        - [x] ExperianDataDict
- model_design
    - performance_window
        - [ ] roll_rate_analysis
        - [ ] vintage_analysis
    - reject_inference
        - [ ] fuzzy_augmentation
        - [ ] performance_scoring
        - [ ] performance_supplmentation
- model_development
    - [ ] woe
- model_selection
    - [x] score_alignment
        - [x] cumulative bad rates
    - [ ] boruta
    - [ ] boruta shap
    - [ ] FeatureSelector
- monitor
- deployment
