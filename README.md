# MSPE49600-Project03

**Name:** Billy Frazer
**Course:** MSPE 49600 – Data Analytics for Motorsports, Purdue University, Fall 2026

## Project 3: RBF Surrogates, Cross-Validation, and Neural Networks

### Overview
Using the IMS setup study from Project 2, the laps were averaged to one row per setup and split by setup into 20 training and 7 test configurations, so no leakahe. Both inputs were normalized to [0, 1] using the fixed design bounds.

A Gaussian radial basis function (RBF) model was implemented from scratch:

The implementation matched SciPy's RBFInterpolator to about 1e-14 s.

### Hyperparameter tuning
The kernel width sigma and regularization lambda were selected by leave-one-out cross-validation (LOOCV) over a 15 × 9 grid, using training data only. The selected pair was simgama approx = 0.652 and lambda = 1e-8, which reduced RMSE_CV from about 5.51 s to 0.079 s.

### Model comparison (mean Sector 3 time, 7 test setups)
| Model | RMSE_CV (s) | Test RMSE (s) | Test R² |
|---|---|---|---|
| Default RBF | 5.51 | 3.17 | −29.2 |
| Tuned RBF | 0.079 | 0.122 | 0.955 |
| Quadratic | 0.027 | 0.026 | 0.998 |

The quadratic model had the lowest cross-validation and test error and is the recommended predictor of Sector 3 time. The tuned RBF improved greatly on the default but did not beat the simpler quadratic on this smooth, sparsely sampled response.

### Neural networks (simulated lap-time data)
Two MLPRegressor models were trained on 75 of 100 simulated laps, using fuel load and tire age as inputs, and compared with a quadratic reference model on the remaining 25 laps.

### Files
- project_03.ipynb: analysis notebook
- wfrazer_pr03_grouped.csv, wfrazer_pr03_validation.csv: IMS setup data and model comparison
- wfrazer_pr03_laptime_metrics.csv, wfrazer_pr03_laptime_predictions.csv: neural-network results
- wfrazer_pr03_*.png: figures
- requirements.txt: Python packages used

### Use of AI assistance
I used Claude as a learning and coding assistant during this project.
- draft and debug Python code for the RBF model, LOOCV hyperparameter search, model comparison, and neural-network tasks.
- understand the concepts behind the work: Gaussian kernels, regularization, cross-validation, data leakage, parameters versus hyperparameters, and activation functions,
- edit the Markdown discussions.