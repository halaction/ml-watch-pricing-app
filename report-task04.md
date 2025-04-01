# Task 4. Hyperparameter Optimization

## Introduction
This report compares two hyperparameter optimization methods for a `DecisionTreeRegressor`:  
1. **Grid Search** 
2. **Tree-structured Parzen Estimator (TPE)**

## Implemented Methods

### Grid Search
**Approach**: Evaluates all combinations of predefined hyperparameter values.  
**Code**:
```python fast-track copy 3.ipynb
params = {
    "model__regressor__criterion": ["squared_error", "friedman_mse"],
    "model__regressor__splitter": ["random", "best"],
    "model__regressor__max_depth": [10, 20, 30, 40],
    "model__regressor__min_samples_leaf": [5, 10, 20],
    "model__regressor__max_features": [None, "sqrt", "log2"],
}
optimize_grid(cfg, model, params, data_train, n_jobs=1, verbose=5)
```

### TPE (Optuna)
**Approach**: Uses Bayesian optimization to explore parameter space efficiently.  
**Code**:
```python fast-track copy 3.ipynb
params = {
    "criterion": CategoricalDistribution(choices=["squared_error", "friedman_mse"]),
    "splitter": CategoricalDistribution(choices=["random", "best"]),
    "max_depth": IntDistribution(low=10, high=40),
    "min_samples_leaf": IntDistribution(low=5, high=20),
    "max_features": CategoricalDistribution(choices=[None, "sqrt", 'log2']),
}
optimize_tpe(cfg, model, params, data_train, n_trials=100, timeout=300, n_jobs=1)
```

## Conclusions
1. **TPE is faster and more efficient** for large parameter spaces.  
2. **Grid Search guarantees reproducibility** but is computationally expensive.  
3. **Model sensitivity**:  
   - Most sensitive to `max_depth` and `min_samples_leaf`.  
   - Insensitive to `splitter` (both methods found `best` marginally better).  

**Recommendations**:  
- Use TPE for initial exploration, then refine with Grid Search near optima.  
- Monitor `min_samples_leaf` closely to avoid overfitting.  