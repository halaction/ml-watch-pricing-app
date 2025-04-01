# Task 3. Cross-Validation Strategies

## Introduction

This report documents the implementation and analysis of three cross-validation strategies:
- **K-Fold CV**
- **Stratified K-Fold CV**
- **Leave-one-out CV**

See full implementation in `fast-track.ipynb`. 


## Implemented Methods

### K-Fold

**Mathematical Description** 

Given dataset `X` with `n` samples, split into `k` folds of size `n / k` (approximately).  
For each fold `i` (where `i ∈ [1, k]`) do:  
- **Validation set**: Fold `i`.  
- **Training set**: Union of all folds `≠ i`.  

**Visualization**

```
K-Fold

X = [0 1 2 3 4 5 6 7 8 9]
y = [0 0 0 0 0 1 1 1 1 1]

Fold 1:
Split: [9 2 7 1 4 0 5 8] [3 6] 
Balance: [0.5 0.5] [0.5 0.5]

Fold 2:
Split: [3 6 7 1 4 0 5 8] [9 2] 
Balance: [0.5 0.5] [0.5 0.5]

Fold 3:
Split: [3 6 9 2 4 0 5 8] [7 1] 
Balance: [0.5 0.5] [0.5 0.5]

Fold 4:
Split: [3 6 9 2 7 1 5 8] [4 0] 
Balance: [0.375 0.625] [1.]

Fold 5:
Split: [3 6 9 2 7 1 4 0] [5 8] 
Balance: [0.625 0.375] [1.]
```


**Analysis**

- Random shuffling ensures no imbalance asymptotically, but folds may not preserve original distribution.
- Potentially poor generalization due to unstable distributions.


### Stratified K-Fold CV

**Mathematical Description**
For each class `c` in `y`, split samples into `k` folds while maintaining class proportions.  
- **Validation set**: Union of fold `i` across all classes.  
- **Training set**: Union of remaining folds.  


**Visualization**

```
Stratified k-Fold

X = [0 1 2 3 4 5 6 7 8 9]
y = [0 0 0 0 0 1 1 1 1 1]

Fold 1:
Split: [4 9 3 8 0 6 1 5] [2 7] 
Balance: [0.5 0.5] [0.5 0.5]

Fold 2:
Split: [2 7 3 8 0 6 1 5] [4 9] 
Balance: [0.5 0.5] [0.5 0.5]

Fold 3:
Split: [2 7 4 9 0 6 1 5] [3 8] 
Balance: [0.5 0.5] [0.5 0.5]

Fold 4:
Split: [2 7 4 9 3 8 1 5] [0 6] 
Balance: [0.5 0.5] [0.5 0.5]

Fold 5:
Split: [2 7 4 9 3 8 0 6] [1 5] 
Balance: [0.5 0.5] [0.5 0.5]
```

**Analysis**
- Preserves class balance in both training and validation sets, even with imbalanced data.
- High generalization capability due to balancing.


### Leave-one-out 

**Mathematical Description**

For each sample `i` in `X`:  
- **Validation set**: `{i}`.  
- **Training set**: `X \ {i}`.  

**Visualization**
```
Leave-one-out

X = [0 1 2 3 4 5 6 7 8 9]
y = [0 0 0 0 0 1 1 1 1 1]

Fold 1:
Split: [1 2 3 4 5 6 7 8 9] [0] 
Balance: [0.44444444 0.55555556] [1.]

Fold 2:
Split: [0 2 3 4 5 6 7 8 9] [1] 
Balance: [0.44444444 0.55555556] [1.]

Fold 3:
Split: [0 1 3 4 5 6 7 8 9] [2] 
Balance: [0.44444444 0.55555556] [1.]

Fold 4:
Split: [0 1 2 4 5 6 7 8 9] [3] 
Balance: [0.44444444 0.55555556] [1.]

Fold 5:
Split: [0 1 2 3 5 6 7 8 9] [4] 
Balance: [0.44444444 0.55555556] [1.]

Fold 6:
Split: [0 1 2 3 4 6 7 8 9] [5] 
Balance: [0.55555556 0.44444444] [1.]

Fold 7:
Split: [0 1 2 3 4 5 7 8 9] [6] 
Balance: [0.55555556 0.44444444] [1.]

Fold 8:
Split: [0 1 2 3 4 5 6 8 9] [7] 
Balance: [0.55555556 0.44444444] [1.]

Fold 9:
Split: [0 1 2 3 4 5 6 7 9] [8] 
Balance: [0.55555556 0.44444444] [1.]

Fold 10:
Split: [0 1 2 3 4 5 6 7 8] [9] 
Balance: [0.55555556 0.44444444] [1.]
```

**Analysis**
- Provides best generalization among all CV methods due to minimal bias.
- Computationally expensive (one fold for each sample), only viable for tiny datasets.


## Conclusions
- **K-Fold** is a general-purpose choice for balanced data. 
- **Stratified K-Fold** is optimal for imbalanced datasets.  
- **Leave-one-out** is best for small datasets but computationally prohibitive.  
- Use **Stratified K-Fold** for classification tasks with skewed classes.  
- Benchmark **K-Fold** and **Leave-one-out** for small datasets to evaluate bias-variance trade-offs.  

