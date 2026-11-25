# Learned gate parameters

## Purpose

The gate estimates whether reasoning mode will **rescue** a fast-mode error: its positive class (`1`) means the calibrated fast prediction is incorrect and the calibrated reasoning prediction is correct. It was fit as a scikit-learn pipeline with `StandardScaler` followed by `LogisticRegression`. The JSON file contains both fitted parameters and estimator settings.

## Input feature order

The order follows `features(row)` in `notebooks/current_jev_executed.ipynb`. Each input is a length-2,052 numeric vector:

| Indices | Feature | Definition |
| --- | --- | --- |
| 0 | Maximum probability | Largest element of the **raw fast-mode** option probability vector. |
| 1 | Top-two margin | Largest minus second-largest raw fast-mode probability. |
| 2 | Entropy | `-sum(p * log(max(p, 1e-12)))` over raw fast-mode option probabilities. |
| 3 | Number of options | Number of valid answer options for the example. |
| 4?2051 | Fast hidden state | 2,048 values from the final transformer hidden state at the fast prompt readout position, in model output order. |

The notebook computes these features from `row['fast']['probs']` and `row['fast']['hidden']`; do not substitute calibrated probabilities or reasoning-mode features.

## Reconstructing predictions

Read `learned_gate.json`. Let `x` be a row vector in the feature order above, and let `mean` and `scale` be the scaler arrays. Apply the stored scaling settings (the fitted scaler uses mean centering and standard deviation scaling):

```text
z[j] = (x[j] - mean[j]) / scale[j]
score = dot(coef[0], z) + intercept[0]
P(class 1 | x) = 1 / (1 + exp(-score))
```

The class order is stored in `classes` (`[0, 1]`); the probability above is for class `1`. Predict class `1` when its probability is greater than the class-0 probability, equivalently when `score > 0` for this binary model. `learned_gate.json` stores the complete `coef`, `intercept`, classes, feature counts, scaler arrays and settings, logistic-regression settings, and iteration count. Its arrays can also be loaded into a compatible sklearn pipeline if preferred.

## Limitations and assumptions

- Inputs must use the same model/tokenizer, answer-option construction, hidden-state extraction, option ordering, and feature definitions as the notebook. The 2,048-value hidden representation is model-specific.
- The gate is a fitted model, not a guarantee that reasoning will improve an individual answer. Its target is specifically the rescue event described above.
- The JSON parameters were exported from the fitted pickle and numerically checked against that loaded pipeline. The pickle emitted a scikit-learn version warning: it was saved under 1.9.1 and inspected under 1.8.0. The portable affine/logistic parameters avoid needing to unpickle the estimator, but do not eliminate assumptions about constructing input features correctly.
- This artifact does not include the base or tuned language model weights.
