# Bank-marketing-classification
Bank Marketing – Term Deposit Subscription Classification

TANKALA KAVYASRI - 2026CT05039 -ML Assignment

Problem statement
A bank sells term deposits by telephone. Calling every client is expensive, so the bank wants to predict from client and campaign data whether a client will subscribe (`y` = yes/no). This is an imbalanced binary classification problem. Five classifiers are implemented on the same data and compared with Accuracy, AUC, Precision, Recall, F1 and MCC: Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes and Random Forest.

Dataset description

- Source:UCI Machine Learning Repository – Bank Marketing, https://archive.ics.uci.edu/dataset/222/bank+marketing, file
- Size:45,211 rows, 16 input features + target `y`.
- **Why chosen:** <WRITE 2–3 SENTENCES IN YOUR OWN WORDS – real business problem, mixed feature types, hidden missing values, class imbalance; meets the ≥12 features / ≥1,000 rows rule>.
- Target:`y` (1 = subscribed, 0 = not subscribed).
- Numeric features: age, balance, day, duration, campaign, pdays, previous.
 -Categorical features: job, marital, education, default, housing, loan, contact, month, poutcome.
- Class distribution: no = 39,922 (88.30 %), yes = 5,289 (11.70 %), imbalance ratio 7.55 
- Missing values: there are no real NaNs; the bank wrote `unknown` instead. After converting to NaN: job 288 (0.64 %), education 1,857 (4.11 %), contact 13,020 (28.80 %), poutcome 36,959 (81.75 %).
  - Handling: Dropped `contact` and `poutcome` because 29 % / 82 % is missing, removed the 288 rows with unknown `job`. Result: 44,923 rows × 13 input features.
- Duplicates: 0 exact duplicates found.
- Preprocessing: 80:20 split (`andom_state=42) → train 35,938 / test 8,985 rows, class ratio 88.3 : 11.7 in both. Numeric features: median imputer + StandardScaler. Categorical features: one-hot encoding (39 model inputs in total). All transformers are fitted on the training set only and merely applied to the test set.

Outputs: `results/model_comparison.csv`, `results/*.png`, `data/test_data.csv`.
 Models used:

All five models use the same preprocessed training data and the same 20 % test set. Each model is defined in its own file under `model/`.

| Model | File | Configuration |
|---|---|---|
| Logistic Regression | `model/logistic_regression.py` | C=1.0, max_iter=1000, random_state=42 |
| Decision Tree | `model/decision_tree.py` | max_depth=8, min_samples_leaf=20, random_state=42 |
| K-Nearest Neighbors | `model/knn.py` | n_neighbors=15, weights="distance" |
| Naive Bayes (Gaussian) | `model/naive_bayes.py` | defaults (var_smoothing=1e-9) |
| Random Forest (ensemble) | `model/random_forest.py` | n_estimators=200, min_samples_leaf=2, random_state=42 |

Comparison table (test set, positive class = yes):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8939 | 0.8883 | 0.6017 | 0.2759 | 0.3783 | 0.3591 |
| Decision Tree | 0.8918 | 0.8676 | 0.5542 | 0.3844 | 0.4539 | 0.4042 |
| K-Nearest Neighbors | 0.8936 | 0.8713 | 0.6017 | 0.2674 | 0.3702 | 0.3532 |
| Naive Bayes (Gaussian) | 0.8558 | 0.7978 | 0.3928 | 0.4272 | 0.4093 | 0.3277 |
| Random Forest | 0.9016 | 0.9220 | 0.6299 | 0.3853 | 0.4782 | 0.4431 |

Visualisations:
![Metric bars](results/03_metric_bars.png)
![Grouped metrics](results/04_grouped_metrics.png)
![Heat-map](results/05_metric_heatmap.png)
![ROC curves](results/06_roc_curves.png)
![Confusion matrices](results/07_confusion_matrices.png)

Observations:
Random Forest :Best Accuracy (0.902), AUC (0.922), Precision (0.630), F1 (0.478) and MCC (0.443); recall (0.385) is on par with the single tree. Averaging 200 trees reduces variance, although the train–test gap (0.973 vs 0.902) shows it still partly memorises the training data.
Logistic Regression : Second-best AUC (0.888) and no overfitting (train 0.8933 vs test 0.8939). Precision is decent (0.602) but recall is low (0.276): of 1,051 real subscribers it finds only 290, because a linear model on a 12 %-positive class rarely pushes probabilities above 0.5.
Decision Tree: Finds more subscribers than LR (recall 0.384, 404 of 1,051) and has the second-best F1 (0.454) and MCC (0.404), at the cost of precision (0.554, 325 false alarms). The depth limit of 8 keeps the train–test gap small (0.018). AUC is lower (0.868) because few leaves give coarse probabilities.
K-Nearest Neighbors :Lowest recall (0.267) and F1 (0.370). Most of a client's 15 neighbours are "no", so the vote rarely turns "yes". Training accuracy is 1.0 versus 0.894 on test (gap 0.106), which is memorisation with distance weighting. Slow at prediction time.
Naive Bayes (Gaussian) :Lowest accuracy (0.856, below the 0.883 baseline), AUC (0.798), precision (0.393) and MCC (0.328). Recall (0.427) is higher than LR and KNN, but it makes 694 false alarms. Its independence and Gaussian assumptions do not hold for one-hot and skewed features.

Overall: accuracy hardly separates the models (four of five are within 1 point of each other), so MCC, F1, recall and AUC carry the comparison. Random Forest is the best overall; Naive Bayes is the only model that clearly trades precision for recall; every model misses at least 57 % of subscribers at the 0.5 threshold.
