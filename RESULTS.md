# Assignment 5: Health Data Classification Results

This file contains your manual interpretations and analysis of the model results from the different parts of the assignment.


## Part 1: Logistic Regression on Imbalanced Data

### Interpretation of Results

In this section, provide your interpretation of the Logistic Regression model's performance on the imbalanced dataset. Consider:

- Which metric performed best and why?
- Which metric performed worst and why?
- How much did the class imbalance affect the results?
- What does the confusion matrix tell you about the model's predictions?

*Part1
accuracy: 0.9168
precision: 0.6615
recall: 0.3007
f1_score: 0.4135
roc_auc: 0.9084

Results Interpretation:
best_metric: accuracy
worst_metric: recall
imbalance_impact_score: 0.5596999351262653*

Best Metric: Accuracy (91.68%) — high due to majority class dominance.

Worst Metric: Recall (30.07%) — low minority class detection.

Imbalance Effect: Significant; imbalance impact score of 0.56 shows degraded minority performance.

Confusion Matrix Insight: Many true negatives, few true positives, high false negatives — model favors majority class.

## Part 2: Tree-Based Models with Time Series Features

### Comparison of Random Forest and XGBoost

In this section, compare the performance of the Random Forest and XGBoost models:

- Which model performed better according to AUC score?
- Why might one model outperform the other on this dataset?
- How did the addition of time-series features (rolling mean and standard deviation) affect model performance?

*Part2
Random Forest AUC: 0.7800
XGBoost AUC: 0.7641*

Better Model: Random Forest performed better with an AUC of 0.7800 vs. XGBoost’s 0.7641.

Reason for Difference: Random Forest may better capture non-linearities and noise in smaller or moderately sized datasets without overfitting, unlike XGBoost which can be sensitive to hyperparameters.

Effect of Time-Series Features: The inclusion of rolling mean and standard deviation likely improved both models by adding temporal context, helping capture short-term trends in the data.

## Part 3: Logistic Regression with Balanced Data

### Improvement Analysis

In this section, analyze the improvements gained by addressing class imbalance:

- Which metrics showed the most significant improvement?
- Which metrics showed the least improvement?
- Why might some metrics improve more than others?
- What does this tell you about the importance of addressing class imbalance?

**Part3
Evaluation Metrics (Part 3 - Balanced):
accuracy: 0.8352
precision: 0.3769
recall: 0.8705
f1: 0.5261
auc: 0.9086

Model Comparison (% Improvement over Part 1):
accuracy: -8.90%
precision: -43.02%
recall: 189.49%
f1: nan%

Most Improved Metric: Recall increased dramatically (+189.49%), showing much better detection of the minority class.

Least Improved Metric: Precision dropped significantly (−43.02%), indicating more false positives.

Reason for Variation: Balancing the data helped the model recognize minority cases (boosting recall) but at the cost of misclassifying some majority cases (lowering precision).

Conclusion: Addressing class imbalance is crucial—it shifts the model from majority bias toward fairer performance across classes, especially important in sensitive domains like healthcare.

*

## Overall Conclusions

Summarize your key findings from all three parts of the assignment:

- What were the most important factors affecting model performance?
- Which techniques provided the most significant improvements?
- What would you recommend for future modeling of this dataset?

Key Factors: Class imbalance and temporal context were the primary drivers of model performance.

Best Techniques: SMOTE balancing (for recall gains) and time-series features (for added predictive power) were most effective.

Future Recommendations: Always address class imbalance before model training, and continue exploring temporal features or ensemble methods for further improvements.*
