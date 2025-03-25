# LeNet Evaluation Report

**Model:** LeNet  
**Dataset:** Breast Histopathology  
**Input Size:** 50x50 RGB  
**Loss Function:** Weighted CrossEntropyLoss  
**Optimizer:** AdamW  
**Learning Rate Schedule:** Warmup + Cosine Decay  
**Early Stopping:** Enabled (Patience = 5)

## Performance Metrics
| Metric              | Value     |
|---------------------|-----------|
| Accuracy            | 84.7%     |
| Balanced Accuracy   | 85.2%     |
| Precision           | 68.2%     |
| Recall              | 86.5%     |
| Specificity         | 83.9%     |
| F1 Score            | 76.3%     |
| AUC                 | 92.4%     |

## Confusion Matrix
|                | Predicted Non-IDC | Predicted IDC |
|----------------|-------------------|---------------|
| Actual Non-IDC | 33,349            | 6,362         |
| Actual IDC     | 2,134             | 13,660        |

## Summary
LeNet achieves strong recall and AUC, indicating effective detection of IDC cases. However, precision is moderate due to a notable number of false positives. Overall, the model generalizes well and hits good score on our target task which is detecting as much tumor as possible (good Recall).
