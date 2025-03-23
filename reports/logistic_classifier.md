# Logistic Classifier

## Model Architecture
- Input: Flattened 50x50x3 image (7500 features)
- Batch Normalization layer
- Dropout layer (rate: 0.5)
- Linear layer (7500 → 2 classes)

## Training Configuration
- Optimizer: Adam with weight decay (1e-5)
- Learning rate: 1e-5 with adaptive reduction (20% reduction if improvement < 1%)
- Batch size: 64
- Loss function: Weighted Cross Entropy Loss (handles class imbalance)
- Number of epochs: 10

## Implementation Details
- Added batch normalization for better training stability
- Implemented dropout for regularization
- Class weights are automatically calculated to handle class imbalance
- Comprehensive evaluation metrics including:
  - Accuracy
  - Balanced Accuracy
  - Precision
  - Recall
  - Specificity
  - F1 Score
  - AUC-ROC
  - Confusion Matrix


## Results

The model achieved an **accuracy of 78.09%** and an **AUC-ROC of 0.854**, indicating strong overall performance. **Recall improved to 0.786**, significantly reducing false negatives. **Precision** is **0.586**, with an **F1 score of 0.671**, showing a better balance between sensitivity and specificity. **Balanced accuracy** is **0.782**, reflecting improved handling of class imbalance.








