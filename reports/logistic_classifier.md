# Logistic Classifier

## Model Architecture
- Input: Flattened 50x50x3 image (7500 features)
- Batch Normalization layer
- Dropout layer (rate: 0.5)
- Linear layer (7500 → 2 classes)

## Training Configuration
- Optimizer: SGD
- Learning rate: 1e-3
- Batch size: 64
- Loss function: Cross Entropy Loss

## Implementation Details
- Added batch normalization for better training stability
- Implemented dropout for regularization
- Comprehensive evaluation metrics including:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - AUC-ROC
  - Confusion Matrix

## Results

## Results

The model achieved an **accuracy of 75.62%** and an **AUC-ROC of 0.755**, indicating solid overall performance. However, **recall was low (0.27)**, indicating many positives were missed. Precision was **0.68**, with an **F1 score of 0.39**. The confusion matrix is: [[37717, 1994], [11536, 4258]]. The model is biased toward the negative class, resulting in a high number of false negatives. This is due to the high class imbalance towards the negative. 



