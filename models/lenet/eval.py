import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import datetime
import numpy as np
from model import LeNet
from dataset import BreastHistopathologyDataset
from utils import get_device

def evaluate_model(model, dataset, device=None, logger=None, save_results=True):
    """
    Evaluate the LeNet model on the given dataset
    Args:
        model: Trained LeNet model
        dataset: BreastHistopathologyDataset instance
        device: Device to use for inference (if None, will be automatically selected)
        logger: Logger instance
        save_results: Whether to save results to files
    Returns:
        Dictionary containing evaluation metrics
    """
    if device is None:
        device = get_device()
    model.eval() #fixes dropout and batchnorm
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=4) #we load the data using several subprocesses
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    if logger:
        logger.info(f"Evaluating model with batch size: 1024")
    
    with torch.no_grad(): #we don't need to compute gradients for evaluation
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images) #we get the outputs of the model
            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1) 
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # Calculate additional metrics for imbalanced datasets
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (recall + specificity) / 2
    
    # Log results
    if logger:
        logger.info(f"Evaluation results:")
        logger.info(f"  - Accuracy: {accuracy:.4f}")
        logger.info(f"  - Balanced Accuracy: {balanced_accuracy:.4f}")
        logger.info(f"  - Precision: {precision:.4f}")
        logger.info(f"  - Recall: {recall:.4f}")
        logger.info(f"  - Specificity: {specificity:.4f}")
        logger.info(f"  - F1 Score: {f1:.4f}")
        logger.info(f"  - AUC: {roc_auc:.4f}")
        logger.info(f"  - Confusion Matrix: \n{cm}")
    
    if save_results:
        # Create results directory if it doesn't exist
        current_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(current_dir, 'results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Create timestamp for unique filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics as JSON
        metrics = {
            'accuracy': float(accuracy),
            'balanced_accuracy': float(balanced_accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'f1_score': float(f1),
            'auc': float(roc_auc),
            'confusion_matrix': cm.tolist()
        }
        
        metrics_file = os.path.join(results_dir, f"metrics_{timestamp}.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        if logger:
            logger.info(f"Saved metrics to {metrics_file}")
        
        # Plot and save confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        classes = ['Non-IDC', 'IDC']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations to confusion matrix
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        cm_file = os.path.join(results_dir, f"confusion_matrix_{timestamp}.png")
        plt.savefig(cm_file)
        if logger:
            logger.info(f"Saved confusion matrix plot to {cm_file}")
        
        # Plot and save ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        roc_file = os.path.join(results_dir, f"roc_curve_{timestamp}.png")
        plt.savefig(roc_file)
        if logger:
            logger.info(f"Saved ROC curve plot to {roc_file}")
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'auc': roc_auc,
        'confusion_matrix': cm
    }

def main():
    # Load the model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'checkpoints', 'lenet_model.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    device = get_device()
    model = LeNet().to(device)
    model.load(model_path)
    
    # Load validation dataset
    dataset = BreastHistopathologyDataset(
        root_dir='data/',  
        train=False
    )
    
    # Evaluate model
    metrics = evaluate_model(model, dataset, device)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])

if __name__ == '__main__':
    main() 