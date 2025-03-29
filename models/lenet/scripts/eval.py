import os
import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import json
import datetime
import numpy as np

from ..src.model import LeNet
from ..src.dataset import BreastTumorDataset
from ..src.logger import setup_logger

def evaluate_model(model, test_loader, device, logger=None, save_results=True, timestamp=None):
    """
    Evaluate the LeNet model on the given dataset
    
    Args:
        model: Trained LeNet model
        test_loader: DataLoader for test data
        device: Device to use for inference
        logger: Logger instance
        save_results: Whether to save results to files
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    if logger:
        logger.info(f"Evaluating model on {len(test_loader.dataset)} samples")
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
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
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(current_dir, 'results/training_evaluation')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Use provided timestamp or create a new one
        if timestamp is None:
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

def evaluate_model_from_checkpoint(data_dir, checkpoint_path, batch_size=64, logger=None, save_results=True, timestamp=None):
    """
    Evaluate the model from a checkpoint
    
    Args:
        data_dir: Path to the dataset directory
        checkpoint_path: Path to the model checkpoint
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 
                         'cpu')
    
    # Setup logger if not provided
    if logger is None:
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.path.join(current_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = setup_logger(f'lenet_{timestamp}', log_type='training_evaluation')
    
    logger.info(f"Using device: {device}")
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Load the model
    model = LeNet().to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Create test dataset
    test_dataset = BreastTumorDataset(
        root_dir=data_dir,
        train=False,
        val_ratio=0.2,
        seed=42
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Evaluate model
    logger.info(f"Evaluating model on test set...")
    metrics = evaluate_model(model, test_loader, device, logger, save_results=save_results, timestamp=timestamp)
    
    return metrics

def main(args=None):
    """Main function for model evaluation"""
    if args is None:
        parser = argparse.ArgumentParser(description='Evaluate LeNet model')
        parser.add_argument('--data_dir', type=str, default='data/', help='Path to dataset')
        parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
        parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
        args = parser.parse_args()
    
    # Evaluate the model
    metrics = evaluate_model_from_checkpoint(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size
    )
    
    # Print a summary of the results
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    
    return metrics

if __name__ == '__main__':
    main() 