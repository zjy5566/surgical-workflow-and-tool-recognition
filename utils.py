"""
GenAI Statement: 
Developed with the assistance of Gemini. Designed specialized evaluation metrics 
for surgical instrument recognition, including clinical logic consistency 
analysis and precision-based false positive filtering.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, 
    recall_score, average_precision_score
)
from scipy.stats import pearsonr

# =============================================================================
# 1. LOSS FUNCTIONS
# =============================================================================

class DiceLoss(nn.Module):
    """
    Dice Loss for classification tasks to handle class imbalance.
    Commonly used in surgical data science to manage sparse tool/phase occurrences.
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # targets: (batch,) -> one_hot: (batch, num_classes)
        num_classes = logits.size(1)
        inputs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes).float()
        
        dims = (0,) # Calculate across Batch dimension
        intersection = torch.sum(inputs * targets_one_hot, dims)
        cardinality = torch.sum(inputs + targets_one_hot, dims)
        
        dice_loss = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice_loss.mean()

# =============================================================================
# 2. EVALUATION METRICS
# =============================================================================

def evaluation_metrics(outputs, labels, threshold=0.5):
    """
    Comprehensive evaluation for Task A (Phase Recognition & Progress).
    Divided into Frame-level and Video-level metrics.
    Note: Duration and Remaining Time evaluate normalized progress ratios.
    """
    results = {}

    # --- 2.1 Frame-level Metrics ---
    # Current Phase (Classification)
    phase_preds = torch.argmax(outputs['phase'], dim=1).cpu().numpy()
    phase_gt = labels['curr_phase'].cpu().numpy()
    results['phase_acc'] = accuracy_score(phase_gt, phase_preds)
    results['phase_f1'] = f1_score(phase_gt, phase_preds, average='macro', zero_division=0)

    # Next Phase Prediction
    next_preds = torch.argmax(outputs['next_phase'], dim=1).cpu().numpy()
    next_gt = labels['next_phase'].cpu().numpy()
    results['next_phase_acc'] = accuracy_score(next_gt, next_preds)

    # Phase Existence (Presence awareness)
    exist_probs = torch.sigmoid(outputs['existence']).cpu().detach().numpy()
    exist_preds = (exist_probs > threshold).astype(int)
    exist_gt = labels['phase_existence'].cpu().numpy()
    results['exist_f1'] = f1_score(exist_gt, exist_preds, average='macro', zero_division=0)

    # --- 2.2 Video-level Metrics ---
    # Phase Duration Ratio (Regression)
    # Measures the bias in predicting the proportion of each step relative to the whole surgery
    dur_preds = outputs['durations'].cpu().detach().numpy()
    dur_gt = labels['phase_durations'].cpu().numpy()
    results['dur_ratio_mae'] = np.mean(np.abs(dur_preds - dur_gt))

    # Remaining Progress (Regression)
    # Measures the precision of predicting "completion percentage" (1.0 -> 0.0)
    if 'remaining_time' in outputs:
        rem_preds = outputs['remaining_time'].cpu().detach().numpy().flatten()
        rem_gt = labels['remaining_time'].cpu().numpy().flatten()
        
        # MAE here represents "Percentage Error of Progress"
        results['progress_mae'] = np.mean(np.abs(rem_preds - rem_gt))
        
        # Pearson Correlation reflects synchronization between prediction and decay curves
        if len(rem_gt) > 1:
            try:
                corr, _ = pearsonr(rem_preds, rem_gt)
                results['progress_corr'] = corr if not np.isnan(corr) else 0.0
            except:
                results['progress_corr'] = 0.0
    
    # Compatibility field for checkpointing
    results['acc'] = results['phase_acc']
    
    return results

def evaluation_metrics_task_b(outputs, targets, existence_logits=None, threshold=0.5, tool_names=None):
    """
    Evaluation metrics for Task B (Tool Identification).
    Includes alignment logic for video-level existence vs frame-level detection.
    """
    probs = torch.sigmoid(outputs).cpu().detach().numpy()
    preds = (probs > threshold).astype(int)
    gt = targets.cpu().numpy()
    
    num_tools = gt.shape[1]
    if tool_names is None:
        tool_names = [f'Tool_{i}' for i in range(num_tools)]

    # --- 1. Frame-level Metrics ---
    per_tool_metrics = {}
    for i in range(num_tools):
        name = tool_names[i]
        try:
            ap = average_precision_score(gt[:, i], probs[:, i])
        except:
            ap = 0.0
        f1 = f1_score(gt[:, i], preds[:, i], zero_division=0)
        per_tool_metrics[f'f1_{name}'] = f1
        per_tool_metrics[f'ap_{name}'] = ap

    results = {
        'mF1': f1_score(gt, preds, average='macro', zero_division=0),
        'mAP': average_precision_score(gt, probs, average='macro') if gt.max() > 0 else 0.0,
        'mPrecision': precision_score(gt, preds, average='macro', zero_division=0),
        'mRecall': recall_score(gt, preds, average='macro', zero_division=0)
    }

    # --- 2. Presence/Existence Evaluation ---
    if existence_logits is not None:
        # preprocess predictions
        e_probs = torch.sigmoid(existence_logits).cpu().detach().numpy()
        e_preds = (e_probs > threshold).astype(int).flatten()
        
        # preprocess ground truth
        # if multiple tools exist, consider tool presence as overall existence
        if gt.ndim > 1:
            e_gt = (gt.max(axis=1) > 0).astype(int)  # Take max across tool dimension
        else:
            e_gt = gt.astype(int)

        # 3. Align lengths (to handle cases where the last batch from DataLoader might have inconsistent lengths)
        min_len = min(len(e_gt), len(e_preds))
        e_gt = e_gt[:min_len]
        e_preds = e_preds[:min_len]

        # 4. Calculate overall existence F1 (Binary since there's only one flag)
        # Here, binary measures the accuracy of "tool presence" recognition
        results['exist_Overall_F1'] = f1_score(e_gt, e_preds, average='binary', zero_division=0)
        results['exist_Overall_Precision'] = precision_score(e_gt, e_preds, average='binary', zero_division=0)

    results.update(per_tool_metrics)
    return results

# =============================================================================
# 3. TRAINING UTILITIES
# =============================================================================

class Logger:
    """Saves training logs to a local file and prints to console."""
    def __init__(self, filename):
        self.filename = filename

    def log(self, message):
        print(message)
        with open(self.filename, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

class EarlyStopping:
    """Stops training when validation performance ceases to improve."""
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = 0
        self.delta = delta

    def __call__(self, val_acc, model, path, logger):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model, path, logger)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.log(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model, path, logger)
            self.counter = 0

    def save_checkpoint(self, val_acc, model, path, logger):
        """Saves model when validation accuracy improves."""
        if self.verbose:
            logger.log(f'Validation accuracy increased ({self.val_acc_max:.4f} --> {val_acc:.4f}). Saving model...')
        torch.save(model.state_dict(), path)
        self.val_acc_max = val_acc

def save_training_plots(history, save_path):
    """Generates and saves performance plots for Loss, Phase Accuracy, and Tool mAP."""
    plt.figure(figsize=(15, 5))
    
    # Loss Convergence
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='#1f77b4', lw=2)
    plt.title('Loss Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Task A: Phase Performance
    plt.subplot(1, 3, 2)
    plt.plot(history['val_acc'], label='Phase Acc', color='#2ca02c', lw=2)
    plt.title('Task A: Phase Performance')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Task B: Tool Identification
    plt.subplot(1, 3, 3)
    plt.plot(history['val_mAP'], label='Tool mAP', color='#d62728', lw=2)
    plt.title('Task B: Tool Identification')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()