"""
GenAI Statement: 
Developed with the assistance of Gemini.
Focus:
1. Implements a two-stage pipeline using a pre-trained CholecAutoModel (Task A) as a feature extractor.
2. Replaces Ground Truth (GT) priors with Task A predictions (Phase, Progress, Remaining Time).
3. Optimized for real-world inference scenarios where surgical phase is unknown.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
import time
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import get_dataloader
from model import MultiTaskTimedToolModel, CholecAutoModel 
from utils import (
    evaluation_metrics_task_b, 
    Logger, 
    EarlyStopping,
    save_training_plots
)
from config import cfg

def train_task_b_multi_pred():
    # 0. Environment Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs('checkpoints_B_Multi', exist_ok=True)
    logger = Logger('checkpoints_B_Multi/train_multi_pred_guided.log')
    TOOL_NAMES = cfg.TOOL_NAMES
    
    # Pre-trained Task A path
    PATH_A = "checkpoints_A/best_task_a.pth" 

    logger.log(f"Starting Prediction-Guided Multi-Task Training at {time.ctime()}")

    # 1. Data Loading
    train_loader = get_dataloader(split='train', batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader(split='val', batch_size=cfg.BATCH_SIZE, shuffle=False)

    # 2. Model Initialization
    # Initialize Task A as the Prior Generator
    model_A = CholecAutoModel(num_phases=cfg.NUM_PHASES).to(device)
    if os.path.exists(PATH_A):
        model_A.load_state_dict(torch.load(PATH_A, map_location=device))
        logger.log(f"Successfully loaded Pre-trained Task A from {PATH_A}")
    else:
        logger.log("CRITICAL WARNING: Pre-trained Task A not found! Using random weights.")
    
    model_A.eval()
    for param in model_A.parameters():
        param.requires_grad = False # Freeze Task A

    # Initialize Task B Model
    model_B = MultiTaskTimedToolModel(num_tools=cfg.NUM_TOOLS, num_phases=cfg.NUM_PHASES).to(device)

    # 3. Training Config
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model_B.parameters(), lr=cfg.LEARNING_RATE_B)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS_B, eta_min=1e-6)    
    early_stopping = EarlyStopping(patience=8, verbose=True)

    history = {'train_loss': [], 'val_acc': [], 'val_f1': [], 'val_mAP': []}

    # 4. Training Loop
    num_epochs = cfg.NUM_EPOCHS_B
    for epoch in range(num_epochs):
        model_B.train()
        total_train_loss = 0.0
        pbar = tqdm(train_loader, ncols=100, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for images, labels in pbar:
            images = images.to(device)
            tool_gt = labels['tools'].to(device).float()
            presence_gt = (tool_gt.sum(dim=1, keepdim=True) > 0).float()
            
            # --- Generate Prediction Priors from Task A ---
            with torch.no_grad():
                out_A = model_A(images)
                # Phase probabilities (B, 7)
                p_probs = torch.softmax(out_A['phase'], dim=1).detach()
                # Use full duration vector (B, 7) to match model combined_dim (143)
                p_durations = out_A['durations'].detach()
                # Remaining time (B, 1)
                p_rem = out_A['remaining_time'].detach()

            optimizer.zero_grad()
            
            # Forward pass using Task A's predictions as inputs
            tool_logits, presence_logits = model_B(images, p_probs, p_durations, p_rem)
            
            # Multi-task Loss
            loss_t = criterion(tool_logits, tool_gt)
            loss_p = criterion(presence_logits, presence_gt)
            loss = loss_t + loss_p
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()

        # --- Validation Phase ---
        model_B.eval()
        all_tool_logits, all_gts = [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                
                # Predict priors for validation
                out_A_val = model_A(images)
                v_p = torch.softmax(out_A_val['phase'], dim=1)
                v_dur = out_A_val['durations']
                v_re = out_A_val['remaining_time']

                t_logits, _ = model_B(images, v_p, v_dur, v_re)
                
                all_tool_logits.append(t_logits)
                all_gts.append(labels['tools'].to(device))

        all_tool_logits = torch.cat(all_tool_logits, dim=0)
        all_gts = torch.cat(all_gts, dim=0)

        # Metrics calculation
        metrics = evaluation_metrics_task_b(all_tool_logits, all_gts, tool_names=TOOL_NAMES)
        
        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['val_f1'].append(metrics['mF1'])
        history['val_mAP'].append(metrics['mAP'])
        history['val_acc'].append(metrics['mPrecision'])

        logger.log(f"Epoch {epoch+1}: Loss: {avg_train_loss:.4f} | mAP: {metrics['mAP']:.4f} | mF1: {metrics['mF1']:.4f}")
        
        save_training_plots(history, 'checkpoints_B_Multi/task_b_pred_guided_metrics.png')
        early_stopping(metrics['mAP'], model_B, 'checkpoints_B_Multi/best_pred_guided_multi.pth', logger)
        
        if early_stopping.early_stop: 
            logger.log("Validation mAP plateaued. Stopping.")
            break

    logger.log("Prediction-Guided Multi-Task Training Finished.")

if __name__ == "__main__":
    train_task_b_multi_pred()