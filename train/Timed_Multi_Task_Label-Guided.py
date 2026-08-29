"""
GenAI Statement: 
Developed with the assistance of Gemini.
Focus:
1. Removed complex masking and manual loss intervention.
2. Implements basic BCE: Tool Classification + Presence Recognition.
3. Evaluates raw logits during validation to assess true model performance.
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
from model import MultiTaskTimedToolModel 
from utils import (
    evaluation_metrics_task_b, 
    save_training_plots, 
    Logger, 
    EarlyStopping
)
from config import cfg

def train_task_b_multi():
    # 0. Environment Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs('checkpoints_B_Multi', exist_ok=True)
    logger = Logger('checkpoints_B_Multi/train_multi_pure.log')
    TOOL_NAMES = cfg.TOOL_NAMES # Using names from global config
    
    logger.log(f"Starting Pure Multi-Task Training at {time.ctime()}")

    # 1. Data Loading
    train_loader = get_dataloader(split='train', batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader(split='val', batch_size=cfg.BATCH_SIZE, shuffle=False)

    # 2. Model Initialization
    # Model expects 143 input features: 128 (visual) + 7 (phase) + 7 (durations) + 1 (remaining)
    model_B = MultiTaskTimedToolModel(num_tools=cfg.NUM_TOOLS, num_phases=cfg.NUM_PHASES).to(device)

    # 3. Loss & Optimization
    # BCEWithLogitsLoss for multi-label (tools) and binary (presence) classification
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
            
            # Generate Presence Ground Truth: 1 if any tool is present, else 0
            presence_gt = (tool_gt.sum(dim=1, keepdim=True) > 0).float()
            
            # Prepare Temporal Priors (Ground Truth Guided)
            with torch.no_grad():
                # Shape: (B, 7)
                p_probs = nn.functional.one_hot(labels['curr_phase'].to(device), num_classes=cfg.NUM_PHASES).float()
                # FIXED: Use full (B, 7) durations to match model.py combined_dim logic
                p_durations = labels['phase_durations'].to(device).float()
                # Shape: (B, 1)
                p_rem = labels['remaining_time'].to(device).view(-1, 1)

            optimizer.zero_grad()
            
            # Forward pass: Dual head output
            tool_logits, presence_logits = model_B(images, p_probs, p_durations, p_rem)
            
            # Core Loss: Tool Classification + Presence Recognition (Weight 1:1)
            loss_t = criterion(tool_logits, tool_gt)
            loss_p = criterion(presence_logits, presence_gt)
            
            loss = loss_t + loss_p
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()

        # --- Validation Phase (No intervention) ---
        model_B.eval()
        all_tool_logits, all_gts = [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                v_p = nn.functional.one_hot(labels['curr_phase'].to(device), num_classes=cfg.NUM_PHASES).float()
                v_dur = labels['phase_durations'].to(device).float()
                v_re = labels['remaining_time'].to(device).view(-1, 1)

                # Get only tool logits for standard evaluation
                t_logits, _ = model_B(images, v_p, v_dur, v_re)
                
                all_tool_logits.append(t_logits)
                all_gts.append(labels['tools'].to(device))

        all_tool_logits = torch.cat(all_tool_logits, dim=0)
        all_gts = torch.cat(all_gts, dim=0)

        # Calculate Metrics
        metrics = evaluation_metrics_task_b(all_tool_logits, all_gts, tool_names=TOOL_NAMES)
        
        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['val_f1'].append(metrics['mF1'])
        history['val_mAP'].append(metrics['mAP'])
        history['val_acc'].append(metrics['mPrecision']) # Plotting Precision in Acc slot

        logger.log(f"Epoch {epoch+1}: Loss: {avg_train_loss:.4f} | mAP: {metrics['mAP']:.4f} | mF1: {metrics['mF1']:.4f}")
        
        save_training_plots(history, 'checkpoints_B_Multi/task_b_pure_metrics.png')
        early_stopping(metrics['mAP'], model_B, 'checkpoints_B_Multi/best_pure_multi.pth', logger)
        
        if early_stopping.early_stop: 
            logger.log("Early stopping triggered.")
            break

    logger.log("Pure Multi-Task Training Finished.")

if __name__ == "__main__":
    train_task_b_multi()