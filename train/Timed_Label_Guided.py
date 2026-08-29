"""
GenAI Statement: 
Developed with the assistance of Gemini.
Refinement: Integrated high-performance metrics (mAP, Macro-F1) and optimized 
validation logic for Task B instrument identification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import time
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import get_dataloader
from model import TimedToolModel
from utils import (
    evaluation_metrics_task_b, 
    save_training_plots, 
    Logger, 
    EarlyStopping
)
from config import cfg

def train_task_b():
    # 0. Environment Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs('checkpoints_B', exist_ok=True)
    logger = Logger('checkpoints_B/train_task_b.log')
    
    # Use tool names from config for consistency
    TOOL_NAMES = cfg.TOOL_NAMES
    
    logger.log(f"Starting Realistic Task B Training at {time.ctime()} on {device}")

    # 1. Data Loading
    train_loader = get_dataloader(split='train', batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader(split='val', batch_size=cfg.BATCH_SIZE, shuffle=False)

    # 2. Model Initialization
    # Inputs: (Visual: 128, Phase: 7, Durations: 7, Remaining: 1) = 143
    model_B = TimedToolModel(num_tools=cfg.NUM_TOOLS, num_phases=cfg.NUM_PHASES).to(device)

    # 3. Optimizer & Scheduler
    criterion_bce = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model_B.parameters(), lr=cfg.LEARNING_RATE_B)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS_B, eta_min=1e-6)    
    early_stopping = EarlyStopping(patience=10, verbose=True) 

    # History dictionary keys aligned with utils.save_training_plots
    history = {'train_loss': [], 'val_acc': [], 'val_f1': [], 'val_mAP': []}

    # 4. Training Loop
    num_epochs = cfg.NUM_EPOCHS_B
    for epoch in range(num_epochs):
        model_B.train()
        total_train_loss = 0.0
        
        current_lr = optimizer.param_groups[0]['lr']
        pbar = tqdm(train_loader, ncols=100, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for images, labels in pbar:
            images = images.to(device)
            b = images.size(0)
            
            # Simulated Task A inputs (Using Ground Truth for guidance)
            with torch.no_grad():
                # Correctly create one-hot phase probabilities (B, 7)
                pred_phase_probs = nn.functional.one_hot(
                    labels['curr_phase'].to(device), 
                    num_classes=cfg.NUM_PHASES
                ).float()
                
                # FIXED: Model expects (B, 7) durations
                phase_durations = labels['phase_durations'].to(device).view(b, -1)
                
                # Ensure remaining time is (B, 1)
                pred_remaining = labels['remaining_time'].to(device).view(b, 1)

            optimizer.zero_grad()
            
            # Forward pass with combined temporal-spatial inputs
            tool_logits = model_B(images, pred_phase_probs, phase_durations, pred_remaining)
            
            loss = criterion_bce(tool_logits, labels['tools'].to(device).float())
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{current_lr:.1e}"})

        scheduler.step()

        # --- Validation Phase ---
        model_B.eval()
        all_logits, all_gts = [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                b = images.size(0)
                
                # Align validation inputs with training logic
                v_phase_probs = nn.functional.one_hot(
                    labels['curr_phase'].to(device), 
                    num_classes=cfg.NUM_PHASES
                ).float()
                v_durations = labels['phase_durations'].to(device).view(b, -1)
                v_remaining = labels['remaining_time'].to(device).view(b, 1)

                logits_B = model_B(images, v_phase_probs, v_durations, v_remaining)
                
                all_logits.append(logits_B) 
                all_gts.append(labels['tools'].to(device))

        # Concatenate results for overall metric calculation
        all_logits = torch.cat(all_logits, dim=0)
        all_gts = torch.cat(all_gts, dim=0)

        # Calculate metrics using Task B specific utility
        metrics = evaluation_metrics_task_b(all_logits, all_gts, tool_names=TOOL_NAMES)
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_f1'].append(metrics['mF1'])
        history['val_mAP'].append(metrics['mAP'])
        history['val_acc'].append(metrics['mPrecision']) # Precision placeholder for plotter

        # Logging results
        logger.log(f"Epoch {epoch+1}: Loss: {avg_train_loss:.4f} | mAP: {metrics['mAP']:.4f} | mF1: {metrics['mF1']:.4f}")
        
        # Specific instrument performance logging
        logger.log(f"  > Hook F1: {metrics.get('f1_Hook', 0):.4f} | Grasper F1: {metrics.get('f1_Grasper', 0):.4f}")

        # Save visualization curves
        save_training_plots(history, 'checkpoints_B/task_b_metrics.png')
        
        # Early stopping based on mAP (most sensitive to tool transition accuracy)
        early_stopping(metrics['mAP'], model_B, 'checkpoints_B/best_task_b_realistic.pth', logger)
        
        if early_stopping.early_stop: 
            logger.log("Early stopping triggered. Model converged.")
            break

    logger.log("Task B Training Process Finished.")

if __name__ == "__main__":
    train_task_b()