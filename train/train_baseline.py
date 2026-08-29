"""
Statement on GenAI: 
Developed with the assistance of Gemini. 
Implements a Baseline Task B trainer relying solely on visual features 
from the current frame, excluding temporal or phase-based priors to 
benchmark the impact of the proposed phase-aware guidance system.
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
from model import ToolBaselineModel  # Vision-only model
from utils import (
    evaluation_metrics_task_b, 
    save_training_plots, 
    Logger, 
    EarlyStopping
)
from config import cfg

def train_baseline():
    # 0. Environment Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs('checkpoints_baseline', exist_ok=True)
    logger = Logger('checkpoints_baseline/train_baseline.log')
    logger.log(f"Starting Baseline Task B Training at {time.ctime()} on {device}")
    logger.log("Experiment Mode: Pure Visual Baseline (No Phase/Temporal/Remaining Time Priors).")

    # 1. Data Loading
    train_loader = get_dataloader(split='train', batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader(split='val', batch_size=cfg.BATCH_SIZE, shuffle=False)

    # 2. Model Initialization
    # Baseline maps visual features directly to tool label space
    model_B = ToolBaselineModel(num_tools=cfg.NUM_TOOLS).to(device)

    # 3. Loss & Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model_B.parameters(), lr=cfg.LEARNING_RATE_B)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS_B, eta_min=1e-6)    
    early_stopping = EarlyStopping(patience=7, verbose=True)

    # history keys aligned with save_training_plots expectations
    history = {
        'train_loss': [], 
        'val_f1': [], 
        'val_mAP': [], 
        'val_acc': [] # Placeholder for Precision in plotting
    }

    # 4. Training Loop
    num_epochs = cfg.NUM_EPOCHS_B
    for epoch in range(num_epochs):
        model_B.train()
        total_train_loss = 0.0
        
        current_lr = optimizer.param_groups[0]['lr']
        pbar = tqdm(train_loader, ncols=100, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for images, labels in pbar:
            images = images.to(device)
            
            optimizer.zero_grad()
            
            # --- Baseline Forward: Only Images ---
            tool_logits = model_B(images)
            
            loss = criterion(tool_logits, labels['tools'].to(device).float())
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{current_lr:.1e}"})

        # --- Validation Phase ---
        model_B.eval()
        all_logits, all_gts = [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                
                # Baseline prediction: ignores labels['curr_phase'] or other priors
                logits_B = model_B(images)
                
                all_logits.append(logits_B)
                all_gts.append(labels['tools'].to(device))

        # Metrics Calculation
        all_logits = torch.cat(all_logits, dim=0)
        all_gts = torch.cat(all_gts, dim=0)

        # Using the standard Task B metric evaluator
        metrics = evaluation_metrics_task_b(all_logits, all_gts, tool_names=cfg.TOOL_NAMES)
        
        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['val_f1'].append(metrics['mF1'])
        history['val_mAP'].append(metrics['mAP'])
        history['val_acc'].append(metrics['mPrecision'])

        logger.log(f"Epoch {epoch+1}: Loss: {avg_train_loss:.4f} | mAP: {metrics['mAP']:.4f} | mF1: {metrics['mF1']:.4f}")
        
        # Save results and update scheduler
        save_training_plots(history, 'checkpoints_baseline/task_b_baseline_curves.png')
        early_stopping(metrics['mAP'], model_B, 'checkpoints_baseline/best_baseline.pth', logger)
        
        scheduler.step()
        if early_stopping.early_stop: 
            logger.log("Baseline training reached early stopping.")
            break

    logger.log("Baseline Task B Training Complete. Comparison data ready.")

if __name__ == "__main__":
    train_baseline()