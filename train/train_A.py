"""
GenAI Statement: 
Developed with the assistance of Gemini. Integrated multi-task loss weight balancing, 
CosineAnnealingLR scheduling, and monitoring logic for regression tasks (Remaining Time).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import time
import numpy as np

from dataset import get_dataloader
from model import CholecAutoModel
from utils import DiceLoss, evaluation_metrics, save_training_plots, Logger, EarlyStopping
from config import cfg

def train_task_a():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs('checkpoints_A', exist_ok=True)
    logger = Logger('checkpoints_A/train_task_a.log')
    logger.log(f"Starting Training: {time.ctime()} | Device: {device}")

    # 1. Data Loading
    train_loader = get_dataloader(split='train', batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader(split='val', batch_size=cfg.BATCH_SIZE, shuffle=False)

    # 2. Model Initialization
    model = CholecAutoModel(num_phases=cfg.NUM_PHASES).to(device)

    # 3. Loss Functions & Optimizer
    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss()
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_mse = nn.MSELoss()
    
    # Use LEARNING_RATE_A from config.py
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE_A)
    
    # LR Strategy: Cosine Annealing
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS_A, eta_min=1e-6)    
    
    # Early Stopping based on Validation Accuracy
    early_stopping = EarlyStopping(patience=7, verbose=True)

    # Initialize history with keys expected by save_training_plots
    history = {
        'train_loss': [], 
        'val_acc': [], 
        'val_mAP': [], # Placeholder for Task B compatibility in utils plot function
        'val_progress_mae': [], 
        'lrs': []
    }

    # 4. Training Loop
    num_epochs = cfg.NUM_EPOCHS_A 
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        
        current_lr = optimizer.param_groups[0]['lr']
        history['lrs'].append(current_lr)
        
        pbar = tqdm(train_loader, ncols=100, desc=f"Epoch {epoch+1}/{num_epochs} [LR: {current_lr:.6f}]")
        for images, labels in pbar:
            images = images.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)

            # --- Multi-task Loss Weighting ---
            # 1. Current Phase Classification (Weight: 1.0)
            target_phase = labels['curr_phase'].to(device)
            l_curr = criterion_ce(outputs['phase'], target_phase) + \
                     criterion_dice(outputs['phase'], target_phase)
            
            # 2. Next Phase Prediction (Weight: 0.5)
            l_next = criterion_ce(outputs['next_phase'], labels['next_phase'].to(device))
            
            # 3. Phase Existence (Weight: 0.5)
            l_exist = criterion_bce(outputs['existence'], labels['phase_existence'].to(device))
            
            # 4. Total Phase Durations (Weight: 0.1)
            l_dur = criterion_mse(outputs['durations'], labels['phase_durations'].to(device))
            
            # 5. Remaining Time Progress (Weight: 1.0)
            l_rem = criterion_mse(outputs['remaining_time'], labels['remaining_time'].to(device))
            
            # Combined Loss
            loss = l_curr + 0.5 * l_next + 0.5 * l_exist + 0.1 * l_dur + 1.0 * l_rem
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()

        # --- Validation Phase ---
        model.eval()
        val_metrics_list = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                # Move labels to device for metrics calculation
                labels_dev = {k: v.to(device) for k, v in labels.items()}
                outputs = model(images)
                
                # Get metrics (returns dict with 'phase_acc', 'progress_mae', etc.)
                m = evaluation_metrics(outputs, labels_dev)
                val_metrics_list.append(m)
        
        # Aggregate Metrics
        avg_acc = np.mean([x['phase_acc'] for x in val_metrics_list])
        avg_progress_mae = np.mean([x['progress_mae'] for x in val_metrics_list])
        avg_train_loss = total_train_loss / len(train_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_acc'].append(avg_acc)
        history['val_mAP'].append(0.0) # Dummy for plotting consistency
        history['val_progress_mae'].append(avg_progress_mae)

        # Logging
        logger.log(f"Epoch {epoch+1}: Loss: {avg_train_loss:.4f} | "
                   f"Val Acc: {avg_acc:.2%} | Progress MAE: {avg_progress_mae:.4f} | LR: {current_lr:.6f}")

        # Visualization (Saves plots to disk)
        save_training_plots(history, 'checkpoints_A/task_a_curves.png')
        
        # Check Early Stopping
        early_stopping(avg_acc, model, 'checkpoints_A/best_task_a.pth', logger)
        
        if early_stopping.early_stop:
            logger.log("Early stopping triggered. Training terminated.")
            break

    logger.log("Task A Training Process Complete.")

if __name__ == "__main__":
    train_task_a()