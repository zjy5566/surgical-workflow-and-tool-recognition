"""
GenAI Statement: 
Developed with the assistance of Gemini.
Focus:
1. Integrated collection and evaluation of presence_logits for Multi-Task models.
2. Displays [Tool Existence] metrics to measure global semantic recognition accuracy.
3. Synchronized with the updated utils.py metric system.
"""

import torch
import numpy as np
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from config import cfg
from model import CholecAutoModel, TimedToolModel, ToolBaselineModel, MultiTaskTimedToolModel
from dataset import CholecAutoSequenceDataset
from utils import evaluation_metrics, evaluation_metrics_task_b, Logger

# Configuration constants
TOOL_NAMES = cfg.TOOL_NAMES

def test_all_models(task_a_path=None, task_b_path=None, multi_path=None, 
                    baseline_path=None, task_b_label_guided=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_file = f"test_results_{datetime.now().strftime('%m%d_%H%M')}.log"
    logger = Logger(log_file)
    
    mode_text = 'Label-Guided (GT)' if task_b_label_guided else 'Prediction-Guided (Task A)'
    logger.log(f"Test Session Started at {datetime.now()}\nMode: {mode_text}")

    # 1. Data Loading
    # Ensure dataset is loaded for testing phase
    test_dataset = CholecAutoSequenceDataset(split='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 2. Model Initialization and Loading
    models_to_test = {}
    
    # Task A Model (Phase Recognition)
    model_A = CholecAutoModel(num_phases=cfg.NUM_PHASES).to(device)
    if task_a_path and os.path.exists(task_a_path):
        model_A.load_state_dict(torch.load(task_a_path, map_location=device))
        model_A.eval()
        models_to_test['Task_A'] = model_A
    
    # Task B Model (Tool Identification - Single Task)
    if task_b_path and os.path.exists(task_b_path):
        m = TimedToolModel(num_tools=cfg.NUM_TOOLS, num_phases=cfg.NUM_PHASES).to(device)
        m.load_state_dict(torch.load(task_b_path, map_location=device))
        m.eval()
        models_to_test['Task_B'] = m
        
    # Multi-Task Model (Tools + Presence)
    if multi_path and os.path.exists(multi_path):
        m = MultiTaskTimedToolModel(num_tools=cfg.NUM_TOOLS, num_phases=cfg.NUM_PHASES).to(device)
        m.load_state_dict(torch.load(multi_path, map_location=device))
        m.eval()
        models_to_test['Multi_Task'] = m

    # Baseline Model (Visual only)
    if baseline_path and os.path.exists(baseline_path):
        m = ToolBaselineModel(num_tools=cfg.NUM_TOOLS).to(device)
        m.load_state_dict(torch.load(baseline_path, map_location=device))
        m.eval()
        models_to_test['Baseline'] = m

    # 3. Execution Phase
    for name, model in models_to_test.items():
        logger.log(f"\n" + "="*60 + f"\nTesting Model: {name}\n" + "="*60)
        
        # Container for outputs
        all_outputs = {'logits': [], 'gt': [], 'presence_logits': []}

        # Specific containers for Task A metrics
        if name == 'Task_A':
            phase_list, next_list, exist_list, dur_list, rem_list = [], [], [], [], []
            gt_collection = {k: [] for k in ['curr_phase', 'next_phase', 'phase_existence', 'phase_durations', 'remaining_time']}

        with torch.no_grad():
            for imgs, labels in tqdm(test_loader, desc=f"Inference {name}", ncols=100):
                imgs = imgs.to(device)
                
                if name == 'Task_A':
                    out = model(imgs)
                    phase_list.append(out['phase'])
                    next_list.append(out['next_phase'])
                    exist_list.append(out['existence'])
                    dur_list.append(out['durations'])
                    rem_list.append(out['remaining_time'])
                    for k in gt_collection.keys():
                        gt_collection[k].append(labels[k].to(device))
                
                elif name == 'Baseline':
                    logits = model(imgs)
                    all_outputs['logits'].append(logits)
                    all_outputs['gt'].append(labels['tools'].to(device))
                
                else: # Task_B or Multi_Task
                    # Prepare temporal priors
                    if task_b_label_guided:
                        # Using Ground Truth Priors
                        p_in = nn.functional.one_hot(labels['curr_phase'].to(device), num_classes=cfg.NUM_PHASES).float()
                        prog_in = labels['phase_durations'].to(device).float().view(-1, cfg.NUM_PHASES)
                        rem_in = labels['remaining_time'].to(device).view(-1, 1)
                    else:
                        # Using Task A Predicted Priors
                        out_A = model_A(imgs)
                        p_in = torch.softmax(out_A['phase'], dim=1)
                        prog_in = out_A['durations'] # (B, 7)
                        rem_in = out_A['remaining_time'] # (B, 1)

                    if name == 'Task_B':
                        logits = model(imgs, p_in, prog_in, rem_in)
                    else: # Multi_Task
                        logits, pres_logits = model(imgs, p_in, prog_in, rem_in)
                        all_outputs['presence_logits'].append(pres_logits)
                    
                    all_outputs['logits'].append(logits)
                    all_outputs['gt'].append(labels['tools'].to(device))

        # 4. Metric Calculation and Logging
        if name == 'Task_A':
            final_outs = {
                'phase': torch.cat(phase_list), 'next_phase': torch.cat(next_list),
                'existence': torch.cat(exist_list), 'durations': torch.cat(dur_list),
                'remaining_time': torch.cat(rem_list)
            }
            final_gts = {k: torch.cat(v) for k, v in gt_collection.items()}
            metrics = evaluation_metrics(final_outs, final_gts)
            
            logger.log(f"[Frame-level] Phase Acc: {metrics['phase_acc']:.2%} | Macro-F1: {metrics['phase_f1']:.4f}")
            logger.log(f"[Frame-level] Next Phase Acc: {metrics['next_phase_acc']:.2%} | Existence F1: {metrics['exist_f1']:.4f}")
            logger.log("-" * 30)
            logger.log(f"[Video-level] Progress MAE: {metrics['progress_mae']*100:.2f}%")
            logger.log(f"[Video-level] Duration Ratio MAE: {metrics['dur_ratio_mae']*100:.2f}%")
            if 'progress_corr' in metrics:
                logger.log(f"[Video-level] Progress Correlation: {metrics['progress_corr']:.4f}")

        else: # Tool Identification Models
            logits_tensor = torch.cat(all_outputs['logits'])
            gt_tensor = torch.cat(all_outputs['gt'])
            
            # Pass presence_logits for Multi_Task evaluation if available
            pres_tensor = None
            if all_outputs['presence_logits']:
                pres_tensor = torch.cat(all_outputs['presence_logits'])
            
            metrics = evaluation_metrics_task_b(logits_tensor, gt_tensor, existence_logits=pres_tensor, tool_names=TOOL_NAMES)
            
            logger.log(f"Macro mAP: {metrics['mAP']:.4f} | Macro mF1: {metrics['mF1']:.4f}")
            logger.log(f"Precision: {metrics['mPrecision']:.4f} | Recall: {metrics['mRecall']:.4f}")
            
            # Print global existence metrics if they exist
            if 'exist_Overall_F1' in metrics:
                logger.log("-" * 20 + " Tool Existence (Global) " + "-" * 20)
                logger.log(f"Global Existence F1: {metrics['exist_Overall_F1']:.4f}")
                logger.log(f"Global Existence Precision: {metrics['exist_Overall_Precision']:.4f}")
                

            logger.log("-" * 20 + " Per-Tool Frame-level F1 " + "-" * 20)
            for t_name in TOOL_NAMES:
                logger.log(f"{t_name:12}: {metrics[f'f1_{t_name}']:.4f}")

    logger.log(f"\nAll tests completed. Results saved to {log_file}")

if __name__ == "__main__":
    # Example usage for various test scenarios
    
    # Scenario 1: Evaluation with Ground Truth Priors (Upper Bound Analysis)
    test_all_models(
        task_a_path="checkpoints_A/best_task_a.pth",
        task_b_path="checkpoints_B/best_task_b_realistic.pth",
        multi_path="checkpoints_B_Multi/best_pure_multi.pth",
        baseline_path="checkpoints_baseline/best_baseline.pth",
        task_b_label_guided=True 
    )

    # Scenario 2: Evaluation with Task A Predictions (Real-world Inference)
    test_all_models(
        task_a_path="checkpoints_A/best_task_a.pth",
        multi_path="checkpoints_B_Multi/best_pred_guided_multi.pth",
        task_b_label_guided=False 
    )