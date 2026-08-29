"""
GenAI Statement: 
Developed with the assistance of Gemini.
Functionality:
1. Comprehensive comparison across all models (Baseline, Realistic, Multi-Task GT/Pred).
2. Enhanced readability with background phase-shading and Phase Name labels.
3. Optimized offset logic ensuring 5 concurrent signals (including GT) remain distinct.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import torch.nn as nn
from config import cfg
from model import CholecAutoModel, TimedToolModel, ToolBaselineModel, MultiTaskTimedToolModel
from dataset import CholecAutoSequenceDataset
from tqdm import tqdm

class SingleVideoDataset(CholecAutoSequenceDataset):
    """Dataset wrapper to isolate a single video for visualization."""
    def __init__(self, video_id=49):
        self.video_to_test = video_id
        super().__init__(split='test')

    def _extract_all_meta(self):
        # Override metadata extraction to only include the target video
        self.target_video_ids = [self.video_to_test]
        return super()._extract_all_meta()

def run_visualization(task_a_path=None, task_b_path=None, multi_gt_path=None, multi_pred_path=None, baseline_path=None, 
                      video_id=49, show_task_b=True, show_multi_gt=True, 
                      show_multi_pred=True, show_baseline=True):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TOOL_NAMES = cfg.TOOL_NAMES
    PHASE_NAMES = cfg.PHASE_NAMES
    phase_colors = plt.cm.tab10(np.linspace(0, 1, len(PHASE_NAMES)))
    
    # Path Configuration
    PATH_A = task_a_path or os.path.join("checkpoints_A", "best_task_a.pth")
    PATH_B = task_b_path or os.path.join("checkpoints_B", "best_task_b_realistic.pth")
    PATH_MULTI_GT= multi_gt_path or os.path.join("checkpoints_B_Multi", "best_pure_multi.pth")
    PATH_MULTI_Pred= multi_pred_path or os.path.join("checkpoints_B_Multi", "best_pred_guided_multi.pth")
    PATH_BASE = baseline_path or os.path.join("checkpoints_baseline", "best_baseline.pth")

    # 1. Model Initialization and Loading
    test_dataset = SingleVideoDataset(video_id=video_id)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model_A = CholecAutoModel(num_phases=cfg.NUM_PHASES).to(device)
    model_A.load_state_dict(torch.load(PATH_A, map_location=device))
    model_A.eval()

    model_B, model_Multi_GT, model_Multi_Pred, model_Base = None, None, None, None
    if show_task_b:
        model_B = TimedToolModel(num_tools=cfg.NUM_TOOLS, num_phases=cfg.NUM_PHASES).to(device)
        model_B.load_state_dict(torch.load(PATH_B, map_location=device))
        model_B.eval()
    if show_multi_gt or show_multi_pred:
        model_Multi_GT = MultiTaskTimedToolModel(num_tools=cfg.NUM_TOOLS, num_phases=cfg.NUM_PHASES).to(device)
        model_Multi_GT.load_state_dict(torch.load(PATH_MULTI_GT, map_location=device))
        model_Multi_GT.eval()
        model_Multi_Pred = MultiTaskTimedToolModel(num_tools=cfg.NUM_TOOLS, num_phases=cfg.NUM_PHASES).to(device)
        model_Multi_Pred.load_state_dict(torch.load(PATH_MULTI_Pred, map_location=device))
        model_Multi_Pred.eval()
    if show_baseline:
        model_Base = ToolBaselineModel(num_tools=cfg.NUM_TOOLS).to(device)
        model_Base.load_state_dict(torch.load(PATH_BASE, map_location=device))
        model_Base.eval()

    results = {
        'tool_gt': [], 'phase_gt': [], 'phase_pred': [],
        'tool_pred_prop': [], 'tool_multi_gt': [], 'tool_multi_pred': [], 'tool_pred_base': []
    }

    # 2. Inference Loop
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc=f"Visualizing Video {video_id}"):
            imgs = imgs.to(device)
            out_A = model_A(imgs)
            p_probs = torch.softmax(out_A['phase'], dim=1)
            p_prog = out_A['durations'] # Full (B, 7) vector
            p_rem = out_A['remaining_time']
            p_pred_idx = torch.argmax(p_probs, dim=1).item()
            
            if show_task_b:
                out_B = model_B(imgs, p_probs, p_prog, p_rem)
                results['tool_pred_prop'].append((torch.sigmoid(out_B) > 0.5).int().cpu().numpy()[0])
            
            
            if show_multi_gt:
                gt_p = nn.functional.one_hot(labels['curr_phase'].to(device), num_classes=cfg.NUM_PHASES).float()
                gt_pr = labels['phase_durations'].to(device).float()
                gt_re = labels['remaining_time'].to(device).view(-1, 1)
                out_M_gt, _ = model_Multi_GT(imgs, gt_p, gt_pr, gt_re)
                results['tool_multi_gt'].append((torch.sigmoid(out_M_gt) > 0.5).int().cpu().numpy()[0])
            if show_multi_pred:
                out_M_pred, _ = model_Multi_Pred(imgs, p_probs, p_prog, p_rem)
                results['tool_multi_pred'].append((torch.sigmoid(out_M_pred) > 0.5).int().cpu().numpy()[0])
            
            if show_baseline:
                out_Base = model_Base(imgs)
                results['tool_pred_base'].append((torch.sigmoid(out_Base) > 0.5).int().cpu().numpy()[0])
            
            results['tool_gt'].append(labels['tools'].numpy()[0])
            results['phase_gt'].append(labels['curr_phase'].item())
            results['phase_pred'].append(p_pred_idx)

    for k in results:
        if results[k]: results[k] = np.array(results[k])

    # 3. Plotting Logic
    num_frames = len(results['phase_gt'])
    mid_idx = num_frames // 2
    # Split long videos into two parts for better horizontal resolution
    segments = [(0, mid_idx, "Part_1"), (mid_idx, num_frames, "Part_2")]

    if not os.path.exists("visualization"): os.makedirs("visualization")

    for start, end, part_name in segments:
        fig, axes = plt.subplots(7, 1, figsize=(24, 20), sharex=True)
        frames = np.arange(start, end)
        gt_phases_sub = results['phase_gt'][start:end]
        phase_errors = results['phase_gt'][start:end] != results['phase_pred'][start:end]
        
        # Calculate phase intervals for label placement
        phase_change_indices = np.where(np.diff(gt_phases_sub) != 0)[0] + 1
        curr_start = 0
        phase_intervals = []
        for change_idx in phase_change_indices:
            phase_intervals.append((curr_start, change_idx, gt_phases_sub[curr_start]))
            curr_start = change_idx
        phase_intervals.append((curr_start, len(gt_phases_sub), gt_phases_sub[curr_start]))

        for i in range(7):
            ax = axes[i]
            # Background shading for Phases
            for p_idx in range(len(PHASE_NAMES)):
                where = (gt_phases_sub == p_idx)
                if np.any(where):
                    ax.fill_between(frames, -0.3, 1.4, where=where, color=phase_colors[p_idx], alpha=0.1)
            
            # Phase Name Labels on the top plot
            if i == 0:
                for p_start, p_end, p_val in phase_intervals:
                    mid_frame = start + (p_start + p_end) / 2
                    ax.text(mid_frame, 1.45, PHASE_NAMES[int(p_val)], 
                            color=phase_colors[int(p_val)], fontsize=10, 
                            fontweight='bold', ha='center', va='bottom')

            # Phase Prediction Error Indicator (Red bar at bottom)
            ax.fill_between(frames, -0.28, -0.22, where=phase_errors, color='red', alpha=0.4, label='Phase Pred Error')

            # Data Curves with Offsets for Visibility
            # GT is a thick background line
            ax.plot(frames, results['tool_gt'][start:end, i], color='black', alpha=0.1, lw=12, label='Ground Truth')
            
            if show_baseline:
                ax.plot(frames, results['tool_pred_base'][start:end, i] - 0.08, color='royalblue', lw=1.2, linestyle='--', label='Baseline')
            
            if show_task_b:
                ax.plot(frames, results['tool_pred_prop'][start:end, i] + 0.04, color='crimson', lw=1.8, label='Task_B (Realistic)')
            
            if show_multi_gt:
                ax.plot(frames, results['tool_multi_gt'][start:end, i] + 0.12, color='forestgreen', lw=1.5, label='Multi-Task (GT Guided)')
            
            if show_multi_pred:
                ax.plot(frames, results['tool_multi_pred'][start:end, i] + 0.20, color='darkorange', lw=1.5, label='Multi-Task (Pred Guided)')
            
            ax.set_ylim(-0.35, 1.6)
            ax.set_ylabel(f"{TOOL_NAMES[i]}", rotation=0, labelpad=80, va='center', fontweight='bold')
            
            if i == 0:
                ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.3), ncol=5, fontsize='small', frameon=True)

        plt.xlabel("Temporal Frame Index (Sampled)", fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join("visualization", f"comparison_video_{video_id}_{part_name}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    print(f"Visualization for Video {video_id} complete. Saved to 'visualization' directory.")

if __name__ == "__main__":

    run_visualization(video_id=49)