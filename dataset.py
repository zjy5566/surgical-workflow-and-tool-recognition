"""
Statement on GenAI: 
This code was developed with the assistance of Gemini. 
Integrated data augmentation strategies (RandomFlip, ColorJitter, Rotation) 
specifically for surgical video sequences to improve generalization 
while maintaining temporal consistency.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
import json
import os
from pathlib import Path
import warnings
import random
from config import cfg
from tqdm import tqdm
from matplotlib import pyplot as plt

# Ignore non-critical user warnings
warnings.filterwarnings("ignore", category=UserWarning)

class CholecAutoSequenceDataset(Dataset):
    """
    Dataset class for Cholec80 video sequences.
    Samples overlapping windows for temporal Task A and identifies tools for Task B.
    """
    def __init__(self, split='train'):
        self.split = split
        
        # 1. Path Loading
        self.config_path = os.path.join("tf_cholec80", "configs", "config.json")
        if not os.path.exists(self.config_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.config_path = os.path.join(script_dir, "tf_cholec80", "configs", "config.json")

        with open(self.config_path, 'r') as f:
            self.internal_cfg = json.load(f)
        
        self.base_dir = self.internal_cfg['cholec80_dir']
        self.video_root_dir = Path(self.base_dir) / 'frames'
        self.phase_dir = Path(self.base_dir) / 'phase_annotations'
        self.tool_dir = Path(self.base_dir) / 'tool_annotations'

        # 2. Data Splitting (Cholec80 Standard)
        all_ids = list(range(1, 81))
        if split == 'train':
            self.target_video_ids = all_ids[:40]
        elif split == 'val':
            self.target_video_ids = all_ids[40:48]
        else:
            self.target_video_ids = all_ids[48:80]
        # 3. Metadata Parsing
        self.video_meta = self._extract_all_meta()
        
        # 4. Sample Construction
        self.samples = self._build_samples()

        # 5. Transformations
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.resize = T.Resize((cfg.IMG_HEIGHT, cfg.IMG_WIDTH))

    def _extract_all_meta(self):
        """Parses phase and tool text files to create lookup maps for each video."""
        meta = {}
        for v_id in self.target_video_ids:
            v_key = f"video{v_id:02d}"
            phase_file = self.phase_dir / f"{v_key}-phase.txt"
            tool_file = self.tool_dir / f"{v_key}-tool.txt"
            v_folder = self.video_root_dir / v_key
            
            if not phase_file.exists() or not v_folder.exists():
                continue

            img_files = [f for f in os.listdir(v_folder) if f.endswith('.png')]
            num_actual_imgs = len(img_files)
            if num_actual_imgs == 0: continue

            phase_lookup = {}
            durations = np.zeros(cfg.NUM_PHASES, dtype=np.float32)
            sequence = []
            phase_intervals = [] 
            total_video_frames = 0
            
            with open(phase_file, 'r', encoding='utf-8', errors='ignore') as f:
                next(f) # Skip header
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 2: continue
                    raw_idx = int(parts[0])
                    p_name = parts[1]
                    if p_name in cfg.PHASE_NAMES:
                        p_idx = cfg.PHASE_NAMES.index(p_name)
                        phase_lookup[raw_idx] = p_idx
                        durations[p_idx] += 1
                        total_video_frames += 1
                        
                        # Detect phase transitions and intervals
                        if not sequence or sequence[-1] != p_idx:
                            sequence.append(p_idx)
                            phase_intervals.append({'phase': p_idx, 'start': raw_idx, 'end': raw_idx})
                        else:
                            phase_intervals[-1]['end'] = raw_idx

            norm_durations = durations / float(total_video_frames) if total_video_frames > 0 else durations

            # Tool annotation parsing
            tool_lookup = {}
            if tool_file.exists():
                with open(tool_file, 'r', encoding='utf-8', errors='ignore') as f:
                    next(f)
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) < 8: continue
                        raw_idx = int(parts[0])
                        tool_vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                        tool_lookup[raw_idx] = tool_vec

            valid_indices = []
            aligned_phases = {}
            aligned_tools = {}
            aligned_remaining = {} 

            # Align frame indices across images, phases, and tools
            for n in range(num_actual_imgs):
                raw_target = n * 25 # Cholec80 downsampled rate
                if raw_target in phase_lookup and raw_target in tool_lookup:
                    valid_indices.append(n)
                    curr_p = phase_lookup[raw_target]
                    aligned_phases[n] = curr_p
                    aligned_tools[n] = tool_lookup[raw_target]
                    
                    # Calculate progress within current phase (1.0 to 0.0)
                    for interval in phase_intervals:
                        if interval['start'] <= raw_target <= interval['end']:
                            denom = (interval['end'] - interval['start'])
                            remaining = (interval['end'] - raw_target) / float(denom) if denom > 0 else 0.0
                            aligned_remaining[n] = remaining
                            break

            meta[v_key] = {
                'durations': norm_durations,
                'existence': (durations > 0).astype(np.float32),
                'tool_map': aligned_tools,
                'phase_map': aligned_phases,
                'remaining_map': aligned_remaining,
                'valid_indices': valid_indices,
                'sequence': sequence
            }
        return meta

    def _build_samples(self):
        """Constructs windowed sequences based on split requirements."""
        samples = []
        # Use stride of 1 for training; stride 5 for val/test to speed up evaluation
        step = 1 if (self.split == 'train') else 5
        for v_key, data in self.video_meta.items():
            valid_idxs = data['valid_indices']
            if len(valid_idxs) < cfg.SEQ_LENGTH: continue
            for i in range(cfg.SEQ_LENGTH, len(valid_idxs), step):
                window_frames = valid_idxs[i - cfg.SEQ_LENGTH : i]
                samples.append({
                    'video_key': v_key,
                    'window_indices': window_frames,
                    'curr_img_idx': window_frames[-1]
                })
        return samples

    def _apply_augmentation(self, imgs):
        """
        Applies consistent random transformations to the entire sequence.
        Crucial for maintaining temporal coherence in Task A.
        """
        # 1. Color Jitter
        if random.random() > 0.5:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            imgs = [F.adjust_brightness(img, brightness) for img in imgs]
            imgs = [F.adjust_contrast(img, contrast) for img in imgs]
            imgs = [F.adjust_saturation(img, saturation) for img in imgs]

        # 2. Random Horizontal Flip
        if random.random() > 0.5:
            imgs = [F.hflip(img) for img in imgs]

        # 3. Random Minor Rotation
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            imgs = [F.rotate(img, angle) for img in imgs]

        return imgs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        v_key = s['video_key']
        meta = self.video_meta[v_key]
        
        v_folder = self.video_root_dir / v_key
        pil_imgs = []
        
        # Load sequence of PIL images
        for n in s['window_indices']:
            img_path = v_folder / f"{v_key}_{n+1:06d}.png"
            img = Image.open(img_path).convert('RGB')
            pil_imgs.append(self.resize(img))
        
        # Apply augmentation only to training data
        if self.split == 'train':
            pil_imgs = self._apply_augmentation(pil_imgs)
        
        # Convert to Tensor and normalize
        imgs = [self.normalize(F.to_tensor(img)) for img in pil_imgs]
        seq_imgs = torch.stack(imgs)
        
        curr_n = s['curr_img_idx']
        curr_p = meta['phase_map'][curr_n]
        curr_rem = meta['remaining_map'][curr_n]
        
        # Identify the next phase in the clinical sequence
        seq = meta['sequence']
        try:
            curr_pos = seq.index(curr_p)
            next_p = seq[curr_pos + 1] if curr_pos < len(seq) - 1 else cfg.END_PHASE_ID
        except:
            next_p = cfg.END_PHASE_ID

        labels = {
            'curr_phase': torch.tensor(curr_p, dtype=torch.long),
            'next_phase': torch.tensor(next_p, dtype=torch.long),
            'phase_existence': torch.from_numpy(meta['existence']),
            'phase_durations': torch.from_numpy(meta['durations']),
            'remaining_time': torch.tensor([curr_rem], dtype=torch.float32),
            'tools': torch.from_numpy(meta['tool_map'][curr_n])
        }

        return seq_imgs, labels

def get_dataloader(split='train', batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS):
    """Utility function to create DataLoaders."""
    dataset = CholecAutoSequenceDataset(split=split)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True
    )
    return loader