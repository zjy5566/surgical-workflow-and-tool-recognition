import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CholecAutoModel(nn.Module):
    """
    Task A Model: Temporal workflow recognition and progress prediction.
    Architecture: ResNet-18 Feature Extractor + LSTM + Multi-head Regression/Classification.
    """
    def __init__(self, num_phases=7):
        super(CholecAutoModel, self).__init__()
        # Backbone: ResNet18 (Pretrained on ImageNet)
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Remove the final FC layer to use as a spatial encoder
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Temporal Component: LSTM (Input: 512-dim features, Hidden: 256-dim)
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, batch_first=True)
        
        # --- Task A Output Heads ---
        self.phase_head = nn.Linear(256, num_phases)           # Predict current phase (Logits)
        self.next_phase_head = nn.Linear(256, num_phases + 1)  # Predict next phase (including 'None' state)
        self.existence_head = nn.Linear(256, num_phases)       # Predict presence of phases in the video
        self.duration_head = nn.Linear(256, num_phases)        # Predict normalized duration of each phase
        
        # Regression Head: Predict remaining time ratio of current phase (1.0 -> 0.0)
        self.remaining_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() # Constrain output to [0, 1] range
        )

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        # Flatten sequence for backbone processing: (B*L, C, H, W)
        x = x.view(batch_size * seq_len, c, h, w)
        
        features = self.feature_extractor(x)
        # Reshape back to sequence: (B, L, 512)
        features = features.view(batch_size, seq_len, -1)
        
        lstm_out, _ = self.lstm(features)
        # Extract the hidden state of the last frame in the sequence
        last_out = lstm_out[:, -1, :] 
        
        return {
            'phase': self.phase_head(last_out),
            'next_phase': self.next_phase_head(last_out),
            'existence': self.existence_head(last_out),
            'durations': self.duration_head(last_out),
            'remaining_time': self.remaining_head(last_out)
        }

class TimedToolModel(nn.Module):
    """
    Task B Model (Single Task): Tool identification using visual features and Task A priors.
    """
    def __init__(self, num_tools=7, num_phases=7):
        super(TimedToolModel, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.visual_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.visual_fc = nn.Linear(512, 128)
        
        # Combined Dimension Calculation:
        # 128 (Visual) + num_phases (Phase Probs) + num_phases (Phase Durations) + 1 (Remaining Time)
        combined_dim = 128 + num_phases + num_phases + 1 
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_tools)
        )

    def forward(self, x, pred_phase_probs, phase_durations, pred_remaining):
        # Extract the last frame if sequence input is provided
        if len(x.shape) == 5:
            x = x[:, -1, :, :, :] 
            
        b, c, h, w = x.shape
        
        v_feat = self.visual_backbone(x).view(b, -1)
        v_feat = F.relu(self.visual_fc(v_feat))
        
        # Late Fusion: Concatenate visual features with temporal priors
        combined = torch.cat([v_feat, pred_phase_probs, phase_durations, pred_remaining], dim=1)
        return self.fusion_layer(combined)

class ToolBaselineModel(nn.Module):
    """
    Baseline Model: Pure visual-based tool identification for performance comparison.
    """
    def __init__(self, num_tools=7):
        super(ToolBaselineModel, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.visual_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, num_tools)

    def forward(self, x, *args):
        # *args used for interface compatibility with training scripts
        if len(x.shape) == 5:
            x = x[:, -1, :, :, :] 
            
        b, c, h, w = x.shape
        v_feat = self.visual_backbone(x).view(b, -1)
        return self.fc(v_feat)

class MultiTaskTimedToolModel(nn.Module):
    """
    Task B Model (Multi-Task): Tool identification with an auxiliary Presence Recognition task.
    """
    def __init__(self, num_tools=7, num_phases=7):
        super(MultiTaskTimedToolModel, self).__init__()
        # 1. Visual Feature Extraction
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.visual_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.visual_fc = nn.Linear(512, 128)
        
        # 2. Feature Fusion Layer
        # Total dim: 128 + num_phases + num_phases + 1
        combined_dim = 128 + num_phases + num_phases + 1 
        
        self.shared_fusion = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Head 1: Fine-grained Tool Identification (Multi-label)
        self.tool_head = nn.Linear(64, num_tools)
        
        # Head 2: Auxiliary Task - Tool Presence Prediction (Binary)
        # Determines if ANY tool is visible in the frame
        self.presence_head = nn.Linear(64, 1)

    def forward(self, x, pred_phase_probs, phase_durations, pred_remaining):
        if len(x.shape) == 5:
            x = x[:, -1, :, :, :]
            
        b = x.size(0)
        v_feat = self.visual_backbone(x).view(b, -1)
        v_feat = F.relu(self.visual_fc(v_feat))
        
        # Dimension alignment and concatenation
        phase_durations = phase_durations.view(b, -1)
        pred_remaining = pred_remaining.view(b, 1)
        combined = torch.cat([v_feat, pred_phase_probs, phase_durations, pred_remaining], dim=1)
        
        shared_feat = self.shared_fusion(combined)
        
        tool_logits = self.tool_head(shared_feat)
        presence_logits = self.presence_head(shared_feat)
        
        return tool_logits, presence_logits