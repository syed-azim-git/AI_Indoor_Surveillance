import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class UnimodalAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        attn_weights = F.softmax(self.attention(x), dim=1)  # (batch_size, seq_len, 1)
        attended = torch.bmm(attn_weights.transpose(1, 2), x)  # (batch_size, 1, input_dim)
        return attended.squeeze(1), attn_weights  # (batch_size, input_dim)

class ParallelCrossModalAttention(nn.Module):
    def __init__(self, audio_dim, visual_dim, hidden_dim):
        super().__init__()
        
        # Audio guided by visual
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.visual_proj_for_audio = nn.Linear(visual_dim, hidden_dim)
        self.joint_av = nn.Linear(hidden_dim, 1)
        
        # Visual guided by audio
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.audio_proj_for_visual = nn.Linear(audio_dim, hidden_dim)
        self.joint_va = nn.Linear(hidden_dim, 1)
        
    def forward(self, audio, visual):
        # audio: (batch_size, audio_dim)
        # visual: (batch_size, visual_dim)
        
        batch_size = audio.size(0)
        
        # Reshape for broadcasting
        audio_expanded = audio.unsqueeze(1)  # (batch_size, 1, audio_dim)
        visual_expanded = visual.unsqueeze(1)  # (batch_size, 1, visual_dim)
        
        # Audio guided by visual (C^a)
        audio_proj = self.audio_proj(audio_expanded)  # (batch_size, 1, hidden_dim)
        visual_proj_a = self.visual_proj_for_audio(visual)  # (batch_size, hidden_dim)
        visual_proj_a = visual_proj_a.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        joint_av = torch.tanh(audio_proj + visual_proj_a)  # (batch_size, 1, hidden_dim)
        attn_av = F.softmax(self.joint_av(joint_av), dim=1)  # (batch_size, 1, 1)
        context_a = torch.bmm(attn_av.transpose(1, 2), audio_expanded)  # (batch_size, 1, audio_dim)
        
        # Visual guided by audio (C^v)
        visual_proj = self.visual_proj(visual_expanded)  # (batch_size, 1, hidden_dim)
        audio_proj_v = self.audio_proj_for_visual(audio)  # (batch_size, hidden_dim)
        audio_proj_v = audio_proj_v.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        joint_va = torch.tanh(visual_proj + audio_proj_v)  # (batch_size, 1, hidden_dim)
        attn_va = F.softmax(self.joint_va(joint_va), dim=1)  # (batch_size, 1, 1)
        context_v = torch.bmm(attn_va.transpose(1, 2), visual_expanded)  # (batch_size, 1, visual_dim)
        
        return context_a.squeeze(1), context_v.squeeze(1)  # (batch_size, audio_dim), (batch_size, visual_dim)

class AudioVisualFusion(nn.Module):
    def __init__(self, audio_dim, visual_dim, output_dim):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(audio_dim + visual_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, audio, visual):
        # Concatenate along feature dimension
        combined = torch.cat([audio, visual], dim=1)
        return self.fusion(combined)

class AudioVisualFusionModel(nn.Module):
    def __init__(self, audio_dim=40, visual_dim=1024, hidden_dim=256, fusion_dim=512, num_categories=112):
        super().__init__()
        
        # Unimodal attention
        self.audio_attention = UnimodalAttention(audio_dim, hidden_dim)
        self.visual_attention = UnimodalAttention(visual_dim, hidden_dim)
        
        # Cross-modal attention
        self.cross_modal_attention = ParallelCrossModalAttention(audio_dim, visual_dim, hidden_dim)
        
        # Audio-visual fusion modules
        self.av_fusion = AudioVisualFusion(audio_dim, visual_dim, fusion_dim)
        self.va_fusion = AudioVisualFusion(visual_dim, audio_dim, fusion_dim)
        
        # Prediction head for event category
        self.category_prediction = nn.Sequential(
            nn.Linear(fusion_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_categories)
        )
        
    def forward(self, audio_seq, visual_seq):
        """
        Process entire sequences and divide into 5 segments
        
        Args:
            audio_seq: (batch_size, 20, audio_dim)
            visual_seq: (batch_size, 20, visual_dim)
            
        Returns:
            Tensor of shape (batch_size, 5, num_categories)
        """
        batch_size = audio_seq.size(0)
        num_segments = 5
        time_steps_per_segment = audio_seq.size(1) // num_segments  # Should be 4
        
        # Reshape to (batch_size, num_segments, time_steps_per_segment, feature_dim)
        audio_reshaped = audio_seq.view(batch_size, num_segments, time_steps_per_segment, -1)
        visual_reshaped = visual_seq.view(batch_size, num_segments, time_steps_per_segment, -1)
        
        segment_outputs = []
        
        for i in range(num_segments):
            # Process each segment
            audio_segment = audio_reshaped[:, i]  # (batch_size, time_steps_per_segment, audio_dim)
            visual_segment = visual_reshaped[:, i]  # (batch_size, time_steps_per_segment, visual_dim)
            
            # Unimodal attention
            audio_attended, _ = self.audio_attention(audio_segment)  # (batch_size, audio_dim)
            visual_attended, _ = self.visual_attention(visual_segment)  # (batch_size, visual_dim)
            
            # Cross-modal attention
            audio_context, visual_context = self.cross_modal_attention(audio_attended, visual_attended)
            
            # Audio-visual fusion
            o_va = self.av_fusion(visual_context, audio_context)  # (batch_size, fusion_dim)
            o_av = self.va_fusion(audio_context, visual_context)  # (batch_size, fusion_dim)
            
            # Combine fusion outputs
            combined = torch.cat([o_va, o_av], dim=1)  # (batch_size, fusion_dim*2)
            
            # Category prediction
            segment_logits = self.category_prediction(combined)  # (batch_size, num_categories)
            segment_outputs.append(segment_logits)
        
        # Stack along a new dimension
        return torch.stack(segment_outputs, dim=1)  # (batch_size, 5, num_categories)
    
    def predict(self, audio_seq, visual_seq):
        """
        Make predictions with softmax and return class indices
        
        Args:
            audio_seq: (batch_size, 20, audio_dim)
            visual_seq: (batch_size, 20, visual_dim)
            
        Returns:
            Class indices of shape (batch_size, 5)
        """
        logits = self.forward(audio_seq, visual_seq)  # (batch_size, 5, num_categories)
        probabilities = F.softmax(logits, dim=2)  # Apply softmax along the category dimension
        predicted_classes = torch.argmax(probabilities, dim=2)  # (batch_size, 5)
        return predicted_classes, probabilities
