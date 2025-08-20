"""
SmokeyNet-Lite Temporal Verifier

Lightweight implementation based on SmokeyNet for temporal verification of smoke detections.
Uses EfficientNet-B0 backbone + LSTM for processing ROI sequences.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import torchvision.models as models
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("Warning: Torchvision not available. Using simplified backbone.")


class EfficientNetBackbone(nn.Module):
    """EfficientNet-B0 backbone for feature extraction"""
    
    def __init__(self, pretrained: bool = True, freeze_early_layers: bool = True):
        super().__init__()
        
        if TORCHVISION_AVAILABLE:
            if pretrained:
                weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            else:
                weights = None
            
            # Load EfficientNet-B0
            self.backbone = efficientnet_b0(weights=weights)
            
            # Remove classifier
            self.backbone.classifier = nn.Identity()
            
            # Get feature dimension
            self.feature_dim = 1280  # EfficientNet-B0 output dim
            
            # Optionally freeze early layers
            if freeze_early_layers:
                self._freeze_early_layers()
        else:
            # Simplified backbone when torchvision not available
            self.backbone = self._create_simple_backbone()
            self.feature_dim = 512
    
    def _freeze_early_layers(self, freeze_ratio: float = 0.5):
        """Freeze early layers for transfer learning"""
        total_layers = len(list(self.backbone.features.children()))
        freeze_until = int(total_layers * freeze_ratio)
        
        for i, child in enumerate(self.backbone.features.children()):
            if i < freeze_until:
                for param in child.parameters():
                    param.requires_grad = False
    
    def _create_simple_backbone(self):
        """Create simplified backbone when EfficientNet not available"""
        return nn.Sequential(
            # Initial convolution
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Feature extraction blocks
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input image
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Feature tensor (B, feature_dim)
        """
        return self.backbone(x)


class TemporalLSTM(nn.Module):
    """LSTM module for temporal sequence processing"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output dimension
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process temporal sequence
        
        Args:
            x: Input sequence (B, T, feature_dim)
            
        Returns:
            LSTM output (B, T, output_dim)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        return lstm_out


class SmokeyNetLite(nn.Module):
    """
    SmokeyNet-Lite: Lightweight temporal verifier for smoke detection
    
    Architecture:
    - EfficientNet-B0 backbone for spatial features
    - LSTM for temporal modeling
    - Binary classification head (smoke/no-smoke)
    """
    
    def __init__(
        self,
        sequence_length: int = 3,
        input_size: Tuple[int, int] = (224, 224),
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 2,
        dropout: float = 0.3,
        pretrained_backbone: bool = True
    ):
        super().__init__()
        
        self.sequence_length = sequence_length
        self.input_size = input_size
        
        # Backbone for feature extraction
        self.backbone = EfficientNetBackbone(pretrained=pretrained_backbone)
        
        # Temporal LSTM
        self.temporal_lstm = TemporalLSTM(
            input_dim=self.backbone.feature_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            bidirectional=True,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.temporal_lstm.output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)  # Binary classification
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for module in [self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for sequence of ROI crops
        
        Args:
            x: Input sequence (B, T, C, H, W) where T is sequence length
            
        Returns:
            Dictionary with predictions and features
        """
        batch_size, seq_len, channels, height, width = x.shape
        
        # Reshape for backbone processing: (B*T, C, H, W)
        x_reshaped = x.view(batch_size * seq_len, channels, height, width)
        
        # Extract spatial features
        spatial_features = self.backbone(x_reshaped)  # (B*T, feature_dim)
        
        # Reshape back to sequence: (B, T, feature_dim)
        feature_dim = spatial_features.shape[-1]
        temporal_features = spatial_features.view(batch_size, seq_len, feature_dim)
        
        # Process temporal sequence
        lstm_output = self.temporal_lstm(temporal_features)  # (B, T, lstm_output_dim)
        
        # Use last timestep for classification
        last_output = lstm_output[:, -1, :]  # (B, lstm_output_dim)
        
        # Classification
        logits = self.classifier(last_output)  # (B, 2)
        
        return {
            'logits': logits,
            'probabilities': torch.softmax(logits, dim=-1),
            'temporal_features': temporal_features,
            'lstm_output': lstm_output
        }
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Prediction with post-processing
        
        Args:
            x: Input sequence (B, T, C, H, W)
            threshold: Classification threshold
            
        Returns:
            Prediction results
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            
            probabilities = output['probabilities']
            smoke_prob = probabilities[:, 1]  # Probability of smoke class
            predictions = (smoke_prob > threshold).long()
            
            return {
                'predictions': predictions,
                'smoke_probability': smoke_prob,
                'confidence': torch.max(probabilities, dim=-1)[0],
                'logits': output['logits']
            }
    
    def verify_temporal_consistency(
        self,
        roi_sequences: List[torch.Tensor],
        threshold: float = 0.5,
        min_frames: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Verify temporal consistency for multiple ROI sequences
        
        Args:
            roi_sequences: List of ROI sequences for different detections
            threshold: Classification threshold
            min_frames: Minimum frames required for positive prediction
            
        Returns:
            List of verification results
        """
        results = []
        
        for roi_sequence in roi_sequences:
            if len(roi_sequence) < min_frames:
                # Not enough frames for verification
                results.append({
                    'verified': False,
                    'reason': 'insufficient_frames',
                    'smoke_probability': 0.0,
                    'frame_predictions': []
                })
                continue
            
            # Pad sequence if needed
            if roi_sequence.shape[0] < self.sequence_length:
                # Repeat last frame to reach required length
                padding_needed = self.sequence_length - roi_sequence.shape[0]
                last_frame = roi_sequence[-1:].repeat(padding_needed, 1, 1, 1)
                roi_sequence = torch.cat([roi_sequence, last_frame], dim=0)
            
            # Add batch dimension
            sequence_batch = roi_sequence.unsqueeze(0)  # (1, T, C, H, W)
            
            # Get prediction
            prediction_result = self.predict(sequence_batch, threshold)
            
            # Extract results
            is_smoke = bool(prediction_result['predictions'][0])
            smoke_prob = float(prediction_result['smoke_probability'][0])
            confidence = float(prediction_result['confidence'][0])
            
            results.append({
                'verified': is_smoke,
                'smoke_probability': smoke_prob,
                'confidence': confidence,
                'sequence_length': roi_sequence.shape[0],
                'reason': 'temporal_verification'
            })
        
        return results
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Extract intermediate feature maps for visualization"""
        # This would be used for Grad-CAM or other interpretability methods
        return self.backbone(x)


def create_smokeynet_lite(
    sequence_length: int = 3,
    input_size: Tuple[int, int] = (224, 224),
    pretrained: bool = True
) -> SmokeyNetLite:
    """
    Factory function to create SmokeyNet-Lite model
    
    Args:
        sequence_length: Number of frames in sequence
        input_size: Input image size (H, W)
        pretrained: Use pretrained backbone
        
    Returns:
        Configured SmokeyNet-Lite model
    """
    return SmokeyNetLite(
        sequence_length=sequence_length,
        input_size=input_size,
        pretrained_backbone=pretrained
    )


if __name__ == "__main__":
    # Test the model
    model = create_smokeynet_lite()
    
    # Test with dummy sequence
    batch_size = 2
    sequence_length = 3
    dummy_input = torch.randn(batch_size, sequence_length, 3, 224, 224)
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Model created successfully")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Probabilities shape: {output['probabilities'].shape}")
    
    # Test prediction
    prediction = model.predict(dummy_input)
    print(f"Predictions: {prediction['predictions']}")
    print(f"Smoke probabilities: {prediction['smoke_probability']}")