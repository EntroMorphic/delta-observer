#!/usr/bin/env python3
"""
[LEGACY] Post-hoc Delta Observer

NOTE: This is the legacy post-hoc implementation that trains on frozen
activations from already-trained models. The primary implementation now
uses online observation during training.

For the current approach, see: models/delta_observer.py

This implementation achieves R²=0.9505, while the online observer achieves
R²=0.9879. The difference is due to temporal information captured during
online observation.

Original description:
Delta Observer: Learning the semantic primitive between monolithic and
compositional representations.

Architecture:
- Dual encoders (mono 64→32, comp 64→32)
- Shared latent space (64→16)
- Three decoders (mono reconstruction, comp reconstruction, bit position classification)

Loss:
- Reconstruction (both directions)
- Contrastive (same input → close, different → far)
- Classification (predict bit position)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import json
from datetime import datetime


class DeltaObserverDataset(Dataset):
    """Dataset for Delta Observer training."""
    
    def __init__(self, data_path):
        data = np.load(data_path)
        
        self.mono_act = torch.tensor(data['mono_activations'], dtype=torch.float32)
        self.comp_act = torch.tensor(data['comp_activations'], dtype=torch.float32)
        self.carry_counts = torch.tensor(data['carry_counts'], dtype=torch.long)
        self.bit_positions = torch.tensor(data['bit_positions'], dtype=torch.long)
        self.inputs = torch.tensor(data['inputs'], dtype=torch.float32)
    
    def __len__(self):
        return len(self.mono_act)
    
    def __getitem__(self, idx):
        return {
            'mono_act': self.mono_act[idx],
            'comp_act': self.comp_act[idx],
            'carry_count': self.carry_counts[idx],
            'bit_position': self.bit_positions[idx],
            'input': self.inputs[idx],
        }


class DeltaObserver(nn.Module):
    """
    Delta Observer network.
    
    Learns the semantic primitive that distinguishes monolithic and compositional
    representations of the same computation.
    """
    
    def __init__(self, mono_dim=64, comp_dim=64, latent_dim=16):
        super().__init__()
        
        # Dual encoders
        self.mono_encoder = nn.Sequential(
            nn.Linear(mono_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        self.comp_encoder = nn.Sequential(
            nn.Linear(comp_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Shared latent space encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(64, 32),  # Concatenated encodings
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, latent_dim),  # Semantic bottleneck
        )
        
        # Decoders
        self.mono_decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, mono_dim),
        )
        
        self.comp_decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, comp_dim),
        )
        
        # Bit position classifier
        self.bit_classifier = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),  # 4 bit positions
        )
        
        # Carry count regressor (for analysis)
        self.carry_regressor = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        
        self.latent_dim = latent_dim
    
    def encode(self, mono_act, comp_act):
        """Encode both activations to shared latent space."""
        mono_enc = self.mono_encoder(mono_act)
        comp_enc = self.comp_encoder(comp_act)
        
        # Concatenate and encode to latent
        joint = torch.cat([mono_enc, comp_enc], dim=-1)
        latent = self.shared_encoder(joint)
        
        return latent
    
    def decode(self, latent):
        """Decode latent to both representations."""
        mono_recon = self.mono_decoder(latent)
        comp_recon = self.comp_decoder(latent)
        return mono_recon, comp_recon
    
    def classify(self, latent):
        """Classify bit position from latent."""
        return self.bit_classifier(latent)
    
    def predict_carry(self, latent):
        """Predict carry count from latent (for analysis)."""
        return self.carry_regressor(latent)
    
    def forward(self, mono_act, comp_act):
        """Full forward pass."""
        # Encode
        latent = self.encode(mono_act, comp_act)
        
        # Decode
        mono_recon, comp_recon = self.decode(latent)
        
        # Classify
        bit_logits = self.classify(latent)
        
        # Predict carry
        carry_pred = self.predict_carry(latent)
        
        return {
            'latent': latent,
            'mono_recon': mono_recon,
            'comp_recon': comp_recon,
            'bit_logits': bit_logits,
            'carry_pred': carry_pred,
        }


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss: same input → close embeddings, different input → far embeddings.
    """
    
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, inputs):
        """
        embeddings: [batch, latent_dim]
        inputs: [batch, input_dim] - original inputs to check if same
        """
        batch_size = embeddings.size(0)
        
        # Compute pairwise similarity
        embeddings_norm = F.normalize(embeddings, dim=1)
        similarity = torch.matmul(embeddings_norm, embeddings_norm.t()) / self.temperature
        
        # Create labels: 1 if same input, 0 if different
        # For 4-bit adder, inputs are unique, so this is always 0 except diagonal
        # But we can use approximate similarity based on input distance
        input_dist = torch.cdist(inputs, inputs, p=2)
        labels = (input_dist < 1.0).float()  # Close inputs should have close embeddings
        
        # Contrastive loss: maximize similarity for similar inputs
        loss = F.binary_cross_entropy_with_logits(similarity, labels)
        
        return loss


def train_delta_observer(
    model,
    train_loader,
    val_loader,
    epochs=100,
    lr=0.001,
    device='cpu',
    save_path='models/delta_observer_best.pt',
):
    """Train the Delta Observer."""
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    contrastive_loss_fn = ContrastiveLoss(temperature=0.5)
    
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
    }
    
    print("="*80)
    print("DELTA OBSERVER TRAINING")
    print("="*80)
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Device: {device}")
    print("="*80)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            mono_act = batch['mono_act'].to(device)
            comp_act = batch['comp_act'].to(device)
            bit_position = batch['bit_position'].to(device)
            inputs = batch['input'].to(device)
            carry_count = batch['carry_count'].to(device).float()
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(mono_act, comp_act)
            
            # Reconstruction loss
            recon_loss = (
                F.mse_loss(outputs['mono_recon'], mono_act) +
                F.mse_loss(outputs['comp_recon'], comp_act)
            )
            
            # Contrastive loss
            contrast_loss = contrastive_loss_fn(outputs['latent'], inputs)
            
            # Classification loss
            class_loss = F.cross_entropy(outputs['bit_logits'], bit_position)
            
            # Carry prediction loss (auxiliary)
            carry_loss = F.mse_loss(outputs['carry_pred'].squeeze(), carry_count)
            
            # Combined loss
            loss = recon_loss + 0.5 * contrast_loss + class_loss + 0.1 * carry_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Accuracy
            pred = outputs['bit_logits'].argmax(dim=1)
            train_correct += (pred == bit_position).sum().item()
            train_total += bit_position.size(0)
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                mono_act = batch['mono_act'].to(device)
                comp_act = batch['comp_act'].to(device)
                bit_position = batch['bit_position'].to(device)
                inputs = batch['input'].to(device)
                carry_count = batch['carry_count'].to(device).float()
                
                outputs = model(mono_act, comp_act)
                
                recon_loss = (
                    F.mse_loss(outputs['mono_recon'], mono_act) +
                    F.mse_loss(outputs['comp_recon'], comp_act)
                )
                contrast_loss = contrastive_loss_fn(outputs['latent'], inputs)
                class_loss = F.cross_entropy(outputs['bit_logits'], bit_position)
                carry_loss = F.mse_loss(outputs['carry_pred'].squeeze(), carry_count)
                
                loss = recon_loss + 0.5 * contrast_loss + class_loss + 0.1 * carry_loss
                
                val_loss += loss.item()
                
                pred = outputs['bit_logits'].argmax(dim=1)
                val_correct += (pred == bit_position).sum().item()
                val_total += bit_position.size(0)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Update scheduler
        scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, save_path)
        
        # Log
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    print("="*80)
    print("TRAINING COMPLETE")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_path}")
    print("="*80)
    
    return history


if __name__ == "__main__":
    # Load dataset
    dataset = DeltaObserverDataset('/home/ubuntu/geometric-microscope/analysis/delta_observer_dataset.npz')
    
    # Split 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = DeltaObserver(mono_dim=64, comp_dim=64, latent_dim=16)
    
    # Train
    history = train_delta_observer(
        model,
        train_loader,
        val_loader,
        epochs=100,
        lr=0.001,
        device='cpu',
        save_path='/home/ubuntu/geometric-microscope/models/delta_observer_best.pt',
    )
    
    # Save history
    with open('/home/ubuntu/geometric-microscope/logs/delta_observer_training.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n✅ Training history saved to logs/delta_observer_training.json")
