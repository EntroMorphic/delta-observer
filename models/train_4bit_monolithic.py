#!/usr/bin/env python3
"""
Train monolithic 4-bit adder.

Goal: Achieve 100% accuracy on all 512 cases.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import json
from datetime import datetime
import math


# Fine structure constant
ALPHA = 1.0 / 137.0


class AdderDataset(Dataset):
    """4-bit adder dataset."""
    
    def __init__(self, inputs, outputs):
        self.inputs = inputs.float()
        self.outputs = outputs.float()
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


class Monolithic4BitAdder(nn.Module):
    """
    Monolithic 4-bit adder neural network.
    
    Simple task, so we don't need a huge network.
    """
    
    def __init__(self, hidden_size=64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(9, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 5),
        )
        
        self.hidden_size = hidden_size
    
    def forward(self, x, return_activations=False):
        if return_activations:
            activations = {}
            
            x = self.net[0](x)  # Linear
            activations['layer1_pre'] = x.detach().cpu().numpy()
            x = self.net[1](x)  # ReLU
            activations['layer1_post'] = x.detach().cpu().numpy()
            
            x = self.net[2](x)  # Linear
            activations['layer2_pre'] = x.detach().cpu().numpy()
            x = self.net[3](x)  # ReLU
            activations['layer2_post'] = x.detach().cpu().numpy()
            
            x = self.net[4](x)  # Linear
            activations['layer3_pre'] = x.detach().cpu().numpy()
            x = self.net[5](x)  # ReLU
            activations['layer3_post'] = x.detach().cpu().numpy()
            
            x = self.net[6](x)  # Output
            
            return x, activations
        
        return self.net(x)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, 
                                    peak_lr, min_lr):
    """Learning rate scheduler with warmup and cosine decay."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / peak_lr, cosine_decay)
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Accuracy: all 5 bits must match
        pred = torch.round(outputs).long()
        true = targets.long()
        correct += (pred == true).all(dim=1).sum().item()
        total += inputs.size(0)
    
    return total_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            
            pred = torch.round(outputs).long()
            true = targets.long()
            correct += (pred == true).all(dim=1).sum().item()
            total += inputs.size(0)
    
    return total_loss / len(dataloader), correct / total


def main():
    """Train monolithic 4-bit adder."""
    print("=" * 80)
    print("MONOLITHIC 4-BIT ADDER TRAINING")
    print("=" * 80)
    
    # Hyperparameters
    HIDDEN_SIZE = 64
    BATCH_SIZE = 32  # Small dataset, small batches
    PEAK_LR = ALPHA
    MIN_LR = ALPHA / 1000.0
    WARMUP_EPOCHS = 10
    EPOCHS = 200  # Give it plenty of time to converge
    TRAIN_SPLIT = 0.8
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    print(f"⚛️  Fine Structure Constant α = 1/137 ≈ {ALPHA:.10f}")
    
    # Load dataset
    print("\n📂 Loading dataset...")
    data = torch.load("/home/ubuntu/geometric-microscope/data/4bit_adder_dataset.pt")
    inputs = data['inputs']
    outputs = data['outputs']
    print(f"   Total samples: {len(inputs)}")
    
    dataset = AdderDataset(inputs, outputs)
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    print(f"\n🧠 Creating monolithic 4-bit adder (hidden_size={HIDDEN_SIZE})...")
    model = Monolithic4BitAdder(hidden_size=HIDDEN_SIZE).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Optimizer and scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=PEAK_LR)
    
    num_training_steps = len(train_loader) * EPOCHS
    num_warmup_steps = len(train_loader) * WARMUP_EPOCHS
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps, PEAK_LR, MIN_LR
    )
    
    print(f"\n📈 Learning rate schedule:")
    print(f"   Warmup: → {PEAK_LR:.6f} (α) over {WARMUP_EPOCHS} epochs")
    print(f"   Decay: {PEAK_LR:.6f} → {MIN_LR:.8f} (α/1000) over {EPOCHS-WARMUP_EPOCHS} epochs")
    
    # Training loop
    print(f"\n🏃 Training for {EPOCHS} epochs...")
    print("-" * 80)
    
    training_log = {
        'task': '4-bit adder',
        'model_type': 'monolithic',
        'hyperparameters': {
            'hidden_size': HIDDEN_SIZE,
            'batch_size': BATCH_SIZE,
            'peak_lr': float(PEAK_LR),
            'min_lr': float(MIN_LR),
            'warmup_epochs': WARMUP_EPOCHS,
            'epochs': EPOCHS,
            'train_split': TRAIN_SPLIT,
            'alpha': float(ALPHA),
        },
        'model_params': total_params,
        'device': str(device),
        'start_time': datetime.now().isoformat(),
        'epochs': []
    }
    
    best_val_acc = 0
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        current_lr = scheduler.get_last_lr()[0]
        
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'val_loss': float(val_loss),
            'val_acc': float(val_acc),
            'lr': float(current_lr),
        }
        training_log['epochs'].append(epoch_log)
        
        # Print every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"LR: {current_lr:.6f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, "/home/ubuntu/geometric-microscope/models/4bit_monolithic_best.pt")
        
        # Early stopping if perfect
        if val_acc >= 1.0:
            print(f"\n🎯 Perfect accuracy achieved at epoch {epoch+1}!")
            break
    
    training_log['end_time'] = datetime.now().isoformat()
    training_log['best_val_acc'] = float(best_val_acc)
    training_log['final_epoch'] = epoch + 1
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'val_acc': val_acc,
    }, "/home/ubuntu/geometric-microscope/models/4bit_monolithic_final.pt")
    
    with open("/home/ubuntu/geometric-microscope/logs/4bit_monolithic_training.json", 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print("-" * 80)
    print(f"\n✅ Training complete!")
    print(f"   Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"   Final validation accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"   Stopped at epoch: {epoch+1}/{EPOCHS}")
    print(f"\n💾 Models saved:")
    print(f"   - 4bit_monolithic_best.pt")
    print(f"   - 4bit_monolithic_final.pt")
    print(f"\n📊 Training log saved: logs/4bit_monolithic_training.json")


if __name__ == "__main__":
    main()
