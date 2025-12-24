import torch
import torch.nn as nn
import torch.optim as optim
import time
from utils import accuracy

class TrainRunner:
    def __init__(self, model, device, criterion, optimizer, scheduler=None):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_acc = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            
            self.optimizer.step()
            
            acc = accuracy(output, target)
            total_loss += loss.item()
            total_acc += acc
            
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {acc:.4f}")
                
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        return avg_loss, avg_acc

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        total_acc = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                acc = accuracy(output, target)
                total_loss += loss.item()
                total_acc += acc
                
        avg_loss = total_loss / len(val_loader)
        avg_acc = total_acc / len(val_loader)
        return avg_loss, avg_acc

    def run(self, train_loader, val_loader, epochs=10, save_path='best_model.pth'):
        best_acc = 0
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            start_time = time.time()
            
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            
            if self.scheduler:
                self.scheduler.step()
                
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Time: {time.time() - start_time:.2f}s")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                print("Saved new best model.")
                
        return history, best_acc
