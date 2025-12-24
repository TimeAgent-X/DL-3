import matplotlib.pyplot as plt
import torch
import os

def plot_curves(train_losses, val_losses, train_accs, val_accs, save_path='loss_curve.png'):
    plt.figure(figsize=(12, 5))
    
    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy', marker='o')
    plt.plot(val_accs, label='Validation Accuracy', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Curves saved to {save_path}")
    plt.close()

def plot_comparison(results, save_path='comparison_curve.png'):
    # results: list of dicts with keys 'name', 'history'
    plt.figure(figsize=(10, 6))
    
    for res in results:
        name = res['name']
        val_accs = res['history']['val_acc']
        plt.plot(val_accs, label=f"{name} (Best: {max(val_accs):.4f})", marker='o')
        
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.title('Hyperparameter Comparison: Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Comparison curves saved to {save_path}")
    plt.close()

def accuracy(output, target):
    # output: [batch_size, num_classes]
    # target: [batch_size]
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        correct = (pred == target).sum().item()
        return correct / target.size(0)
