import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_loaders
from model import HAN
from train import TrainRunner
from utils import plot_curves
import os
import itertools

def get_args():
    parser = argparse.ArgumentParser(description='HAN for text classification')
    
    # Data params
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_words', type=int, default=30000)
    parser.add_argument('--max_len_word', type=int, default=15)
    parser.add_argument('--max_len_sent', type=int, default=30)
    
    # Model params
    parser.add_argument('--embed_dim', type=int, default=100)
    parser.add_argument('--word_rnn_size', type=int, default=50)
    parser.add_argument('--sent_rnn_size', type=int, default=50)
    parser.add_argument('--word_context_size', type=int, default=100)
    parser.add_argument('--sent_context_size', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--use_layer_norm', action='store_true')
    parser.add_argument('--use_residual', action='store_true')
    
    # Training params
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay', action='store_true', help='Use StepLR learning rate decay')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'tune'])
    parser.add_argument('--save_path', type=str, default='best_model.pth')
    parser.add_argument('--plot_path', type=str, default='loss_curve.png')
    parser.add_argument('--limit_samples', type=int, default=None, help='Limit number of samples for debugging')
    
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_experiment(args):
    print(f"Running experiment with args: {args}")
    set_seed(args.seed)
    
    # Load Data
    train_loader, val_loader, test_loader, tokenizer = get_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_words=args.max_words,
        max_len_word=args.max_len_word,
        max_len_sent=args.max_len_sent,
        limit_samples=args.limit_samples
    )
    
    vocab_size = len(tokenizer.word2idx) + 1 # +1 for safety
    print(f"Vocab size: {vocab_size}")
    
    # Build Model
    model = HAN(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        word_rnn_size=args.word_rnn_size,
        word_context_size=args.word_context_size,
        sent_rnn_size=args.sent_rnn_size,
        sent_context_size=args.sent_context_size,
        num_classes=5, # Yelp 1-5 stars
        dropout=args.dropout,
        num_layers=args.num_layers,
        use_layer_norm=args.use_layer_norm,
        use_residual=args.use_residual
    ).to(args.device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    scheduler = None
    if args.lr_decay:
        # Decay LR by 0.5 every 3 epochs
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    runner = TrainRunner(model, args.device, criterion, optimizer, scheduler=scheduler)
    
    if args.mode == 'train' or args.mode == 'tune':
        history, best_acc = runner.run(train_loader, val_loader, epochs=args.epochs, save_path=args.save_path)
        
        # Plot
        if history and len(history['train_loss']) > 0:
            plot_curves(history['train_loss'], history['val_loss'], 
                        history['train_acc'], history['val_acc'], save_path=args.plot_path)
            
        if args.mode == 'train':
            # Test on Test Set only for train mode (tune mode just returns validation acc)
            print("Loading best model for testing...")
            model.load_state_dict(torch.load(args.save_path))
            test_loss, test_acc = runner.evaluate(test_loader)
            print(f"Test Accuracy: {test_acc:.4f}")
            
        return best_acc, history

    elif args.mode == 'test':
        if not os.path.exists(args.save_path):
            print(f"Model file {args.save_path} not found.")
            return
        model.load_state_dict(torch.load(args.save_path))
        test_loss, test_acc = runner.evaluate(test_loader)
        print(f"Test Accuracy: {test_acc:.4f}")
        return test_acc

def run_tuning(args):
    # Requirements: Dropout, Normalization, LR Decay, Residual, Depth
    # Base configuration to compare against
    base_params = {
        'dropout': 0.5, 'use_layer_norm': False, 'lr_decay': False, 
        'use_residual': False, 'num_layers': 1
    }
    
    # Variations to test individually (Ablation/Contrast)
    # 1. Dropout Impact
    # 2. Norm Impact
    # 3. Decay Impact
    # 4. Residual Impact (requires depth > 1 usually to be useful, but checking logic)
    # 5. Depth Impact
    
    experiments_config = [
        # Base
        {'name': 'Base', 'params': {}},
        
        # Dropout
        {'name': 'Dropout_0.2', 'params': {'dropout': 0.2}},
        {'name': 'Dropout_0.8', 'params': {'dropout': 0.8}},
        
        # Normalization
        {'name': 'With_LayerNorm', 'params': {'use_layer_norm': True}},
        
        # LR Decay
        {'name': 'With_LR_Decay', 'params': {'lr_decay': True}},
        
        # Residual & Depth
        # Residual implies we might want deeper networks
        {'name': 'Depth_2', 'params': {'num_layers': 2}},
        {'name': 'Depth_2_Residual', 'params': {'num_layers': 2, 'use_residual': True}},
        {'name': 'Depth_2_Residual_Norm', 'params': {'num_layers': 2, 'use_residual': True, 'use_layer_norm': True}},
    ]
    
    results = []
    
    print(f"Starting detailed tuning with {len(experiments_config)} configurations...")
    
    for i, exp in enumerate(experiments_config):
        print(f"\n--- Experiment {i+1}/{len(experiments_config)}: {exp['name']} ---")
        
        # Start with base args
        current_args = argparse.Namespace(**vars(args))
        
        # Reset to base params first
        for k, v in base_params.items():
            setattr(current_args, k, v)
            
        # Apply specific changes
        for k, v in exp['params'].items():
            setattr(current_args, k, v)
            
        current_args.save_path = f"model_{exp['name']}.pth"
        current_args.plot_path = f"curve_{exp['name']}.png"
        
        print(f"Config: {vars(current_args)}")
        
        best_acc, history = run_experiment(current_args)
        results.append({'name': exp['name'], 'params': exp['params'], 'acc': best_acc, 'history': history})
        
    print("\n=== Tuning Results ===")
    print(f"{'Experiment':<25} | {'Acc':<10}")
    print("-" * 40)
    for res in results:
        print(f"{res['name']:<25} | {res['acc']:.4f}")
        
    best_exp = max(results, key=lambda x: x['acc'])
    print(f"\nBest Experiment: {best_exp['name']} with Acc: {best_exp['acc']:.4f}")
    
    # Save best curve as the main loss_curve.png for README
    import shutil
    best_curve_path = f"curve_{best_exp['name']}.png"
    if os.path.exists(best_curve_path):
        shutil.copy(best_curve_path, args.plot_path)
        print(f"Copied best curve ({best_curve_path}) to {args.plot_path}")
        
    # Plot Comparison
    from utils import plot_comparison
    plot_comparison(results, save_path='comparison_curve.png')

if __name__ == '__main__':
    args = get_args()
    
    if args.mode == 'tune':
        run_tuning(args)
    else:
        run_experiment(args)
