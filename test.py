import torch
import torch.nn as nn
from data_loader import get_loaders
from model import HAN
from train import TrainRunner
import os

def test_best_model():
    # 1. Configuration (Best Model: Dropout 0.2)
    config = {
        'data_dir': 'data',
        'batch_size': 64,
        'max_words': 30000,
        'max_len_word': 15,
        'max_len_sent': 30,
        'embed_dim': 100,
        'word_rnn_size': 50,
        'sent_rnn_size': 50,
        'word_context_size': 100,
        'sent_context_size': 100,
        'dropout': 0.2,            # Best Params
        'num_layers': 1,           # Base
        'use_layer_norm': False,   # Base
        'use_residual': False,     # Base
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'limit_samples': None,     # Test on full set
        'model_path': 'model_Dropout_0.2.pth' # Weights from user's result
    }

    print("=== Final Testing on Best Model (Dropout 0.2) ===")
    print(f"Device: {config['device']}")

    # 2. Data Loading
    print("Loading test data...")
    # We only need the test loader, but get_loaders returns all
    _, _, test_loader, tokenizer = get_loaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        max_words=config['max_words'],
        max_len_word=config['max_len_word'],
        max_len_sent=config['max_len_sent'],
        limit_samples=config['limit_samples']
    )
    
    vocab_size = len(tokenizer.word2idx) + 1
    print(f"Vocab size: {vocab_size}")

    # 3. Model Initialization
    model = HAN(
        vocab_size=vocab_size,
        embed_dim=config['embed_dim'],
        word_rnn_size=config['word_rnn_size'],
        word_context_size=config['word_context_size'],
        sent_rnn_size=config['sent_rnn_size'],
        sent_context_size=config['sent_context_size'],
        num_classes=5,
        dropout=config['dropout'],
        num_layers=config['num_layers'],
        use_layer_norm=config['use_layer_norm'],
        use_residual=config['use_residual']
    ).to(config['device'])

    # 4. Load Weights
    if os.path.exists(config['model_path']):
        print(f"Loading weights from {config['model_path']}...")
        model.load_state_dict(torch.load(config['model_path'], map_location=config['device']))
    else:
        print(f"Error: Model file '{config['model_path']}' not found!")
        print("Please ensure you have run the experiments and the file exists.")
        return

    # 5. Evaluation
    criterion = nn.CrossEntropyLoss()
    # Optimizer is not needed for testing but required by TrainRunner init
    optimizer = torch.optim.Adam(model.parameters()) 
    
    runner = TrainRunner(model, config['device'], criterion, optimizer)
    
    print("Evaluating...")
    test_loss, test_acc = runner.evaluate(test_loader)
    
    print("-" * 30)
    print(f"Final Test Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("-" * 30)

if __name__ == "__main__":
    test_best_model()
