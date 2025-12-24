import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import re
from collections import Counter
import os

class Tokenizer:
    def __init__(self, max_words=30000, max_len_word=15, max_len_sent=20):
        self.max_words = max_words
        self.max_len_word = max_len_word
        self.max_len_sent = max_len_sent
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.count = 2

    def fit(self, texts):
        word_counts = Counter()
        for text in texts:
            words = self._tokenize(text)
            word_counts.update(words)
        
        most_common = word_counts.most_common(self.max_words - 2)
        for word, _ in most_common:
            self.word2idx[word] = self.count
            self.idx2word[self.count] = word
            self.count += 1
            
    def _tokenize(self, text):
        # Basic tokenization
        text = text.lower()
        text = re.sub(r"[^a-z0-9]", " ", text)
        return text.split()

    def transform(self, texts):
        data = []
        for text in texts:
            # Split into sentences (simple approach: split by punctuation before cleaning)
            # For HAN, we need structure: [sentences, words]
            # But the dataset might be just raw text.
            # Lets try to split sentences by .?!
            raw_sentences = re.split(r'[\.\?\!]', text.lower())
            
            doc_tensor = torch.zeros((self.max_len_sent, self.max_len_word), dtype=torch.long)
            
            sent_idx = 0
            for raw_sent in raw_sentences:
                if not raw_sent.strip():
                    continue
                if sent_idx >= self.max_len_sent:
                    break
                
                words = re.sub(r"[^a-z0-9]", " ", raw_sent).split()
                
                word_idx = 0
                for w in words:
                    if word_idx >= self.max_len_word:
                        break
                    doc_tensor[sent_idx, word_idx] = self.word2idx.get(w, 1) # 1 is UNK
                    word_idx += 1
                
                sent_idx += 1
            data.append(doc_tensor)
        return torch.stack(data)

class YelpDataset(Dataset):
    def __init__(self, data_tensor, labels):
        self.data = data_tensor
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def load_raw_data(file_path, is_test=False, limit=None):
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            item = json.loads(line)
            texts.append(item['text'])
            if not is_test:
                labels.append(int(item['stars']) - 1) # 1-5 -> 0-4
            else:
                # Test set usually has labels too in this dataset challenge, 
                # but if not we might need to handle it.
                # Assuming test.json has stars too.
                labels.append(int(item['stars']) - 1)
    return texts, labels

def get_loaders(data_dir='data', batch_size=64, max_words=30000, 
                max_len_word=15, max_len_sent=20, val_split=0.1, limit_samples=None):
    
    print("Loading raw data...")
    train_file = os.path.join(data_dir, 'yelp_academic_dataset_review.json')
    test_file = os.path.join(data_dir, 'test.json')
    
    train_texts, train_labels = load_raw_data(train_file, limit=limit_samples)
    test_texts, test_labels = load_raw_data(test_file, is_test=True, limit=limit_samples)

    
    print(f"Loaded {len(train_texts)} training samples and {len(test_texts)} test samples.")
    
    # Tokenizer
    print("Building vocabulary...")
    tokenizer = Tokenizer(max_words, max_len_word, max_len_sent)
    tokenizer.fit(train_texts)
    
    print("Transforming data...")
    train_tensor = tokenizer.transform(train_texts)
    test_tensor = tokenizer.transform(test_texts)
    
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    
    # Dataset
    full_train_dataset = YelpDataset(train_tensor, train_labels)
    test_dataset = YelpDataset(test_tensor, test_labels)
    
    # Split Train/Val
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, tokenizer

if __name__ == '__main__':
    # Test the loader
    train_loader, val_loader, test_loader, tokenizer = get_loaders(batch_size=32)
    print("Data Loaders ready.")
    for batch in train_loader:
        print("Batch shape:", batch[0].shape)
        print("Label shape:", batch[1].shape)
        break
