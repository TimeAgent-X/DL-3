import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, rnn_size, context_size):
        super(Attention, self).__init__()
        self.w = nn.Linear(rnn_size * 2, context_size)
        self.context_vector = nn.Parameter(torch.randn(context_size))
        
    def forward(self, rnn_output):
        # rnn_output: [batch, seq_len, rnn_size * 2] (bidirectional)
        
        # u_it = tanh(W * h_it + b)
        u = torch.tanh(self.w(rnn_output)) # [batch, seq_len, context_size]
        
        # alpha_it = exp(u_it^T * u_w) / sum(...)
        # score = u_it . u_w
        scores = torch.matmul(u, self.context_vector) # [batch, seq_len]
        
        # softmax to get weights
        alpha = F.softmax(scores, dim=1).unsqueeze(-1) # [batch, seq_len, 1]
        
        # s = sum(alpha_it * h_it)
        s = torch.sum(rnn_output * alpha, dim=1) # [batch, rnn_size * 2]
        
        return s, alpha.squeeze(-1)

class WordAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, word_rnn_size, word_context_size, dropout=0.5, 
                 num_layers=1, use_layer_norm=False):
        super(WordAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, word_rnn_size, num_layers=num_layers, 
                          bidirectional=True, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = Attention(word_rnn_size, word_context_size)
        self.dropout = nn.Dropout(dropout)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.ln = nn.LayerNorm(word_rnn_size * 2)
        
    def forward(self, x):
        x = self.embedding(x) # [batch, word_len, embed_dim]
        x = self.dropout(x)
        
        h_output, _ = self.gru(x) # [batch, word_len, word_rnn_size*2]
        
        if self.use_layer_norm:
            h_output = self.ln(h_output)
            
        s_output, attention_weights = self.attention(h_output) # [batch, word_rnn_size*2]
        
        return s_output

class SentenceAttention(nn.Module):
    def __init__(self, word_rnn_size, sent_rnn_size, sent_context_size, dropout=0.5, 
                 num_layers=1, use_layer_norm=False):
        super(SentenceAttention, self).__init__()
        self.gru = nn.GRU(word_rnn_size * 2, sent_rnn_size, num_layers=num_layers,
                          bidirectional=True, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = Attention(sent_rnn_size, sent_context_size)
        self.dropout = nn.Dropout(dropout)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.ln = nn.LayerNorm(sent_rnn_size * 2)
        
    def forward(self, x):
        x = self.dropout(x)
        
        h_output, _ = self.gru(x)
        
        if self.use_layer_norm:
            h_output = self.ln(h_output)
            
        v_output, attention_weights = self.attention(h_output)
        
        return v_output

class HAN(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, word_rnn_size=50, word_context_size=100,
                 sent_rnn_size=50, sent_context_size=100, num_classes=5, dropout=0.5,
                 num_layers=1, use_layer_norm=False, use_residual=False):
        super(HAN, self).__init__()
        
        self.word_attention = WordAttention(vocab_size, embed_dim, word_rnn_size, word_context_size, dropout, num_layers, use_layer_norm)
        self.sentence_attention = SentenceAttention(word_rnn_size, sent_rnn_size, sent_context_size, dropout, num_layers, use_layer_norm)
        self.fc = nn.Linear(sent_rnn_size * 2, num_classes)
        self.use_residual = use_residual
        
    def forward(self, x):
        # x: [batch, sent_len, word_len]
        batch_size, sent_len, word_len = x.size()
        
        # Flatten to process all sentences at word level
        x = x.view(batch_size * sent_len, word_len) # [batch*sent_len, word_len]
        
        # Word Level
        x = self.word_attention(x) # [batch*sent_len, word_rnn_size*2]
        
        # Reshape for Sentence Level
        x = x.view(batch_size, sent_len, -1) # [batch, sent_len, word_rnn_size*2]
        
        # Sentence Level
        x = self.sentence_attention(x) # [batch, sent_rnn_size*2]
        
        # Classifier
        logits = self.fc(x)
        
        return logits
