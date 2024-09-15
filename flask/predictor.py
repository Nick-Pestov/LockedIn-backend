import torch
import pandas as pd
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import numpy as np
from collections import Counter

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        h, (hn, cn) = self.lstm(x)
        out = self.fc(h[:, -1, :])
        return out

def load_model(vocab_size, embed_size, hidden_size, num_classes, model_path='immersive-ml/model.pth'):
    model = LSTMModel(vocab_size, embed_size, hidden_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_vocab_and_label_encoder():
    stoi = torch.load('immersive-ml/vocab.pth')
    vocab = torchtext.vocab.vocab(stoi, specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])
    
    label_encoder_classes = torch.load('immersive-ml/label_encoder.pth')
    label_encoder = LabelEncoder()
    label_encoder.classes_ = label_encoder_classes
    
    return vocab, label_encoder

vocab, label_encoder = load_vocab_and_label_encoder()

vocab_size = len(vocab)
embed_size = 50
hidden_size = 64
num_classes = len(label_encoder.classes_)

model = load_model(vocab_size, embed_size, hidden_size, num_classes)

def preprocess_text(text, tokenizer, vocab):
    tokens = tokenizer(text)
    text_tensor = torch.tensor([vocab[token] for token in tokens], dtype=torch.long).unsqueeze(0)  # Add batch dimension
    return text_tensor

def predict_long_text(text, chunk_size=250):
    tokenizer = get_tokenizer('basic_english')

    # Split the text into chunks
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    # Make predictions for each chunk
    predictions = []
    for chunk in chunks:
        text_tensor = preprocess_text(chunk, tokenizer, vocab)
        with torch.no_grad():
            output = model(text_tensor)
            _, predicted = torch.max(output, 1)
            predictions.append(predicted.item())
    
    # Voting scheme
    count = Counter(predictions)
    most_common = count.most_common()
    
    # Handle tie cases
    max_count = most_common[0][1]
    most_common_labels = [label for label, cnt in most_common if cnt == max_count]
    final_prediction = np.random.choice(most_common_labels)
    
    predicted_label = label_encoder.inverse_transform([final_prediction])[0]
    return predicted_label

def predict_text(text):
    tokenizer = get_tokenizer('basic_english')
    text_tensor = preprocess_text(text, tokenizer, vocab)
    with torch.no_grad():
        output = model(text_tensor)
        _, predicted = torch.max(output, 1)
    predicted_label = label_encoder.inverse_transform([predicted.item()])[0]
    return predicted_label
