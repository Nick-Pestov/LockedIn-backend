import pandas as pd
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.optim as optim

df = pd.read_csv("data.csv")

tokenizer = get_tokenizer('basic_english')
def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(df['text']), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

torch.save(vocab.get_stoi(), 'vocab.pth')

label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

torch.save(label_encoder.classes_, 'label_encoder.pth')

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        text_tensor = torch.tensor([self.vocab[token] for token in tokenizer(text)], dtype=torch.long)
        return text_tensor, label

dataset = TextDataset(df['text'].tolist(), df['label'].tolist(), vocab)

def collate_fn(batch):
    texts, labels = zip(*batch)
    texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return texts, labels

dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

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

vocab_size = len(vocab)
embed_size = 50
hidden_size = 64
num_classes = len(label_encoder.classes_)

model = LSTMModel(vocab_size, embed_size, hidden_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def calculate_accuracy(dataloader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in dataloader:
            outputs = model(texts)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

num_epochs = 4
for epoch in range(num_epochs):
    model.train()
    for texts, labels in dataloader:
        outputs = model(texts)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_accuracy = calculate_accuracy(dataloader, model)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Accuracy: {train_accuracy * 100:.2f}%')

torch.save(model.state_dict(), 'model.pth')
