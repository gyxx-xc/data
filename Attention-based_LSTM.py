import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F

class SuicideRiskDataset(Dataset):
    def __init__(self, data, labels, vocab, max_length):
        self.data = data
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]
        input_ids = [self.vocab.get(word, self.vocab['<UNK>']) for word in text.split()][:self.max_length]
        input_ids = torch.tensor(input_ids)
        label = torch.tensor(label)
        return input_ids, label

class AdditiveAttentionLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super(AdditiveAttentionLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        output = self.dropout(context)
        output = self.fc(output)
        return output

def evaluate_model(model, test_dataloader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    f1 = F.f1_score(torch.tensor(y_true), torch.tensor(y_pred), average='weighted')
    return f1.item()

# -------------------- program start here --------------------

# 使用这个Dataset类加载数据
# train_dataset = SuicideRiskDataset(train_data, train_labels, vocab, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AdditiveAttentionLSTM(len(vocab), 128, 64, 4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# test_dataset = SuicideRiskDataset(test_data, test_labels, vocab, max_length)
test_dataloader = DataLoader(test_dataset, batch_size=32)
f1_score = evaluate_model(model, test_dataloader)
print(f'Weighted F1 Score: {f1_score:.4f}')

