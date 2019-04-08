import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class RandomSequenceDataset(Dataset):

    def generate_sequence(self):
        x = torch.rand(10)
        y = x.ge(0.5).float()
        return x, y

    def __init__(self, num_sequences):
        generate = [self.generate_sequence() for _ in range(num_sequences)]
        self.sequences, self.sequence_labels = zip(*generate)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sample = {'sequence': self.sequences[idx], 'labels': self.sequence_labels[idx]}
        return sample


class SequenceClassifier(torch.nn.Module):
    def __init__(self, sequence_len, hidden_size):
        super(SequenceClassifier, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=2, bidirectional=True)
        self.linear1 = torch.nn.Linear(hidden_size * 2, 1)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        lin_output = self.linear1(output)
        preds = self.sig(lin_output)
        return preds

if __name__ == '__main__':
    sequence_dataset = RandomSequenceDataset(10000)
    loader = DataLoader(sequence_dataset)
    sequence_len = 10
    hidden_size = 20
    loss = torch.nn.BCELoss()
    rnn = SequenceClassifier(sequence_len, hidden_size)
    optimizer = torch.optim.Adam(rnn.parameters(), lr = 0.0001)
    for example in loader:
        sample, labels = example['sequence'], example['labels']
        sample = torch.reshape(sample, (sequence_len, 1, 1))
        labels = torch.reshape(labels, (sequence_len, 1))
        output = rnn(sample).squeeze(1)
        l = loss(output, labels)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        print(l)

