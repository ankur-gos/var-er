import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, PickleType, String
from sqlalchemy.orm import sessionmaker


base = declarative_base()
engine = create_engine('sqlite:///:memory:', echo=True)
Session = sessionmaker(bind=engine)

class Embeddings(base):
    __tablename__ = 'embeddings'
    word = Column(String, primary_key=True)
    embedding = Column(PickleType)

    def __repr__(self):
        return f'<Embedding(word={self.word}, embedding={self.embedding})>'


class EmbeddingsDB():

    def __init__(self, session):
        self.session = session


    def load_glove(self, embeddings_file):
        with open(embeddings_file, 'r') as ef:
            for line in eff:
                line = line[:-1] # shave the new line
                spl = line.split(' ')
                word = spl[0]
                embedding = spl[1:]
                embedding = [float(x) for x in embedding]
                new_word = Embeddings(word=word, embedding=embedding)
                self.session.add(new_word)
            self.session.commit()

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


class WSJ(Dataset):

    def read_wsj(self, file_path):
        with open(file_path, 'r') as rf:
            current_example = None
            for line in rf:
                if line == '\n':
                    new_sequence, new_sequence_labels = zip(*current_example)
                    self.sequences.append(new_sequence)
                    self.sequence_labels(new_sequence_labels)
                    current_example = None
                    continue
                line = line[:-1] # shave the newline
                spl = line.split(' ')
                word, _, label = spl
                if label not in self.labels:
                    self.labels.append(label)
                label_ind = self.labels
                if current_example is None:
                    current_example = [(word, label_ind)]
                else:
                    current_example.append((word, label_ind))

    def __init__(self, file_path):
        edb = EmbeddingsDB(session)
        edb.load_glove('data/embeddings/glove.6B.50d.txt')
        self.sequences = []
        self.sequence_labels = []
        self.labels = []
        self.read_wsj(file_path)
        print(self.sequences)
        print(self.sequence_labels)


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
    sequence_dataset = WSJ('data/wsj/train.txt')
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

