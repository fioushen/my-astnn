import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import time

from model_compare import ASTNN

TRAINING_SET_SIZE = 30000
VALIDATION_SET_SIZE = 10000
TEST_SET_SIZE = 10000

print('Reading data...')

w2v = Word2Vec.load('./data/c/w2v_128').wv
embeddings = torch.tensor(np.vstack([w2v.vectors, [0] * 128]))

programs = pd.read_pickle('./data/c/id_code_label_ast_(index_tree).pkl')

training_set = programs[:TRAINING_SET_SIZE]
validation_set = programs[TRAINING_SET_SIZE:TRAINING_SET_SIZE + VALIDATION_SET_SIZE]
test_set = programs[TRAINING_SET_SIZE + VALIDATION_SET_SIZE:TRAINING_SET_SIZE + VALIDATION_SET_SIZE + TEST_SET_SIZE]


def get_batch(dataset, i, batch_size):
    return dataset.iloc[i: i + batch_size]


MAX_LABEL = max(programs['label'])

print('Max label:', MAX_LABEL)

BATCH_SIZE = 64
EPOCH = 15
net = ASTNN(output_dim=MAX_LABEL,
            embedding_dim=128, num_embeddings=len(w2v.vectors) + 1, embeddings=embeddings,
            batch_size=BATCH_SIZE)
net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(net.parameters())


def train(dataset, backward=True):
    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    i = 0

    while i < len(dataset):
        data = get_batch(dataset, i, BATCH_SIZE)
        input, label = data['index_tree'], torch.tensor([label - 1 for label in data['label']]).cuda()
        i += BATCH_SIZE

        net.zero_grad()
        net.batch_size = len(input)
        output = net(input)

        loss = criterion(output, label)

        if backward:
            loss.backward()
            optimizer.step()

        # calc acc
        pred = output.data.argmax(1)
        correct = pred.eq(label).sum().item()
        total_acc += correct
        total += len(input)
        total_loss += loss.item() * len(input)

    return total_loss / total, total_acc / total


print('Start Training...')
for epoch in range(EPOCH):
    start_time = time.time()

    training_loss, training_acc = train(training_set)
    validation_loss, validation_acc = train(validation_set, backward=False)

    end_time = time.time()
    print('[Epoch: %2d/%2d] Train Loss: %.4f, Train Acc: %.3f, Val Loss: %.4f, Val Acc: %.3f, Time Cost: %.3f s'
          % (epoch + 1, EPOCH,
             training_loss, training_acc, validation_loss, validation_acc,
             end_time - start_time))

test_loss, test_acc = train(test_set, backward=False)
print('Test Acc: %.3f' % test_acc)

