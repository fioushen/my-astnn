import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import time

from model_compare import ASTNN

TRAINING_SET_SIZE = 1000
VALIDATION_SET_SIZE = 0
TEST_SET_SIZE = 0

w2v = Word2Vec.load('./data/c/w2v_128').wv
embeddings = torch.tensor(np.vstack([w2v.vectors, [0] * 128]))

programs = pd.read_pickle('./data/c/id_code_label_ast_(index_tree).pkl')

training_set = programs[:TRAINING_SET_SIZE]
validation_set = programs[TRAINING_SET_SIZE:TRAINING_SET_SIZE + VALIDATION_SET_SIZE]
test_set = programs[TRAINING_SET_SIZE + VALIDATION_SET_SIZE:-1]

max_label_id = max(programs['label'])

print(max_label_id)

BATCH_SIZE = 64
net = ASTNN(output_dim=max_label_id,
            embedding_dim=128, num_embeddings=len(w2v.vectors) + 1, embeddings=embeddings,
            batch_size=BATCH_SIZE)
net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(net.parameters())


batches = []
losses = []

t = time.time()

for batch_count in range(300):
    start = time.time()

    data = training_set.sample(n=BATCH_SIZE)
    input, label = data['index_tree'], torch.tensor([label - 1 for label in data['label']]).cuda()

    output = net(input)

    loss = criterion(output, label)
    loss.backward()
    optimizer.step()

    pred = output.argmax(1)
    correct = pred.eq(label).sum().item()
    print('BATCH', batch_count)
    print('ACC', correct / BATCH_SIZE)
    print('loss:', loss.item())
    print('Time cost(s):', time.time() - start)
    batches.append(batch_count)
    losses.append(loss.item())

plt.plot(batches, losses)
plt.show()

data = test_set
net.batch_size = len(data['id'])
output = net(data['index_tree'])
label = torch.tensor([label - 1 for label in data['label']])
pred = output.argmax(1)
correct = pred.eq(label).sum().item()
print('FINAL ACC', correct / len(data['id']))
