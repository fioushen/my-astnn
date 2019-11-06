import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import time

from model import ASTNN

TRAINING_SET_SIZE = 1000

w2v = Word2Vec.load('./data/c/w2v_128').wv
embeddings = torch.tensor(np.vstack([w2v.vectors, [0] * 128]))

programs = pd.read_pickle('./data/c/id_code_label_ast_(index_tree).pkl')\
    .sample(n=TRAINING_SET_SIZE).reset_index(drop=True)

max_label_id = max(programs['label'])

print(max_label_id)

net = ASTNN(output_dim=max_label_id,
            embedding_dim=128, num_embeddings=len(w2v.vectors) + 1, embeddings=embeddings)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

for epoch in range(100):
    running_loss = 0.0
    start = time.time()
    right = 0
    wrong = 0
    for i in range(len(programs['index_tree'])):
        optimizer.zero_grad()

        output, label = net(programs['index_tree'][i]), torch.tensor([programs['label'][i] - 1])
        output = torch.softmax(output, dim=0)
        if output.argmax().item() == programs['label'][i] - 1:
            right += 1
        else:
            wrong += 1
        loss = criterion(output.unsqueeze(0), label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            print('Precision:', right / (wrong + right))
            print('Time cost(s):', time.time() - start)
            running_loss = 0.0
            start = time.time()
