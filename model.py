import torch
import torch.nn as nn
from time import time


class TreeEncoder(nn.Module):
    r""":Transform a statement tree to a vector.

    Input: A tree represented like [1, [4, [2, 3]]] where 1, 4, 2, 3 are indices of embeddings
    Output: A vector represented of the given tree

    Equation: output = max(h), h = Wv + Σ(hi) + b
    v is embedding of root, h is hidden state of root, hi are hidden states of children,
    W and b are weights and biases

    Args:
        num_embeddings: total number of embeddings
        embedding_dim: dimension of embedding
        encode_dim: dimension of outputted vector
        embeddings: a Tensor of [num_embeddings x embedding_dim]
    """

    def __init__(self, num_embeddings, embedding_dim, encode_dim, embeddings):
        super(TreeEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.encode_dim = encode_dim
        self.node_list = []
        self.linear = nn.Linear(embedding_dim, encode_dim)
        self.embedding.weight.data.copy_(embeddings)

    def hidden_state(self, node):
        # result = torch.zeros(self.encode_dim, requires_grad=True)
        result = torch.zeros(self.encode_dim).cuda()
        header = node[0]
        for node in node[1:]:
            result = result.add(self.hidden_state(node))
        # header = self.embedding(torch.tensor(header, dtype=torch.long))
        header = self.embedding(torch.tensor(header, dtype=torch.long).cuda())
        result = result.add(self.linear(header))
        self.node_list.append(result)
        return result

    def forward(self, st_tree):
        # here x is like [1, [2, 3]] where 1, 2, 3 are embeddings index
        self.node_list.clear()
        self.hidden_state(st_tree)
        pool = torch.cat(self.node_list).view([-1, self.encode_dim])
        return torch.max(pool, 0).values


class ASTNN(nn.Module):
    r"""Transform a sequence of statement trees to a vector.

    Data flow: st-tree sequence -> [TreeEncoder (encode layer)] -> vector sequence
    -> [bi-GRU (recurrent layer)] -> output of bi-GRU
    -> [max pooling (pooling layer)] -> representation (vector of given tree sequence)

    Input: A list of tree where tree are like [1, [4, [2, 3]]], here 1, 4, 2, 3 are indices of embeddings
    Output: A vector represented of the given tree sequence

    Args:
        output_dim: dimension of outputted vector
        embedding_dim: dimension of embedding
        embeddings: a Tensor of [num_embeddings * embedding_dim]
        num_embeddings: total number of embeddings
    """

    def __init__(self, output_dim, num_embeddings, embedding_dim, embeddings, batch_size, hidden_dim=100, encode_dim=128):
        super(ASTNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.encode_dim = encode_dim
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

        self.encoder = TreeEncoder(num_embeddings, embedding_dim, encode_dim, embeddings)
        self.biGRU = nn.GRU(input_size=encode_dim, hidden_size=hidden_dim, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.FC = nn.Linear(hidden_dim * 2, output_dim)

    def forward_single(self, tree_seq):
        statement_vectors = [self.encoder(st_tree) for st_tree in tree_seq]

        # [1 * seq_len * input_size(encode_dim)]
        statement_vectors = torch.cat(statement_vectors).view([-1, self.encode_dim]).unsqueeze(0)
        # [1 * seq_len * 2 x hidden_dim]
        gru_output = self.biGRU(statement_vectors, torch.zeros([2, self.batch_size, self.hidden_dim]))[0]
        # [1 * 2 x hidden_dim]
        max_pool = gru_output.max(1).values
        # [1 * output_dim]
        return self.FC(max_pool).view([-1])

    def forward(self, x):
        print('*** Forwarding ***')
        t = time()

        lens = [len(item) for item in x]
        max_len = max(lens)

        encodes = torch.stack([self.encoder(tree) for tree_seq in x for tree in tree_seq])

        print('%s\t%2.2f s' % ('Encode', time() - t))
        t = time()

        seq, start, end = [], 0, 0
        for i in range(self.batch_size):
            end += lens[i]
            if max_len - lens[i]:
                seq.append(torch.zeros(max_len - lens[i], self.encode_dim).cuda())
            seq.append(encodes[start:end])
            start = end
        encodes = torch.cat(seq)
        encodes = encodes.view(self.batch_size, max_len, -1)

        print('%s\t%2.2f s' % ('Pad', time() - t))
        t = time()

        gru_output, hidden = self.biGRU(encodes, torch.zeros([2, self.batch_size, self.hidden_dim]).cuda())

        print('%s\t%2.2f s' % ('GRU', time() - t))
        t = time()

        # unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(gru_output, batch_first=True)
        max_pool = torch.max(gru_output, 1).values

        print('%s\t%2.2f s' % ('Pooling', time() - t))
        t = time()

        y = self.FC(max_pool)

        print('%s\t%2.2f s' % ('Linear', time() - t))

        return y
