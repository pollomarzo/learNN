"""
quantlenet implemented as brevitas example
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn
from brevitas.core.quant import QuantType
from torch.nn import Module


class QuantLeNet(Module):
    def __init__(self, vocab_size, embedding_dim, kernel_sizes, num_filters, num_classes, d_prob, mode, TEXT):
        super(QuantLeNet, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.d_prob = d_prob
        self.mode = mode
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.load_embeddings(TEXT)

        # is this ok? if not split list comprehension. it just holds them anyway,
        # no calculations involved
        self.conv = nn.ModuleList([
            qnn.QuantConv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k, stride=1,
                weight_quant_type=QuantType.INT,
                weight_bit_width=8) for k in kernel_sizes])

        # next line is iffy. am i allowed? just remove if i cant, it helps
        # with regularization but maybe we can find another way.
        # at the same time, it *should* be REALLY easy to implement
        self.dropout = nn.Dropout(d_prob)
        self.fc = qnn.QuantLinear(len(kernel_sizes) * num_filters, num_classes,
                                  bias=True,
                                  weight_quant_type=QuantType.INT,
                                  weight_bit_width=8)

    def forward(self, x):
        batch_size, sequence_length = x.shape
        relu = qnn.QuantReLU(
            quant_type=QuantType.INT, bit_width=8, max_val=6)
        sigmoid = qnn.QuantSigmoid(
            quant_type=QuantType.INT, bit_width=8)

        x = self.embedding(x).transpose(1, 2)
        x = [relu(conv(x)) for conv in self.conv]
        x = [F.max_pool1d(c, c.size(-1)).squeeze(dim=-1) for c in x]
        x = torch.cat(x, dim=1)
        x = self.fc(self.dropout(x))
        x = sigmoid(x)
        return x.squeeze()

    def load_embeddings(self, TEXT):
        if 'static' in self.mode:
            self.embedding.weight.data.copy_(TEXT.vocab.vectors)
            if 'non' not in self.mode:
                self.embedding.weight.data.requires_grad = False
                print('Loaded pretrained embeddings, weights are not trainable.')
            else:
                self.embedding.weight.data.requires_grad = True
                print('Loaded pretrained embeddings, weights are trainable.')
        elif self.mode == 'rand':
            print('Randomly initialized embeddings are used.')
        else:
            raise ValueError(
                'Unexpected value of mode. Please choose from static, nonstatic, rand.')
