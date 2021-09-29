import torch
from torch import nn
import torch.optim as optim
import numpy as np
from pathlib import Path

class Dataset(torch.utils.data.Dataset):
    def __init__(self, training_path):
        # Load input and output
        self.notes = np.load(training_path / 'formatted_notes.npy')
        self.spec = np.load(training_path / 'spectrogram_slice.npy')
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        # TODO
        return

class InputEmbedding(nn.Module):
    def __init__(
        self,
        src,
        embedding_size,
        max_len,
        dropout,
    ):
        super(InputEmbedding, self).__init__()

        # Take the spectrogram frames and pass them through a FC layer



class Transformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len,
        device,
    ):
        super(Transformer, self).__init__()
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)
        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )

        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        # src shape: (src_len, N)
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        # (N, src_len)
        return src_mask

    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        src_positions = (
            torch.arange(0, src_seq_length).unsqueeze(1).expand(trg_seq_length, N)
            .to(self.device)
            )
        
        trg_positions = (
            torch.arange(0, trg_seq_length).unsqueeze(1).expand(trg_seq_length, N)
            .to(self.device)
        )

        # TODO: Make embed src from spectrogram frames
        embed_src = self.dropout(
            (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        )

        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
        )

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(self.device)

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask = src_padding_mask,
            tgt_mask=trg_mask
        )

        out = self.fc_out(out)

        return out