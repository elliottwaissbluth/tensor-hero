import torch
from torch import nn
import torch.optim as optim
import numpy as np
from pathlib import Path

def formatted_notes_to_indices(notes):
    '''
    takes formatted notes and returns a 1D array of the indices within which it is equal to 1
    '''
    # Loop through each row
    indices = np.argwhere(notes == 1)
    indices = indices[:,-1].flatten()
    return indices

class Dataset(torch.utils.data.Dataset):
    def __init__(self, training_path):
        # Load input and output
        self.notes = np.load(training_path / 'formatted_notes.npy')
        self.spec = np.load(training_path / 'spectrogram_slice.npy')

        # Concatenate two extra dimensions for SOS and EOS to self.notes
        notes_append = np.zeros(shape=(self.notes.shape[0], 2))
        self.notes = np.c_[self.notes, notes_append]
        # Add a row at the beginning and end of note for <sos>, <eos>
        notes_append = np.zeros(shape=(1,self.notes.shape[1]))
        self.notes = np.vstack([notes_append, self.notes, notes_append])
        # Add proper values to self.notes
        self.notes[0,-2] = 1  # <sos>
        self.notes[-1,-1] = 1 # <eos>
        self.notes = formatted_notes_to_indices(self.notes)
        self.notes = torch.tensor(self.notes)
      
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return torch.tensor(self.spec, dtype=torch.float), self.notes

class InputEmbedding(nn.Module):
    def __init__(
        self,
        embedding_size,
        dropout,
    ):
        super(InputEmbedding, self).__init__()

        # Take the spectrogram frames and pass them through a FC layer
        self.linear = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.Sigmoid(),
        )
    
    def forward(self, src):
        return self.linear(src)




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
        max_len, # Note, max len is the same for both the input and the output
        device,
    ):
        super(Transformer, self).__init__()
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        # self.src_word_embedding = InputEmbedding(embedding_size, dropout)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
            batch_first=True,
        )

        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

        self.src_word_embedding = nn.ModuleList(
            [
            InputEmbedding(embedding_size, dropout) for _ in range(max_len)
            ]
        )
     

    def make_src_mask(self, src):
        # src shape: (src_len, N)
        # src_mask = src.transpose(0, 1) == self.src_pad_idx
        
        # Create tensor of all "False" values for single input
        src_mask = torch.zeros(1, 400, dtype=torch.bool)
        # (N, src_len)
        return src_mask
    
    def make_embed_src(self, src):
        out = torch.zeros_like(src).to(self.device)
        for idx, emb in enumerate(self.src_word_embedding):
            out[...,idx] = emb(src[...,idx])

        return out


    def forward(self, src, trg):
        src_seq_length, N = src.shape[2], src.shape[0]
        trg_seq_length, N = trg.shape[1], trg.shape[0]

        src_padding_mask = self.make_src_mask(src).to(self.device)
        
        src_positions = (
            torch.arange(0, src_seq_length).unsqueeze(1).expand(src_seq_length, N)
            .to(self.device)
            )

        
        trg_positions = (
            torch.arange(0, trg_seq_length).unsqueeze(1).expand(trg_seq_length, N).permute(1,0)
            .to(self.device)
        )
        # TODO: Make embed src from spectrogram frames

        embed_src = self.dropout(
            (self.make_embed_src(src) + self.src_position_embedding(src_positions).permute(1,2,0))
            .to(self.device)
        ).permute(0,2,1)

        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
            .to(self.device)
        )

        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(self.device)

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask = src_padding_mask,
            tgt_mask=trg_mask
        )

        out = self.fc_out(out)
        return out


if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter

    train_path = Path.cwd() / 'Model_1' / 'toy training data' / 'preprocessed'
    # Define dataset
    train_data = Dataset(train_path)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Training hyperparameters
    num_epochs = 10
    learning_rate = 1e-4
    batch_size = 1

    # Model hyperparameters
    # src_vocab_size = 0 # There is no src vocab since the src is spectrogram frames
    trg_vocab_size = 0 # <output length>
    embedding_size = 512
    num_heads = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    dropout = 0.1
    max_len = 400
    forward_expansion = 2048
    src_pad_idx = 434

    # Tensorboard for nice plots
    writer = SummaryWriter('runs/loss_plot')
    step = 0

    # Define model
    model = Transformer(
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
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    pad_idx = 0 # TODO 
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            spec, notes = batch[0].to(device), batch[1].to(device)

            # forward prop
            output = model(spec, notes[...,:-1])

            output = output.reshape(-1, output.shape[2])
            notes = notes[1:].reshape(-1)
            optimizer.zero_grad()

            loss = criterion(output, notes)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            optimizer.step()
            writer.add_scalar("Training Loss", loss, global_step=step)
            step += 1

