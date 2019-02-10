from torch.nn import Parameter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scaled_dot_attention import ScaledDotAttention


class Encoder(torch.nn.Module):
    """
    Takes as input the raw features, projects them, passes through an LSTM to produce hidden states (encodings: e1 .... eT).
    """
    def __init__(self, n_in: int, n_embed: int, hidden_size: int, birnn: bool = False, n_layers: int = 1) -> None:
        super().__init__()
        if birnn or n_layers > 1:
            raise NotImplementedError()

        self.proj = torch.nn.Linear(n_in, n_embed)
        self.rnn = torch.nn.LSTM(input_size=n_embed, hidden_size=hidden_size, num_layers=n_layers, bidirectional=birnn, batch_first=True)
    
    def forward(self, sequences: torch.Tensor, seq_lens: np.ndarray):
        # sequences: batch * seq_len * n_dim
        embedded = self.proj(sequences)
        # FIXME: should we add relu?
        
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, seq_lens, batch_first=True)
        hiddens, (hT, cT) = self.rnn(packed)  # outputs: batch * seq_len * (num_directions * n_hidden)
        return hiddens, (hT, cT)


class Decoder(torch.nn.Module):
    """
    Uses the encoder hidden states, produces target_length outputs.
    """
    def __init__(self, hidden_size: int = 256, d_k: int = 256) -> None:
        super().__init__()

        self.rnn = torch.nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
        
        # For attention, key - encoder hidden state, query - decoder hidden state, value=one-hot for each position
        self.sda = ScaledDotAttention(d_k=d_k, d_q=hidden_size, model_dim=64)
    
    def forward(self, max_length: int, encoder_hiddens, seq_lens, h0, c0):
        """
        target_length: the length of the sequence to produce: FIXME: need <EOS>
        """
        batch_size, max_seq_len, n_features = encoder_hiddens.shape
        h0, c0 = h0.view(batch_size, -1), c0.view(batch_size, -1)

        assert max_seq_len == seq_lens[0], 'The encoder hidden matrix should: {} time steps. Actual: {}'.format(max_seq_len, seq_lens[0])
        assert np.all(np.array([batch_size, n_features]) == h0.shape), 'Shape mismatch of h0: {}, expected: {}'.format(h0.shape, [batch_size, n_features])
    
        # Iterate for max_length time steps and produce sequences!
        pointers = []
        hidden_prev, cell_state_prev = h0, c0
        
        for ix in range(max_length):
            # FIXME: Usually, the outputs of the network are passed as the next input. However, the outputs are of varying sizes here
            # So, that's not possible.. For now, I'll pass the previous hidden state.
            decoder_hidden, cell_state = self.rnn(hidden_prev, (hidden_prev, cell_state_prev))

            queries = decoder_hidden.view(batch_size, 1, -1)  # batch_size * n_queries * d_q

            # Mask for the attention weights: for each sequence in the batch: 1 <query> * seq_len <keys> - since the encoder hidden states are
            # padded, we need to mask them out
            mask = np.zeros((batch_size, 1, seq_lens[0]), dtype=np.uint8)
            for ix, seq_len in enumerate(seq_lens):
                mask[ix, 0, seq_len:] = 1
            mask = torch.from_numpy(mask).to(decoder_hidden.device)
            
            # Calculate attention weights
            att_weights = self.sda(K=encoder_hiddens, V=None, Q=queries, mask=mask, return_probs=True)
            assert np.all(att_weights.shape == np.array([batch_size, 1, max_seq_len]))
            att_weights = att_weights.view(1, batch_size, -1)

            # Store
            pointers.append(att_weights)
            hidden_prev = decoder_hidden.view(batch_size, -1)  # reshaping since the output is batch_size * 1 * dim
            cell_state_prev = cell_state.view(batch_size, -1)
            
            # FIXME: break if <EOS>
        
        pointers = torch.cat(pointers, dim=0)  # max_length * batch_size * max_seq_len
        pointers = pointers.transpose(0, 1)  # batch_size, * max_length * max_seq_len
        return pointers

class PointerNet(torch.nn.Module):
    def __init__(self, n_in=2, hidden_size=256, embed_size=256) -> None:
        super().__init__()
        self.encoder = Encoder(n_in=n_in, n_embed=embed_size, hidden_size=hidden_size)
        self.decoder = Decoder(hidden_size=hidden_size)

        self.hidden_size = hidden_size
    
    def forward(self, sequence, seq_lens, max_output_len):
        batch_size = sequence.shape[0]

        encoder_hiddens, (hT, cT) = self.encoder(sequence, seq_lens)
        encoder_hiddens_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(encoder_hiddens, batch_first=True)
        assert np.all(encoder_hiddens_padded.shape == np.array([batch_size, seq_lens[0], self.hidden_size]))

        pointers = self.decoder(max_length=max_output_len, encoder_hiddens=encoder_hiddens_padded, seq_lens=seq_lens, h0=hT, c0=cT)
        assert np.all(pointers.shape == np.array([batch_size, max_output_len, seq_lens[0]]))
        
        return pointers.view(batch_size, max_output_len, -1)


def test_decoder():
    encoder = Encoder(n_in=2, n_embed=256, hidden_size=256, birnn=False, n_layers=1)
    decoder = Decoder(n_hidden=256, d_k=256)

    encoder_hiddens, (hT, cT) = encoder(seq, seq_lens)
    encoder_hiddens_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(encoder_hiddens, batch_first=True)

    pointers = decoder(max_length=3, encoder_hiddens=encoder_hiddens_padded, seq_lens=seq_lens, h0=hT, c0=cT)

    # Test the masking
    max_seq_len = seq_lens[0]
    n_outs = pointers.shape[1]
    for ix in range(len(seq_lens)):
        seq_len = seq_lens[ix]
        pointer_probs = pointers[ix].data.numpy()
        n_padded = max_seq_len - seq_len
        if n_padded > 0:
            assert np.all(pointer_probs[:, seq_len:] == np.zeros((n_outs, n_padded)))