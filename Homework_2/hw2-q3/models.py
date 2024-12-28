import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


def reshape_state(state):
    h_state = state[0]
    c_state = state[1]
    new_h_state = torch.cat([h_state[:-1], h_state[1:]], dim=2)
    new_c_state = torch.cat([c_state[:-1], c_state[1:]], dim=2)
    return (new_h_state, new_c_state)


class BahdanauAttention(nn.Module):
    """
    Bahdanau attention mechanism:
    score(h_i, s_j) = v^T * tanh(W_h h_i + W_s s_j)
    """

    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size

        self.Ws = nn.Linear(hidden_size, hidden_size,bias=False)
        self.Wh = nn.Linear(hidden_size, hidden_size,bias=False)
        self.V = nn.Linear(hidden_size, 1,bias=False)
        self.Wout = nn.Linear(2*hidden_size,hidden_size,bias=False)
        #self.Wout = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        #raise NotImplementedError("Add your implementation.")

    def forward(self, query, encoder_outputs, src_lengths):
        """
        query:          (batch_size, max_tgt_len, hidden_size)
        encoder_outputs:(batch_size, max_src_len, hidden_size)
        src_lengths:    (batch_size)
        Returns:
            attn_out:   (batch_size, max_tgt_len, hidden_size) - attended vector
        """
        # Alignment score 
        #print(f"Query size:{query.size()}; Encoder_output size:{encoder_outputs.size()}")

        proj_encoder = self.Wh(encoder_outputs)  # (batch_size, max_src_len, hidden_size)
        proj_query = self.Ws(query)  # (batch_size, max_tgt_len, hidden_size)

        proj_query = proj_query.unsqueeze(2)  # Add a source-time-step dimension
        proj_encoder = proj_encoder.unsqueeze(1)  # Add a target-time-step dimension
        
        # Alignment scores: e_{ti} = v^T * tanh(W_h h_i + W_s s_t)
        e = self.V(self.tanh(proj_query + proj_encoder)).squeeze(-1) 

        #e = self.V(self.tanh(self.Ws(query)+self.Wh(encoder_outputs)))

        # Set padding to -inf to not influence the results
        mask = self.sequence_mask(src_lengths)
        e = e.transpose(1,2)
        e[~mask,:] = float("-inf")
        e = e.transpose(1,2)
        
        # Normalized using a softmax function to compute the attention weights 

        alpha = self.softmax(e)
        #print(f"Alpha size: {alpha.size()} Encoder size: {encoder_outputs.size()} Alignment score size: {e.size()} Mask size: {mask.size()}")
        # Compute context vector
        c = torch.bmm(alpha,encoder_outputs)

        #Combine context vector with the decoder hidden state and produce the attention-enhanced decoder
        #print(f"Context size: {c.size()} Query size: {query.size()} \n")

        context_repeated = c.expand(-1, query.size(1), -1)  # Shape: [64, 16, 128]
        c_ts_t = torch.cat([query, context_repeated], dim=-1) 

        attn_out = self.tanh(self.Wout(c_ts_t))

        return attn_out
        #raise NotImplementedError("Add your implementation.")

    def sequence_mask(self, lengths):
        """
        Creates a boolean mask from sequence lengths.
        True for valid positions, False for padding.
        """
        batch_size = lengths.numel()
        max_len = lengths.max()
        return (torch.arange(max_len, device=lengths.device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1)))


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        hidden_size,
        padding_idx,
        dropout,
    ):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size // 2
        self.dropout = dropout

        self.embedding = nn.Embedding(
            src_vocab_size,
            hidden_size,
            padding_idx=padding_idx,
        )
        self.lstm = nn.LSTM(
            hidden_size,
            self.hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        src,
        lengths,
    ):
        # src: (batch_size, max_src_len)
        # lengths: (batch_size)
        #############################################
        # TODO: Implement the forward pass of the encoder
        # Hints:
        # - Use torch.nn.utils.rnn.pack_padded_sequence to pack the padded sequences
        #   (before passing them to the LSTM)
        # - Use torch.nn.utils.rnn.pad_packed_sequence to unpack the packed sequences
        #   (after passing them to the LSTM)
        #############################################
        
        emb = self.embedding(src)
        
        emb = self.dropout(emb)
        #print(f"Encoder embeddings shape: {emb.size()}")
        packed_embedded = pack(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        output, final_hidden = self.lstm(packed_embedded)

        enc_output, _ = unpack(output, batch_first=True)
        #print(f"Encoder output shape: {enc_output.size()}")
        return enc_output, final_hidden
        #############################################
        # END OF YOUR CODE
        #############################################
        # enc_output: (batch_size, max_src_len, hidden_size)
        # final_hidden: tuple with 2 tensors
        # each tensor is (num_layers * num_directions, batch_size, hidden_size)


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        tgt_vocab_size,
        attn,
        padding_idx,
        dropout,
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.tgt_vocab_size = tgt_vocab_size
        self.dropout = dropout

        self.embedding = nn.Embedding(
            self.tgt_vocab_size, self.hidden_size, padding_idx=padding_idx
        )

        self.dropout = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            batch_first=True,
        )

        self.attn = attn

    def forward(
        self,
        tgt,
        dec_state,
        encoder_outputs,
        src_lengths,
    ):
        # tgt: (batch_size, max_tgt_len)
        # dec_state: tuple with 2 tensors
        # each tensor is (num_layers * num_directions, batch_size, hidden_size)
        # encoder_outputs: (batch_size, max_src_len, hidden_size)
        # src_lengths: (batch_size)
        # bidirectional encoder outputs are concatenated, so we may need to
        # reshape the decoder states to be of size (num_layers, batch_size, 2*hidden_size)
        # if they are of size (num_layers*num_directions, batch_size, hidden_size)
        if dec_state[0].shape[0] == 2:
            dec_state = reshape_state(dec_state)

        #############################################
        # TODO: Implement the forward pass of the decoder
        # Hints:
        # - the input to the decoder is the previous target token,
        #   and the output is the next target token
        # - New token representations should be generated one at a time, given
        #   the previous token representation and the previous decoder state
        # - Add this somewhere in the decoder loop when you implement the attention mechanism in 3.2:
        # if self.attn is not None:
        #     output = self.attn(
        #         output,
        #         encoder_outputs,
        #         src_lengths,
        #     )
        #############################################
       
        emb = self.embedding(tgt)
        # print(f"Decoder tgt size: {tgt.size()} and embedding size: {emb.size()}")
        emb = self.dropout(emb)
        
        output, hidden_n = self.lstm(emb, dec_state) 

        # print(f"Size pre-attention output: {output.size()}; Size pre-attention encoder: {encoder_outputs.size()}")
        if self.attn is not None:
            # print(f"Enconder_outputs len: {encoder_outputs.size()}; Src_lenght:{src_lengths.size()}")
            output = self.attn(output, encoder_outputs, src_lengths)
        # print(f"Size post-attention: {output.size()}")

        if output.size()[1] != 1:
           output = output[:,:-1,:] 

        return output, hidden_n  

        #############################################
        # END OF YOUR CODE
        #############################################
        # outputs: (batch_size, max_tgt_len, hidden_size)
        # dec_state: tuple with 2 tensors
        # each tensor is (num_layers, batch_size, hidden_size)
     


class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
    ):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.generator = nn.Linear(decoder.hidden_size, decoder.tgt_vocab_size)

        self.generator.weight = self.decoder.embedding.weight

    def forward(
        self,
        src,
        src_lengths,
        tgt,
        dec_hidden=None,
    ):

        encoder_outputs, final_enc_state = self.encoder(src, src_lengths)

        if dec_hidden is None:
            dec_hidden = final_enc_state

        output, dec_hidden = self.decoder(
            tgt, dec_hidden, encoder_outputs, src_lengths
        )

        return self.generator(output), dec_hidden
