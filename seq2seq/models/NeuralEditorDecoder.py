import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention
from .baseRNN import BaseRNN

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device


class NeuralEditorDecoder(BaseRNN):
    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, vocab_size, max_len, hidden_size,
            sos_id, eos_id,
            n_layers=1, rnn_cell='gru', bidirectional=False,
            input_dropout_p=0, dropout_p=0, use_attention=False, embedding=None, update_embedding=True):

        super(NeuralEditorDecoder, self).__init__(vocab_size, max_len, hidden_size,
                                                  input_dropout_p, dropout_p,
                                                  n_layers, rnn_cell)

        self.bidirectional_encoder = bidirectional
        self.factor = 2 if self.bidirectional_encoder else 1
        self.rnn = self.rnn_cell( 2 * hidden_size * self.factor, hidden_size * self.factor, n_layers, batch_first=True, dropout=dropout_p)
        #all the vovabary
        self.output_size = vocab_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.init_input = None

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)
        self.embedding.weight.requires_grad = update_embedding
        if use_attention:
            self.attention = Attention(self.hidden_size)

        self.gate = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward_step(self, input_var, hidden, encoder_z, encoder_outputs, function):

        # barch_size * max_len * embedding_size
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        if len(encoder_z.shape) == 2:
            encoder_z_ = encoder_z.unsqueeze(1)
        else:
            encoder_z_ = encoder_z

        if encoder_z_.shape[0] != embedded.shape[0]:
            encoder_z_ = encoder_z_.transpose(1, 0)
        else:
            encoder_z_ = encoder_z_

        if embedded.shape[1] > encoder_z_.shape[1]:
            encoder_z_ = encoder_z_.expand(-1, embedded.shape[1], -1)
        embedded = torch.cat((embedded, encoder_z_), 2)


        #batch, seq_len, num_directions * hidden_size)
        output, hidden = self.rnn(embedded, hidden)
        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)

        predicted_softmax = function(torch.matmul(output,  self.embedding.weight.transpose(1,0)), dim=-1)
        # predicted_softmax = function(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1).view(batch_size, output_size, -1)
        return predicted_softmax, hidden, attn

    def forward(self, inputs=None, encoder_hidden=None, encoder_z=None, z_dis=None, encoder_outputs=None,
                    function=F.log_softmax, teacher_forcing_ratio=0):
        ret_dict = dict()
        if self.use_attention:
            ret_dict[NeuralEditorDecoder.KEY_ATTN_SCORE] = list()

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, None,
                                                             function, teacher_forcing_ratio)
        decoder_hidden = self._init_state(encoder_hidden)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output, step_attn):
            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[NeuralEditorDecoder.KEY_ATTN_SCORE].append(step_attn)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph
        if use_teacher_forcing:
            decoder_input = inputs[:, :-1]
            decoder_output, decoder_hidden, attn = self.forward_step(decoder_input, decoder_hidden, encoder_z, encoder_outputs,
                                                                     function=function)

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                if attn is not None:
                    step_attn = attn[:, di, :]
                else:
                    step_attn = None
                decode(di, step_output, step_attn)
        else:
            decoder_input = inputs[:, 0].unsqueeze(1)
            for di in range(max_length):
                decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, encoder_z, encoder_outputs,
                                                                         function=function)
                step_output = decoder_output.squeeze(1)
                symbols = decode(di, step_output, step_attn)
                decoder_input = symbols

        ret_dict[NeuralEditorDecoder.KEY_SEQUENCE] = sequence_symbols
        ret_dict[NeuralEditorDecoder.KEY_LENGTH] = lengths.tolist()
        return decoder_outputs, z_dis, ret_dict

    def _gate_hidden(self, decoder_hidden, encoder_hidden):

        m_t = torch.sigmoid(self.gate(decoder_hidden))

        return decoder_hidden + encoder_hidden * m_t

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1 # minus the start of sequence symbol

        return inputs, batch_size, max_length
