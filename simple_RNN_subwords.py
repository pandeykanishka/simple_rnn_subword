
# coding: utf-8

# In[1]:

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import os

import torch
import torch.utils.data
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from data_loader import Dataset
from _utils.transformer import *
import argparse


# In[2]:

# Predefined tokens
SOS_token = 1   # SOS_token: start of sentence
EOS_token = 2   # EOS_token: end of sentence

# Useful function for arguments.
def str2bool(v):
    return v.lower() in ("yes", "true")

# Parser
parser = argparse.ArgumentParser(description='Creating Classifier')


# In[3]:

# Optimization Flags #
######################

parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--epochs_per_lr_drop', default=450, type=float,
                    help='number of epochs for which the learning rate drops')

##################
# Training Flags #
##################
parser.add_argument('--num_epochs', default=10, type=int, help='Number of epochs for training for training')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--num_epoch', default=600, type=int, help='Number of training iterations')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--save_folder', default=os.path.expanduser('~/weights'), help='Location to save checkpoint models')
parser.add_argument('--epochs_per_save', default=10, type=int,
                    help='number of epochs for which the model will be saved')
parser.add_argument('--batch_per_log', default=10, type=int, help='Print the log at what number of batches?')

###############
# Model Flags #
###############

parser.add_argument('--auto_encoder', default=True, type=str2bool, help='Use auto-encoder model')
parser.add_argument('--MAX_LENGTH', default=10, type=int, help='Maximum length of sentence')
parser.add_argument('--bidirectional', default=False, type=str2bool, help='bidirectional LSRM')
parser.add_argument('--hidden_size_decoder', default=256, type=int, help='Decoder Hidden Size')
parser.add_argument('--num_layer_decoder', default=1, type=int, help='Number of LSTM layers for decoder')
parser.add_argument('--hidden_size_encoder', default=256, type=int, help='Eecoder Hidden Size')
parser.add_argument('--num_layer_encoder', default=1, type=int, help='Number of LSTM layers for encoder')
parser.add_argument('--teacher_forcing', default=False, type=str2bool, help='If using the teacher frocing in decoder')


# In[4]:

import sys
sys.argv=['']
del sys


# In[5]:

# Add all arguments to parser
args = parser.parse_args()


# In[6]:

# Cuda Flags #
##############
if args.cuda:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")


# In[7]:

# Create training data object
trainset = Dataset(phase='train', max_input_length=10, auto_encoder=args.auto_encoder)

# Extract the languages' attributes
input_lang, output_lang = trainset.langs()


# The trainloader for parallel processing
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)
# iterate through training
dataiter = iter(trainloader)

# Create testing data object
testset = Dataset(phase='test', max_input_length=10, auto_encoder=args.auto_encoder)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                          shuffle=True, num_workers=1, pin_memory=False, drop_last=True)


# In[8]:

# Encoder / Decoder #
#####################

class EncoderRNN(nn.Module):
    """
    The encoder generates a single output vector that embodies the input sequence meaning.
    The general procedure is as follows:
        1. In each step, a word will be fed to a network and it generates
         an output and a hidden state.
        2. For the next step, the hidden step and the next word will
         be fed to the same network (W) for updating the weights.
        3. In the end, the last output will be the representative of the input sentence (called the "context vector").
    """
    def __init__(self, hidden_size, input_size, batch_size, num_layers=1, bidirectional=False):
        """
        * For nn.LSTM, same input_size & hidden_size is chosen.
        :param input_size: The size of the input vocabulary
        :param hidden_size: The hidden size of the RNN.
        :param batch_size: The batch_size for mini-batch optimization.
        :param num_layers: Number of RNN layers. Default: 1
        :param bidirectional: If the encoder is a bi-directional LSTM. Default: False
        """
        super(EncoderRNN, self).__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        # The input should be transformed to a vector that can be fed to the network.
        self.embedding = nn.Embedding(input_size, embedding_dim=hidden_size)

        # The LSTM layer for the input
        if args.bidirectional:
            self.lstm_forward = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers)
            self.lstm_backward = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers)
        else:
            self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers)


    def forward(self, input, hidden):

        if args.bidirectional:
            input_forward, input_backward = input
            hidden_forward, hidden_backward = hidden
            input_forward = self.embedding(input_forward).view(1, 1, -1)
            input_backward = self.embedding(input_backward).view(1, 1, -1)

            out_forward, (h_n_forward, c_n_forward) = self.lstm_forward(input_forward, hidden_forward)
            out_backward, (h_n_backward, c_n_backward) = self.lstm_backward(input_backward, hidden_backward)

            forward_state = (h_n_forward, c_n_forward)
            backward_state = (h_n_backward, c_n_backward)
            output_state = (forward_state, backward_state)

            return output_state
        else:
            # Make the data in the correct format as the RNN input.
            embedded = self.embedding(input).view(1, 1, -1)
            rnn_input = embedded
            # The following descriptions of shapes and tensors are extracted from the official Pytorch documentation:
            # output-shape: (seq_len, batch, num_directions * hidden_size): tensor containing the output features (h_t) from the last layer of the LSTM
            # h_n of shape (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state
            # c_n of shape (num_layers * num_directions, batch, hidden_size): tensor containing the cell state
            output, (h_n, c_n) = self.lstm(rnn_input, hidden)
            return output, (h_n, c_n)

    def initHidden(self):

        if self.bidirectional:
            encoder_state = [torch.zeros(self.num_layers, 1, self.hidden_size, device=device),
                                      torch.zeros(self.num_layers, 1, self.hidden_size, device=device)]
            encoder_state = {"forward": encoder_state, "backward": encoder_state}
            return encoder_state
        else:
            encoder_state = [torch.zeros(self.num_layers, 1, self.hidden_size, device=device),
                              torch.zeros(self.num_layers, 1, self.hidden_size, device=device)]
            return encoder_state


# In[9]:

class DecoderRNN(nn.Module):
    """
    This context vector, generated by the encoder, will be used as the initial hidden state of the decoder.
    Decoding is as follows:
    1. At each step, an input token and a hidden state is fed to the decoder.
        * The initial input token is the <SOS>.
        * The first hidden state is the context vector generated by the encoder (the encoder's
    last hidden state).
    2. The first output, shout be the first sentence of the output and so on.
    3. The output token generation ends with <EOS> being generated or the predefined max_length of the output sentence.
    """
    def __init__(self, hidden_size, output_size, batch_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output, (h_n, c_n) = self.lstm(output, hidden)
        output = self.out(output[0])
        return output, (h_n, c_n)

    def initHidden(self):
        """
        The spesific type of the hidden layer for the RNN type that is used (LSTM).
        :return: All zero hidden state.
        """
        return [torch.zeros(self.num_layers, 1, self.hidden_size, device=device),
                torch.zeros(self.num_layers, 1, self.hidden_size, device=device)]


# In[10]:

class Linear(nn.Module):
    """
    This context vector, generated by the encoder, will be used as the initial hidden state of the decoder.
    In case that their dimension is not matched, a linear layer should be used to transformed the context vector
    to a suitable input (shape-wise) for the decoder cell state (including the memory(Cn) and hidden(hn) states).
    The shape mismatch is True in the following conditions:
    1. The hidden sizes of encoder and decoder are the same BUT we have a bidirectional LSTM as the Encoder.
    2. The hidden sizes of encoder and decoder are NOT same.
    3. ETC?
    """

    def __init__(self, bidirectional, hidden_size_encoder, hidden_size_decoder):
        super(Linear, self).__init__()
        self.bidirectional = bidirectional
        num_directions = int(bidirectional) + 1
        self.linear_connection_op = nn.Linear(num_directions * hidden_size_encoder, hidden_size_decoder)
        self.connection_possibility_status = num_directions * hidden_size_encoder == hidden_size_decoder

    def forward(self, input):

        if self.connection_possibility_status:
            return input
        else:
            return self.linear_connection_op(input)


# In[11]:

import pandas as pd
import numpy as np
import string
from string import digits
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import re


# In[12]:

from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model


# In[13]:

lines= pd.read_table('eng-fra.txt', names=['English', 'French'])


# In[14]:

lines.English


# In[15]:

from bpemb import BPEmb


# In[16]:

type(lines.English)


# In[17]:

bpemb_en = BPEmb(lang="en")


# In[18]:

bpemb_fr = BPEmb(lang="fr")


# In[19]:

subword_English=bpemb_en.encode(lines.English)


# In[20]:

subword_English


# In[21]:

type(subword_English)


# In[22]:

import pandas as pd

#thelist = [ ['sentence 1'], ['sentence 2'], ['sentence 3'] ]
sub_Eng = pd.Series( (v[0:] for v in subword_English) )


# In[23]:

type(sub_Eng)


# In[24]:

sub_Eng


# In[25]:

print( ','.join( repr(e) for e in sub_Eng ) )


# In[26]:

def listToStringWithoutBrackets(list1):
    return str(list1).replace('[','').replace(']','')


# In[27]:

new=listToStringWithoutBrackets(sub_Eng)


# In[28]:

new


# In[29]:

subword_French=bpemb_fr.encode(lines.French)


# In[30]:

subword_French


# In[31]:

sub_French=[]
for word in subword_French:
    #new=word[0]+""+word[1]
    #word.pop(0)
    #word.pop(0)
    word.insert(0, '_start')
    #word.pop(1)
    sub_French.append(word)
    print(word)


# In[32]:

sub_French


# In[33]:

import pickle


# In[34]:

pickle.dump(sub_French, open('sub_French.pickle', 'wb'))


# In[35]:

lines.shape


# In[36]:

# Lowercase all characters
lines.English=lines.English.apply(lambda x: x.lower())
lines.French=lines.French.apply(lambda x: x.lower())


# In[37]:

# Remove quotes
lines.English=lines.English.apply(lambda x: re.sub("'", '', x))
lines.French=lines.French.apply(lambda x: re.sub("'", '', x))


# In[38]:

import string


# In[39]:

exclude = set(string.punctuation) # Set of all special characters
# Remove all the special characters
lines.English=lines.English.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
lines.French=lines.French.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))


# In[40]:

# Remove all numbers from text
remove_digits = str.maketrans('', '', digits)
lines.English=lines.English.apply(lambda x: x.translate(remove_digits))
lines.French = lines.French.apply(lambda x: x.translate(remove_digits))


# In[41]:

# Remove extra spaces
lines.English=lines.English.apply(lambda x: x.strip())
lines.French=lines.French.apply(lambda x: x.strip())
print(lines.English)
lines.English=lines.English.apply(lambda x: re.sub(" +", " ", x))
lines.French=lines.French.apply(lambda x: re.sub(" +", " ", x))


# In[42]:

# Add start and end tokens to target sequences
lines.French = lines.French.apply(lambda x : 'START_ '+ x + ' _END')


# In[43]:

lines.French


# In[44]:

lines.sample(10)


# In[45]:

type(subword_English)


# In[46]:

import pandas as pd

#thelist = [ ['sentence 1'], ['sentence 2'], ['sentence 3'] ]
sub_Fren = pd.Series( (v[:] for v in subword_French) )


# In[47]:

sub_Fren


# In[48]:

type(sub_Fren)


# In[49]:

sub_Fren


# In[50]:

sub_Eng


# In[51]:

type(sub_Eng)


# In[52]:

sub_Eng


# In[89]:

# Vocabulary of English sub_words
all_Eng_sub_words=set()
for Eng in sub_Eng:
    #print(eng)
    for word in Eng:
        #print(word)
        if word not in all_Eng_sub_words:
            all_Eng_sub_words.add(word)


# In[90]:

pickle.dump(all_Eng_sub_words, open('sub_eng.pickle', 'wb'))


# In[54]:

sub_Fren = pickle.load(open('sub_French.pickle', 'rb'))


# In[87]:

# Vocabulary of french sub_words
all_French_sub_words=set()
for Eng in sub_Fren:
    #print(Eng)
    for word in Eng:
        #print(word)
        if word not in all_French_sub_words:
            all_French_sub_words.add(word)


# In[88]:

pickle.dump(all_French_sub_words, open('sub_French.pickle', 'wb'))


# In[ ]:




# In[56]:

# Max Length of source sequence
lenght_list=[]
for l in lines.English:
    print(l)
    lenght_list.append(len(l.split(' ')))
    print(lenght_list)
max_length_src = np.max(lenght_list)
max_length_src


# In[57]:

# Max Length of source sequence of sub_eng
lenght_list=[]
for l in sub_Eng:
    print(l)
    lenght_list.append(len(l))
    #lenght_list.append(len(l.split(' ')))
    print(lenght_list)
max_length_src = np.max(lenght_list)
max_length_src


# In[58]:

# Max Length of target sequence of sub_french
lenght_list=[]
for l in sub_Fren:
    print(l)
    lenght_list.append(len(l))
    #lenght_list.append(len(l.split(' ')))
    print(lenght_list)
max_length_tar = np.max(lenght_list)
max_length_tar


# In[59]:

# Max Length of target sequence
lenght_list=[]
for l in lines.French:
    print(l)
    lenght_list.append(len(l.split(' ')))
max_length_tar = np.max(lenght_list)
max_length_tar


# In[60]:

input_words = sorted(list(all_Eng_sub_words))
target_words = sorted(list(all_French_sub_words))
num_encoder_tokens = len(all_Eng_sub_words)
num_decoder_tokens = len(all_French_sub_words)
num_encoder_tokens, num_decoder_tokens


# In[61]:

num_decoder_tokens += 1 # For zero padding
num_decoder_tokens


# In[91]:

input_token_index = dict([(word, i+1) for i, word in enumerate(input_words)])


# In[93]:

pickle.dump(input_token_index, open('ENG_DICT.pickle', 'wb'))


# In[92]:

input_token_index


# In[94]:

target_token_index = dict([(word, i+1) for i, word in enumerate(target_words)])


# In[95]:

pickle.dump(target_token_index, open('FRENCH_DICT.pickle', 'wb'))


# In[96]:

reverse_input =dict((i,word) for word,i in input_token_index.items())


# In[98]:

pickle.dump(reverse_input, open('REV_ENG_DICT.pickle', 'wb'))


# In[97]:

reverse_target =dict((i,word) for word,i in target_token_index.items())


# In[99]:

pickle.dump(reverse_target, open('REV_FRENCH_DICT.pickle', 'wb'))


# In[65]:

input_token_index.items()


# In[66]:

from sklearn.model_selection import train_test_split


# In[67]:

latent_dim = 10


# In[68]:

from torch import nn, tensor


# In[69]:

emb_layer = nn.Embedding.from_pretrained(tensor(bpemb_fr.vectors))


# In[70]:

emb_layer


# In[71]:

ids = bpemb_fr.encode_ids("Ceci est une phrase française")


# In[72]:

ids


# In[73]:

bpemb_fr.vectors[ids].shape


# In[74]:

emb_layer(tensor(ids)).shape


# In[75]:

bpemb_fr.embed("Ceci est une phrase française").shape


# In[76]:

bpemb_en.most_similar("shire", topn=20)


# In[77]:

# Train - Test Split
X, y = sub_Eng,sub_Fren
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
X_train.shape, X_test.shape


# In[78]:

X_train


# In[79]:

X_train.to_pickle('X_train.pkl')
X_test.to_pickle('X_test.pkl')


# In[80]:

# Training the Model #
######################

def train(input_tensor, target_tensor, mask_input, mask_target, encoder, decoder, bridge, encoder_optimizer, decoder_optimizer, bridge_optimizer, criterion, max_length=args.MAX_LENGTH):
    # To train, each element of the input sentence will be fed to the encoder.
    # At the decoding phase``<SOS>`` will be fed as the first input to the decoder
    # and the last hidden (state,cell) of the encoder will play the role of the first hidden (cell,state) of the decoder.

    # optimizer steps
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    bridge_optimizer.zero_grad()

    # Define a list for the last hidden layer
    encoder_hiddens_last = []
    loss = 0

    #################
    #### DECODER ####
    #################
    for step_idx in range(args.batch_size):
        # reset the LSTM hidden state. Must be done before you run a new sequence. Otherwise the LSTM will treat
        # the new input sequence as a continuation of the previous sequence.
        encoder_hidden = encoder.initHidden()
        input_tensor_step = input_tensor[:, step_idx][input_tensor[:, step_idx] != 0]
        input_length = input_tensor_step.size(0)

        # Switch to bidirectional mode
        if args.bidirectional:
            encoder_outputs = torch.zeros(args.batch_size, max_length, 2 * encoder.hidden_size, device=device)
            encoder_hidden_forward = encoder_hidden['forward']
            encoder_hidden_backward = encoder_hidden['backward']
            for ei in range(input_length):
                (encoder_hidden_forward, encoder_hidden_backward) = encoder(
                    (input_tensor_step[ei],input_tensor_step[input_length - 1 - ei]), (encoder_hidden_forward,encoder_hidden_backward))

            # Extract the hidden and cell states
            hn_forward, cn_forward = encoder_hidden_forward
            hn_backward, cn_backward = encoder_hidden_backward

            # Concatenate the hidden and cell states for forward and backward paths.
            encoder_hn = torch.cat((hn_forward, hn_backward), 2)
            encoder_cn = torch.cat((cn_forward, cn_backward), 2)


            # Only return the hidden and cell states for the last layer and pass it to the decoder
            encoder_hn_last_layer = encoder_hn[-1].view(1, 1, -1)
            encoder_cn_last_layer = encoder_cn[-1].view(1,1,-1)

            # The list of states
            encoder_hidden = [encoder_hn_last_layer, encoder_cn_last_layer]


        else:
            encoder_outputs = torch.zeros(args.batch_size, max_length, encoder.hidden_size, device=device)
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(
                    input_tensor_step[ei], encoder_hidden)
                encoder_outputs[step_idx, ei, :] = encoder_output[0, 0]

            # only return the hidden and cell states for the last layer and pass it to the decoder
            hn, cn = encoder_hidden
            encoder_hn_last_layer = hn[-1].view(1,1,-1)
            encoder_cn_last_layer = cn[-1].view(1,1,-1)
            encoder_hidden = [encoder_hn_last_layer, encoder_cn_last_layer]

        # A linear layer to establish the connection between the encoder/decoder layers.
        encoder_hidden = [bridge(item) for item in encoder_hidden]
        encoder_hiddens_last.append(encoder_hidden)
        
           #### DECODER ####
    #################
    decoder_input = torch.tensor([SOS_token], device=device)
    decoder_hiddens = encoder_hiddens_last

    # teacher_forcing uses the real target outputs as the next input
    # rather than using the decoder's prediction.
    if args.teacher_forcing:

        for step_idx in range(args.batch_size):
            # reset the LSTM hidden state. Must be done before you run a new sequence. Otherwise the LSTM will treat
            # the new input sequence as a continuation of the previous sequence

            target_tensor_step = target_tensor[:, step_idx][target_tensor[:, step_idx] != 0]
            target_length = target_tensor_step.size(0)
            decoder_hidden = decoder_hiddens[step_idx]
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
                # decoder_output, decoder_hidden, decoder_attention = decoder(
                #     decoder_input, decoder_hidden, encoder_outputs)

                loss += criterion(decoder_output, target_tensor_step[di].view(1))
                decoder_input = target_tensor_step[di]  # Teacher forcing

        loss = loss / args.batch_size

    else:
        for step_idx in range(args.batch_size):
            # reset the LSTM hidden state. Must be done before you run a new sequence. Otherwise the LSTM will treat
            # the new input sequence as a continuation of the previous sequence

            target_tensor_step = target_tensor[:, step_idx][target_tensor[:, step_idx] != 0]
            target_length = target_tensor_step.size(0)
            decoder_hidden = decoder_hiddens[step_idx]

            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                # decoder_output, decoder_hidden, decoder_attention = decoder(
                #     decoder_input, decoder_hidden, encoder_outputs)
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += criterion(decoder_output, target_tensor_step[di].view(1))
                if decoder_input.item() == EOS_token:
                    break
        loss = loss / args.batch_size

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


# In[86]:

# # This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))




def trainIters(encoder, decoder, bridge, print_every=1000, plot_every=100, learning_rate=0.1):

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    bridge_optimizer = optim.SGD(bridge.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    n_iters_per_epoch = int(len(trainset) / args.batch_size)
    for i in range(args.num_epochs):

        for iteration, data in enumerate(trainloader, 1):

            # Get a batch
            training_pair = data

            # Input
            input_tensor = training_pair['sub_Eng'][:,:,0,:]
            input_tensor, mask_input = reformat_tensor_mask(input_tensor)

            # Target
            target_tensor = training_pair['sub_Fren'][:,:,1,:]
            target_tensor, mask_target = reformat_tensor_mask(target_tensor)

            if device == torch.device("cuda"):
                input_tensor = input_tensor.cuda()
                target_tensor = target_tensor.cuda()

            loss = train(input_tensor, target_tensor, mask_input, mask_target, encoder,
                         decoder, bridge, encoder_optimizer, decoder_optimizer, bridge_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if iteration % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iteration / n_iters_per_epoch),
                                             iteration, iteration / n_iters_per_epoch * 100, print_loss_avg))

            if iteration % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0


        print('####### Finished epoch %d of %d ########' % (i+1, args.num_epochs))


# In[82]:

# Training and Evaluating
# =======================

encoder1 = EncoderRNN(args.hidden_size_encoder, input_lang.n_words, args.batch_size, num_layers=args.num_layer_encoder, bidirectional=args.bidirectional).to(device)
bridge = Linear(args.bidirectional, args.hidden_size_encoder, args.hidden_size_decoder).to(device)
decoder1 = DecoderRNN(args.hidden_size_decoder, output_lang.n_words, args.batch_size, num_layers=args.num_layer_decoder).to(device)

trainIters(encoder1, decoder1, bridge, print_every=10)

######################################################################
evaluateRandomly(encoder1, decoder1, bridge)


# In[83]:

##############
# Evaluation #
##############
#
# In evaluation, we simply feed the sequence and observe the output.
# The generation will be over once the "EOS" has been generated.

def evaluate(encoder, decoder, bridge, input_tensor, max_length=args.MAX_LENGTH):

    # Required for tensor matching.
    # Remove to see the results for educational purposes.
    with torch.no_grad():

        # Initialize the encoder hidden.
        input_length = input_tensor.size(0)
        encoder_hidden = encoder.initHidden()

        if args.bidirectional:
            encoder_outputs = torch.zeros(args.batch_size, max_length, 2 * encoder.hidden_size, device=device)
            encoder_hidden_forward = encoder_hidden['forward']
            encoder_hidden_backward = encoder_hidden['backward']

            for ei in range(input_length):
                (encoder_hidden_forward, encoder_hidden_backward) = encoder(
                    (input_tensor[ei],input_tensor[input_length - 1 - ei]), (encoder_hidden_forward,encoder_hidden_backward))

            # Extract the hidden and cell states
            hn_forward, cn_forward = encoder_hidden_forward
            hn_backward, cn_backward = encoder_hidden_backward

            # Concatenate the hidden and cell states for forward and backward paths.
            encoder_hn = torch.cat((hn_forward, hn_backward), 2)
            encoder_cn = torch.cat((cn_forward, cn_backward), 2)


            # Only return the hidden and cell states for the last layer and pass it to the decoder
            encoder_hn_last_layer = encoder_hn[-1].view(1, 1, -1)
            encoder_cn_last_layer = encoder_cn[-1].view(1,1,-1)

            # The list of states
            encoder_hidden_last = [encoder_hn_last_layer, encoder_cn_last_layer]

        else:
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(
                    input_tensor[ei], encoder_hidden)

            # only return the hidden and cell states for the last layer and pass it to the decoder
            hn, cn = encoder_hidden
            encoder_hn_last_layer = hn[-1].view(1,1,-1)
            encoder_cn_last_layer = cn[-1].view(1,1,-1)
            encoder_hidden_last = [encoder_hn_last_layer, encoder_cn_last_layer]

        decoder_input = torch.tensor([SOS_token], device=device)  # SOS
        encoder_hidden_last = [bridge(item) for item in encoder_hidden_last]
        decoder_hidden = encoder_hidden_last

        decoded_words = []
        # decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            # decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        # return decoded_words, decoder_attentions[:di + 1]
        return decoded_words


# In[90]:

######################################################################
def evaluateRandomly(encoder, decoder, bridge, n=50):
    for i in range(n):
        pair = testset[i]['sentence']
        input_tensor, mask_input = reformat_tensor_mask(pair[:,0,:].view(1,1,-1))
        input_tensor = input_tensor[input_tensor != 0]
        output_tensor, mask_output = reformat_tensor_mask(pair[:,1,:].view(1,1,-1))
        output_tensor = output_tensor[output_tensor != 0]
        if device == torch.device("cuda"):
            input_tensor = input_tensor.cuda()
            output_tensor = output_tensor.cuda()

        input_sentence = ' '.join(SentenceFromTensor_(input_lang, input_tensor))
        output_sentence = ' '.join(SentenceFromTensor_(output_lang, output_tensor))
        print('Input: ', input_sentence)
        print('Output: ', output_sentence)
        output_words = evaluate(encoder, decoder, bridge, input_tensor)
        output_sentence = ' '.join(output_words)
        print('Predicted Output: ', output_sentence)
        print('')


# In[91]:

evaluateRandomly(encoder1, decoder1, bridge)


# In[92]:

# one word different
from nltk.translate.bleu_score import sentence_bleu
reference = [['he', 'is', 'actually', 'not', 'the', 'manager']]
candidate = ['he', 'is' , 'not', 'not', 'the', 'manager']
score = sentence_bleu(reference, candidate)
print(score)


# In[102]:

# one word different
from nltk.translate.bleu_score import sentence_bleu
reference = [['he', 's', 'in', 'danger', 'of', 'being', 'evicted']]
candidate = ['he', 's' , 'in', 'in', 'in', 'of', 'of']
score = sentence_bleu(reference, candidate)
print(score)


# In[ ]:



