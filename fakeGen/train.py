import torch
import torch.nn as nn

from fakeGen.Discriminator import Discriminator
from fakeGen.seq2seq import Seq2seq
from fakeGen.Generator import Generator
from tqdm import tqdm
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, ItemEncoder, ContextDecoderRNN
from seq2seq.loss import VAELoss, Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint
import torchtext
import sys
import torch.optim as optim


def helper_fusion(pos_data, neg_data):
    pos_batch = pos_data.shape[0]
    neg_batch = neg_data.shape[1]
    label_pos = torch.ones((pos_batch, 1), device=pos_data.device, dtype=torch.long)
    label_neg = torch.ones((neg_batch, 1), device=pos_data.device, dtype=torch.long)
    all_data = torch.cat((pos_batch, neg_batch), dim=0)
    label = torch.cat((label_pos, label_neg), dim=0)

    perm = torch.randperm(label.size()[0])
    all_data = all_data[perm]
    label = label[perm]

    # neg = torch.cat((neg_data, label_pos), dim=-1)
    # all_data = torch.cat((pos, neg), dim=0)
    return all_data, label


def train_LM(seq2seq, train_fake_data, dev_fake_data, opt, loss, optimizer):
    # train

    t = SupervisedTrainer(loss=loss, batch_size=opt.batch_size,
                             checkpoint_every=opt.check_point,
                             print_every=opt.print_every, expt_dir=opt.expt_dir, predic_rate=True)

    seq2seq = t.train(seq2seq, train_fake_data, dev_data=dev_fake_data,
                      num_epochs=opt.epochs,
                      optimizer=optimizer,
                      teacher_forcing_ratio=opt.teach_force_ratio,
                      resume=opt.resume)

    return seq2seq


def pre_train_deceptive(rnn_claissfier, classifier_opt, data, opt):
    epochs = opt.epochs
    for epochs in range(epochs):
        total_loss = 0
        for batch in data:
            feature, label = map(lambda x: x.to(opt.device), batch)
            classifier_opt.zero_grad()
            loss = rnn_claissfier(feature, label)
            loss.backward()
            classifier_opt.step()

            total_loss += loss.item()
    return rnn_claissfier

def train_discriminator(discriminator, dis_opt, seq2seq, rnn_classifier, rnn_opt, gen, pos_data, opt):
    epochs = opt.epochs
    for epoch in range(epochs):
        print('epoch %d : ' % (epoch + 1), end='')
        sys.stdout.flush()
        total_loss = 0
        # clf the simulate text and fake text
        for batch in range(0, opt.POS_NEG_SAMPLES, opt.BATCH_SIZE):
            feature, label = map(lambda x: x.to(opt.device), batch)
            hidden = seq2seq.encoder_seq(feature)
            shape = torch.size((opt.BATCH_SIZE, opt.z_hidden_size))
            if next(discriminator.parameters()).is_cuda:
                z = torch.cuda.FloatTensor(shape)
            else:
                z = torch.FloatTensor(shape)

            torch.randn(shape, out=z)
            # classify the hidden state
            sim_data = gen(z)
            all_data, label = helper_fusion(hidden, sim_data)
            dis_opt.zero_grad()
            loss = discriminator(all_data, label)
            loss.backward()
            dis_opt.step()

            total_loss += loss.item()

        # train the classifier
        for batch in range(0, 10, 1):
            feature, label = map(lambda x: x.to(opt.device), batch)
            shape = torch.size((opt.BATCH_SIZE, opt.z_hidden_size))
            if next(discriminator.parameters()).is_cuda:
                z = torch.cuda.FloatTensor(shape)
            else:
                z = torch.FloatTensor(shape)

            # sim_seq: distribution of words
            sim_seq = seq2seq.decoder_hidden(z)
            sim_label = torch.zeros_like(label)
            loss = rnn_classifier(feature, label, sim_seq, sim_label)
            total_loss += loss.item()

    # return discriminator
    #TODO: print total loss


def train_gen(gen, gen_opt, dis_simulate, fake_data, opt, epochs):
    for epoch in range(epochs):
        for batch in opt.batch_count:
            shape = torch.size((opt.BATCH_SIZE, opt.z_hidden_size))
            if next(gen.parameters()).is_cuda:
                z = torch.cuda.FloatTensor(shape)
            else:
                z = torch.FloatTensor(shape)

            torch.randn(shape, out=z)
            # generate fake review
            sim_data = gen(z)
            gen_opt.zero_grad()
            loss = dis_simulate(sim_data, torch.zeros((sim_data.shape[0], 1), device=opt.device, dtype=torch.long))
            loss.backward()


def prepare_data(opt):

    # seq-seq torch text
    # seq-label simulate dataset
    # seq-label real dataset
    # seq-label all dataset
    tgt = TargetField()
    src = SourceField()
    label = torchtext.data.Field(sequential=False)
    fake_data_lm = torchtext.data.TabularDataset(
        path=opt.fake_data_path, format='tsv',
        fields=[('src', src), ('tgt', tgt), ('label', label)]
    )


    real_data_clf = torchtext.data.TabularDataset(
        path=opt.real_data_path, format='tsv',
        fields=[('src', src), ('label', label)]
    )

    all_data_clf = torchtext.data.TabularDataset(
        path=opt.real_data_path, format='tsv',
        fields=[('src', src), ('label', label)]
    )
    train_clf, test_clf = all_data_clf.split()
    src.build_vocab(all_data_clf.src, max_size=opt.max_word)
    tgt.vocab = src.vocab
    input_vocab = src.vocab
    output_vocab = tgt.vocab

    return fake_data_lm, real_data_clf, train_clf, test_clf, input_vocab, output_vocab


def prepare_loss(tgt, opt):
    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = Perplexity(weight, pad)
    if opt.is_cuda:
        loss.cuda()
    return loss

def prepare_model(opt, vocab_size, tgt):

    # Prepare loss
    encoder = EncoderRNN(vocab_size, opt.max_len, opt.hidden_size,
                         bidirectional=opt.bidirectional, variable_lengths=True)
    decoder = DecoderRNN(vocab_size, opt.max_len, opt.hidden_size* 2 if opt.bidirectional else opt.hidden_size,
                         dropout_p=opt.dropout, use_attention=True, bidirectional=opt.bidirectional,
                         eos_id=tgt.eos_id, sos_id=tgt.sos_id)
    seq2seq = Seq2seq(encoder, decoder)


    gen = Generator(opt.hidden_size, opt.z_size)
    dis_dec = Discriminator(opt.hidden_size, opt.clf_layers)
    dis_gen = Discriminator(opt.hidden_size, opt.clf_layers)
    opt_gen = optim.Adam(gen.parameters(), lr=opt.gen_lr)
    opt_dis_dec = optim.Adam(dis_dec.parameters(), lr=opt.dis_dec_lr)
    opt_dis_gen = optim.Adam(dis_gen.parameters(), lr=opt.dis_gen_lr)

    return seq2seq, gen, opt_gen, dis_dec, opt_dis_dec, dis_gen, opt_dis_gen







