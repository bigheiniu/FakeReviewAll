import torch
import torch.nn as nn

from fakeGen.Discriminator import Discriminator, RNNclaissfier
from fakeGen.seq2seq import Seq2seq
from fakeGen.Generator import Generator
from fakeGen.evaluate import f1_score, accuracy_score, tensor2list

from SeqModel.trainer import SupervisedTrainer
from SeqModel.models import EncoderRNN, DecoderRNN, ItemEncoder, ContextDecoderRNN
from SeqModel.loss import VAELoss, Perplexity
from SeqModel.optim import Optimizer
from SeqModel.dataset import SourceField, TargetField
from SeqModel.evaluator import Predictor
from SeqModel.util.checkpoint import Checkpoint
import torchtext
import sys
import torch.optim as optim
import argparse
import itertools





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

    return all_data, label


def train_LM(opt, loss, seq2seq, fake_data, optimizer=None):
    # train


    train_lm_data, test_lm_data = fake_data.split(split_ratio=0.2)
    t = SupervisedTrainer(loss=loss, batch_size=opt.batch_size,
                             checkpoint_every=opt.check_point,
                             print_every=opt.print_every, expt_dir=opt.expt_dir, predic_rate=True)

    seq2seq = t.train(seq2seq, train_lm_data, dev_data=test_lm_data,
                      num_epochs=opt.lm_epochs,
                      optimizer=optimizer,
                      teacher_forcing_ratio=opt.teach_force_ratio,
                      resume=opt.resume)
    return seq2seq


def pre_train_deceptive(rnn_claissfier, classifier_opt, data, opt):
    rnn_claissfier.train()
    for epochs in range(opt.pre_clf_epochs):
        total_loss = 0
        data_iter = data.__iter__()
        for batch in data_iter:
            feature, input_length = getattr(batch, 'src')
            label = getattr(batch, 'label')
            classifier_opt.zero_grad()
            loss, _ = rnn_claissfier(feature, label)
            loss.backward()
            classifier_opt.step()

            total_loss += loss.item()
        print('[INFO] ---PRE-TRAIN--- clf loss is {}'.format(total_loss))
    return rnn_claissfier

def train_discriminator(discriminator, dis_opt, seq2seq, gen, fake_data, opt):
    discriminator.train()
    seq2seq.eval()
    for epoch in range(opt.dis_epoch):
        print('epoch %d : ' % (epoch + 1), end='')
        sys.stdout.flush()
        total_loss = 0
        # clf the simulate text and fake text
        fake_iter = fake_data.__iter__()
        true_list = []
        pre_list = []
        discriminator.train(True)
        for batch in fake_iter:
            feature = getattr(batch, 'src')
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
            loss, out = discriminator.batchBCELoss(all_data, label)
            loss.backward()
            dis_opt.step()
            true_list.append(tensor2list(out['y_true']))
            pre_list.append(tensor2list(out['y_pre']))
            total_loss += loss.item()

        y_true = list(itertools.chain.from_iterable(true_list))
        y_pre = list(itertools.chain.from_iterable(pre_list))

        f1 = f1_score(y_true, y_pre)
        acc = accuracy_score(y_true, y_pre)

        print("[INFO] ---TRAIN---- discriminator loss {}, acc {}, f1 {}".format(total_loss, f1, acc))
        # train the classifier


    # return discriminator
    #TODO: print total loss

def train_classifier(opt, real_data, gen, seq2seq, rnn_classifier, rnn_opt):
    rnn_classifier.train()
    seq2seq.eval()
    gen.train()
    real_iter = real_data.__iter__()
    total_loss = 0
    true_list = []
    pre_list = []
    for batch in real_iter:
        feature = getattr(batch, 'src')
        label = getattr(batch, 'label')
        shape = torch.size((opt.BATCH_SIZE, opt.z_hidden_size))
        if next(rnn_classifier.parameters()).is_cuda:
            z = torch.cuda.FloatTensor(shape)
        else:
            z = torch.FloatTensor(shape)

            # sim_seq: distribution of words
        sim_hidden = gen(z)
        sim_seq = seq2seq.decoder_hidden(sim_hidden)
        sim_label = torch.zeros_like(label)
        rnn_opt.zero_grad()
        loss, out = rnn_classifier(feature, label, sim_seq, sim_label)
        loss.backward()
        rnn_opt.step()
        total_loss += loss.item()
        true_list.append(tensor2list(out['y_true']))
        pre_list.append(tensor2list(out['y_pre']))

    y_true = list(itertools.chain.from_iterable(true_list))
    y_pre = list(itertools.chain.from_iterable(pre_list))

    f1 = f1_score(y_true, y_pre)
    acc = accuracy_score(y_true, y_pre)

    print("[INFO] ---TRAINING--- clf loss {}, f1 {}, acc {}".format(total_loss, f1, acc))




def clf_test(test_clf_data, rnn_classifier):
    rnn_classifier.eval()
    pre_list = []
    true_list = []
    data_iter = test_clf_data.__iter__()
    for batch in data_iter:
        feature = getattr(batch, 'src')
        label = getattr(batch, 'label')
        # feature and label will be shuffle in clf
        loss, out = rnn_classifier(feature, label)
        y_pre = tensor2list(out['y_pre'])
        y_true = tensor2list(out['target'])
        pre_list.append(y_pre)
        true_list.append(y_true)
    y_pre = list(itertools.chain.from_iterable(pre_list))
    y_true = list(itertools.chain.from_iterable(true_list))
    f1 = f1_score(y_true, y_pre)
    acc = accuracy_score(y_true, y_pre)
    print("[INFO] ---TEST--- acc is {}, f1 is {}".format(acc, f1))
        # print the loss and F1 score






def train_gen(gen, gen_opt, dis_simulate, opt):
    for epoch in range(opt.gen_epoch):
        for _ in opt.batch_count:
            shape = torch.size((opt.BATCH_SIZE, opt.z_hidden_size))
            if next(gen.parameters()).is_cuda:
                z = torch.cuda.FloatTensor(shape)
            else:
                z = torch.FloatTensor(shape)

            torch.randn(shape, out=z)
            # generate fake review
            sim_data = gen(z)
            gen_opt.zero_grad()
            # fool the discriminator
            # seqGAN will use the prediction logit as reward
            loss = dis_simulate(sim_data, torch.ones((sim_data.shape[0], 1), device=opt.device, dtype=torch.long))
            loss.backward()
            gen_opt.step()


def prepare_data(opt):
    # seq-label real dataset
    # seq-label all dataset
    # seq-label fake dataset
    tgt = TargetField()
    src = SourceField()
    label = torchtext.data.Field(sequential=False)
    fake_data_lm = torchtext.data.TabularDataset(
        path=opt.fake_data_path, format='csv',
        fields=[('src', src), ('label', label), ('tgt', tgt)]
    )

    real_data_clf = torchtext.data.TabularDataset(
        path=opt.real_data_path, format='csv',
        fields=[('src', src), ('label', label)]
    )

    train_clf = torchtext.data.TabularDataset(
        path=opt.train_data_path, format='csv',
        fields=[('src', src), ('label', label)]
    )


    test_clf = torchtext.data.TabularDataset(
        path=opt.test_data_path, format='csv',
        fields=[('src', src), ('label', label)]
    )


    src.build_vocab(train_clf.src, test_clf.src, max_size=opt.max_word)
    tgt.build_vocab(train_clf.src, test_clf.src, max_size=opt.max_word)
    label.build_vocab(train_clf)
    input_vocab = src.vocab

    output_vocab = tgt.vocab

    test_clf = torchtext.data.BucketIterator(
        dataset=test_clf, batch_size=opt.batch_size,
        sort=False, sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=opt.device, repeat=False)

    train_clf = torchtext.data.BucketIterator(
        dataset=train_clf, batch_size=opt.batch_size,
        sort=False, sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=opt.device, repeat=False)

    real_data_clf = torchtext.data.BucketIterator(
        dataset=real_data_clf, batch_size=opt.batch_size,
        sort=False, sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=opt.device, repeat=False)

    return fake_data_lm, real_data_clf, train_clf, test_clf, input_vocab, tgt


def prepare_loss(tgt, opt):
    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = Perplexity(weight, pad)
    if opt.cuda:
        loss.cuda()
    return loss

def prepare_model(opt, vocab_size, tgt):

    # Prepare loss
    encoder = EncoderRNN(vocab_size, opt.max_len, opt.hidden_size,
                         bidirectional=opt.bidirectional, variable_lengths=True).to(opt.device)
    decoder = DecoderRNN(vocab_size, opt.max_len, opt.hidden_size* 2 if opt.bidirectional else opt.hidden_size,
                         dropout_p=opt.dropout, use_attention=True, bidirectional=opt.bidirectional,
                         eos_id=tgt.eos_id, sos_id=tgt.sos_id).to(opt.device)
    seq2seq = Seq2seq(encoder, decoder).to(opt.device)


    gen = Generator(opt.hidden_size, opt.z_size).to(opt.device)
    encoder_new = EncoderRNN(vocab_size, opt.max_len, opt.hidden_size,
                         bidirectional=opt.bidirectional, variable_lengths=True).to(opt.device)
    dis_clf = RNNclaissfier(encoder_new, opt.clf_layers).to(opt.device)
    dis_gen = Discriminator(opt.hidden_size, opt.clf_layers).to(opt.device)
    opt_gen = optim.Adam(gen.parameters(), lr=opt.gen_lr)
    opt_dis_clf = optim.Adam(dis_clf.parameters(), lr=opt.dis_dec_lr)
    opt_dis_gen = optim.Adam(dis_gen.parameters(), lr=opt.dis_gen_lr)

    return seq2seq, gen, opt_gen, dis_clf, opt_dis_clf, dis_gen, opt_dis_gen

def build_parser():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('-fake_data_path', type=str,
                        default='/home/yichuan/course/seq2/data/YelpNYC/fake_data.csv')
    parser.add_argument('-real_data_path', type=str,
                        default='/home/yichuan/course/seq2/data/YelpNYC/real_data.csv')
    parser.add_argument('-train_data_path', type=str,
                        default='/home/yichuan/course/seq2/data/YelpNYC/train_data.csv')
    parser.add_argument('-test_data_path', type=str,
                        default='/home/yichuan/course/seq2/data/YelpNYC/test_data.csv')

    # language model
    parser.add_argument('-max_len', type=int, default=200)
    parser.add_argument('-bidirectional', action='store_false', default=True)
    parser.add_argument('-dropout', type=float, default=0.3)
    parser.add_argument('-hidden_size', type=int, default=128)
    parser.add_argument('-max_word', type=int, default=30000)
    # seq2seq model
    parser.add_argument('-batch_size', type=int, default=20)
    parser.add_argument('-check_point', type=int, default=10)
    parser.add_argument('-print_every', type=int, default=100)
    parser.add_argument('-expt_dir', type=str, default='./experiment')
    parser.add_argument('-teach_force_ratio', type=float, default=0.4)
    parser.add_argument('-resume', action='store_true', default=False)



    # GAN discriminator
    parser.add_argument('-clf_layers', type=int, default=3)
    parser.add_argument('-z_size', type=int, default=128)

    # learning rate
    parser.add_argument('-dis_dec_lr', type=float, default=0.05)
    parser.add_argument('-dis_gen_lr', type=float, default=0.005)
    parser.add_argument('-gen_lr', type=float, default=0.005)

    # epochs
    parser.add_argument('-gan_epoch', type=int, default=10)
    parser.add_argument('-clf_epoch', type=int, default=10)
    parser.add_argument('-dis_epoch', type=int, default=1)
    parser.add_argument('-pre_clf_epochs', type=int, default=10)
    parser.add_argument('-lm_epochs', type=int, default=20)

    # cuda
    parser.add_argument('-cuda',  action='store_false')
    return parser







def main(parser):
    opt = parser.parse_args()
    opt.device = torch.device('cuda') if opt.cuda else torch.device('cpu')
    fake_data_lm, real_data_clf, train_clf, test_clf, vocab, tgt = prepare_data(opt)
    seq2seq, gen, opt_gen, rnn_claissfier, classifier_opt, dis_gen, opt_dis_gen = \
        prepare_model(opt, len(vocab), tgt=tgt)
    
    # pre-train the LM model
    loss_seq = prepare_loss(tgt, opt)
    seq2seq = train_LM(opt, loss_seq, seq2seq, fake_data_lm)

    # pre-train the classify model
    pre_train_deceptive(rnn_claissfier, classifier_opt, train_clf, opt)

    # train the generator
    for epoch in range(opt.gan_epoch):
        train_gen(gen, opt_gen, dis_gen, opt)

    # train the discriminator
        train_discriminator(dis_gen, opt_dis_gen, seq2seq, gen, fake_data_lm, opt)

    # train the classification on simulate data and real review
    for epoch in range(opt.clf_epoch):
        # test the classifier
        clf_test(test_clf, rnn_claissfier)
        train_classifier(opt, real_data_clf, gen, seq2seq, rnn_claissfier, classifier_opt)





if __name__ == '__main__':
    parser = build_parser()
    main(parser)








