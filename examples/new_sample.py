import os
import argparse
import logging

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext

import SeqModel
from SeqModel.trainer import SupervisedTrainer
from SeqModel.models import EncoderRNN, DecoderRNN, ItemEncoder, ContextDecoderRNN, NeuralEditorEncoder,Seq2seq
from SeqModel.loss import VAELoss, Perplexity
from SeqModel.optim import Optimizer
from SeqModel.dataset import SourceField, TargetField
from SeqModel.evaluator import Predictor
from SeqModel.util.checkpoint import Checkpoint

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3
torch.autograd.set_detect_anomaly(True)
# Sample usage:
#     # training
#     python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH
#     # resuming from the latest checkpoint of the experiment
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --resume
#      # resuming from a specific checkpoint
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --load_checkpoint $CHECKPOINT_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path', default='/home/yichuan/course/seq2/data/toy_reverse/train/data.txt',
                    help='Path to train data')
parser.add_argument('--dev_path', action='store', dest='dev_path', default='/home/yichuan/course/seq2/data/toy_reverse/dev/data.txt',
                    help='Path to dev data')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')

opt = parser.parse_args()
opt.data_path = "../data/smalldata.txt"
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
else:
    # Prepare dataset
    # src = SourceField()
    tgt = TargetField()
    itemField = torchtext.data.Field(sequential=False, preprocessing=lambda x: str(int(x))+'_i')
    rateField = torchtext.data.Field(sequential=False, preprocessing=lambda x: x +'_r')
    userField = torchtext.data.Field(sequential=False, preprocessing=lambda x: str(int(x))+'_u')


    max_len = 100
    def len_filter(example):
        return len(example.tgt) <= max_len
        # return len(example.src) <= max_len and len(example.tgt) <= max_len
    # train = torchtext.data.TabularDataset(
    #     path=opt.train_path, format='tsv',
    #     fields=[('src', src), ('tgt', tgt)],
    #     filter_pred=len_filter
    # )
    # dev = torchtext.data.TabularDataset(
    #     path=opt.dev_path, format='tsv',
    #     fields=[('src', src), ('tgt', tgt)],
    #     filter_pred=len_filter
    # )

    #
    # data = torchtext.data.TabularDataset(
    #     path = opt.data_path, format='json',
    #     fields={'asin': ('itemId', itemId),
    #          'overall': ('rate', rate),
    #         '':
    #         'reviewText':('tgt', tgt)},
    #     filter_pred = len_filter
    # )

    data = torchtext.data.TabularDataset(
        path=opt.data_path, format='tsv',
        fields=[('userId', userField), ('itemId', itemField), ('rate', rateField), ('tgt', tgt)],
        filter_pred=len_filter
    )

    train, dev = data.split(split_ratio=0.7)
    itemField.build_vocab(data.userId, data.rate, data.itemId)
    rateField.build_vocab(data.userId, data.rate, data.itemId)
    userField.build_vocab(data.userId, data.rate, data.itemId)
    # rate.build_vocab(data.itemId, data.rate)
    tgt.build_vocab(train, max_size=50000)
    input_vocab = itemField.vocab

    output_vocab = tgt.vocab

    # NOTE: If the source field name and the target field name
    # are different from 'src' and 'tgt' respectively, they have
    # to be set explicitly before any training or inference
    # seq2seq.src_field_name = 'src'
    # seq2seq.tgt_field_name = 'tgt'

    # Prepare loss
    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = Perplexity(weight, pad)
    if torch.cuda.is_available():
        loss.cuda()


    seq2seq = None
    optimizer = None
    if not opt.resume:
        # Initialize model
        hidden_size=128
        bidirectional = False
        latent_size = 128
        item_encoder = ItemEncoder(len(input_vocab), hidden_size=hidden_size, predic_rate=True)

        decoder = DecoderRNN(len(tgt.vocab), max_len, hidden_size * 2 if bidirectional else hidden_size,
                             dropout_p=0.2, use_attention=False, bidirectional=bidirectional,
                             eos_id=tgt.eos_id, sos_id=tgt.sos_id)
        # decoder = ContextDecoderRNN(
        #     len(tgt.vocab), max_len, hidden_size * 2 if bidirectional else hidden_size,
        #                      dropout_p=0.2, use_attention=False, bidirectional=bidirectional,
        #                      eos_id=tgt.eos_id, sos_id=tgt.sos_id, use_gC2S=True)

        seq2seq = Seq2seq(item_encoder, decoder)
        if torch.cuda.is_available():
            seq2seq.cuda()

        for param in seq2seq.parameters():
            if param.requires_grad:
                param.data.uniform_(-0.08, 0.08)

        # Optimizer and learning rate scheduler can be customized by
        # explicitly constructing the objects and pass to the trainer.
        #
        # optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=5)
        # scheduler = StepLR(optimizer.optimizer, 1)
        # optimizer.set_scheduler(scheduler)

    # train
    t = SupervisedTrainer(loss=loss, batch_size=512,
                             checkpoint_every=50,
                             print_every=100, expt_dir=opt.expt_dir, predic_rate=True)

    seq2seq = t.train(seq2seq, train,
                      num_epochs=20, dev_data=dev,
                      optimizer=optimizer,
                      teacher_forcing_ratio=0.5,
                      resume=opt.resume)

predictor = Predictor(seq2seq, input_vocab, output_vocab)

while True:
    seq_str = raw_input("Type in a source sequence:")
    seq = seq_str.strip().split()
    print(predictor.predict(seq))
