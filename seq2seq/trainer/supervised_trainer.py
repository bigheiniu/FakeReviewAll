from __future__ import division
import logging
import os
import random
import time

import torch
import torchtext
from torch import optim

import seq2seq
from seq2seq.evaluator import Evaluator
from seq2seq.loss import NLLLoss
from seq2seq.optim import Optimizer
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.loss import cal_bleu_score

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
class SupervisedTrainer(object):
    """ The SupervisedTrainer class helps in setting up a training framework in a
    supervised setting.
    Args:
        expt_dir (optional, str): experiment Directory to store details of the experiment,
            by default it makes a folder in the current directory to store the details (default: `experiment`).
        loss (seq2seq.loss.loss.Loss, optional): loss for training, (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for experiment, (default: 64)
        checkpoint_every (int, optional): number of batches to checkpoint after, (default: 100)
    """
    def __init__(self, expt_dir='experiment', loss=NLLLoss(), batch_size=64,
                 random_seed=None,
                 checkpoint_every=100, print_every=100, predic_rate=False):
        self._trainer = "Simple Trainer"
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
        self.loss = loss
        self.evaluator = Evaluator(loss=self.loss, batch_size=batch_size)
        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)
        self.batch_size = batch_size

        self.logger = logging.getLogger(__name__)
        self.predic_rate = predic_rate
        self.rate_loss = torch.nn.NLLLoss().cuda()

    def _train_batch(self, input_variable, input_lengths, target_variable, model, teacher_forcing_ratio):
        word_loss = self.loss
        # Forward propagation
        (decoder_outputs, decoder_hidden, other), rate_predic = model(input_variable, target_variable,
                                                       teacher_forcing_ratio=teacher_forcing_ratio)

        # Get loss
        word_loss.reset()
        for step, step_output in enumerate(decoder_outputs):
            batch_size = target_variable.size(0)
            word_loss.eval_batch(step_output.contiguous().view(batch_size, -1), target_variable[:, step + 1])
        # Backward propagation
        th = input_variable[2] - 1
        th1 = th.unsqueeze(-1)
        if self.predic_rate:
            rate_loss = self.rate_loss(rate_predic, th)
            t1 = word_loss.acc_loss
            all_loss = rate_loss + word_loss.acc_loss
        else:
            all_loss = word_loss.acc_loss
        model.zero_grad()
        all_loss.backward()
        self.optimizer.step()
        # Get the bleu score
        pre_seq = torch.stack(other['sequence'], dim=1).cpu().numpy().tolist()
        gold_seq = target_variable.cpu().numpy().tolist()
        return word_loss.get_loss(), pre_seq, gold_seq

    def _train_epoches(self, data, model, n_epochs, start_epoch, start_step,
                       dev_data=None, teacher_forcing_ratio=0):
        log = self.logger

        print_loss_total = 0  # Reset every print_every
        epoch_loss_total = 0  # Reset every epoch


        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            device=device, repeat=False)

        steps_per_epoch = len(batch_iterator)
        total_steps = steps_per_epoch * n_epochs

        step = start_step
        step_elapsed = 0
        for epoch in range(start_epoch, n_epochs + 1):
            log.debug("Epoch: %d, Step: %d" % (epoch, step))

            batch_generator = batch_iterator.__iter__()
            # consuming seen batches from previous training
            for _ in range((epoch - 1) * steps_per_epoch, step):
                next(batch_generator)

            model.train(True)
            pred_list = []
            gold_list = []
            for batch in batch_generator:
                step += 1
                step_elapsed += 1

                input_rate = getattr(batch, seq2seq.src_field_rate)
                input_item_id = getattr(batch, seq2seq.src_field_itemId)
                input_user_id = getattr(batch, seq2seq.src_field_userId)
                target_variables = getattr(batch, seq2seq.tgt_field_name)
                input_variables = [input_user_id, input_item_id, input_rate]

                loss, pred_seq, gold_seq = self._train_batch(input_variables, None, target_variables, model, teacher_forcing_ratio)
                pred_list.append(pred_seq)
                gold_list.append(gold_seq)
                # Record average loss
                print_loss_total += loss
                epoch_loss_total += loss

                if step % self.print_every == 0 and step_elapsed > self.print_every:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    log_msg = 'Progress: %d%%, Train %s: %.4f' % (
                        step / total_steps * 100,
                        self.loss.name,
                        print_loss_avg)
                    log.info(log_msg)

                # # Checkpoint
                # if step % self.checkpoint_every == 0 or step == total_steps:
                #     Checkpoint(model=model,
                #                optimizer=self.optimizer,
                #                epoch=epoch, step=step,
                #                input_vocab=data.fields[seq2seq.src_field_name].vocab,
                #                output_vocab=data.fields[seq2seq.tgt_field_name].vocab).save(self.expt_dir)

            if step_elapsed == 0: continue

            bleu_score = cal_bleu_score(pred_list, gold_list)
            epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
            epoch_loss_total = 0
            log_msg = "Finished epoch %d: Train %s: %.4f, bleu_score: %.4f" % (epoch, self.loss.name, epoch_loss_avg, bleu_score)
            if dev_data is not None:
                dev_loss, accuracy, bleu_score = self.evaluator.evaluate(model, dev_data)
                self.optimizer.update(dev_loss, epoch)
                log_msg += ", Dev %s: %.4f, Accuracy: %.4f, bleu_score: %.4f" % (self.loss.name, dev_loss, accuracy, bleu_score)
                model.train(mode=True)
            else:
                self.optimizer.update(epoch_loss_avg, epoch)

            log.info(log_msg)

    def train(self, model, data, num_epochs=5,
              resume=False, dev_data=None,
              optimizer=None, teacher_forcing_ratio=0):
        """ Run training for a given model.
        Args:
            model (seq2seq.models): model to run training on, if `resume=True`, it would be
               overwritten by the model loaded from the latest checkpoint.
            data (seq2seq.dataset.dataset.Dataset): dataset object to train on
            num_epochs (int, optional): number of epochs to run (default 5)
            resume(bool, optional): resume training with the latest checkpoint, (default False)
            dev_data (seq2seq.dataset.dataset.Dataset, optional): dev Dataset (default None)
            optimizer (seq2seq.optim.Optimizer, optional): optimizer for training
               (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))
            teacher_forcing_ratio (float, optional): teaching forcing ratio (default 0)
        Returns:
            model (seq2seq.models): trained model.
        """
        # If training is set to resume
        if resume:
            latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
            resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
            model = resume_checkpoint.model
            self.optimizer = resume_checkpoint.optimizer

            # A walk around to set optimizing parameters properly
            resume_optim = self.optimizer.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop('params', None)
            defaults.pop('initial_lr', None)
            self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step
        else:
            start_epoch = 1
            step = 0
            if optimizer is None:
                optimizer = Optimizer(optim.Adam(model.parameters()), max_grad_norm=5)
            self.optimizer = optimizer

        self.logger.info("Optimizer: %s, Scheduler: %s" % (self.optimizer.optimizer, self.optimizer.scheduler))

        self._train_epoches(data, model, num_epochs,
                            start_epoch, step, dev_data=dev_data,
                            teacher_forcing_ratio=teacher_forcing_ratio)
        return model