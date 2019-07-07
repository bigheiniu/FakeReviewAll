from __future__ import print_function, division

import torch
import torchtext

import SeqModel
from SeqModel.loss import NLLLoss
from SeqModel.loss import cal_mt_score

class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss=NLLLoss(), batch_size=64):
        self.loss = loss
        self.batch_size = batch_size

    def evaluate(self, model, data):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """
        model.eval()

        loss = self.loss
        loss.reset()
        match = 0
        total = 0

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=False,
            device=device, train=False)
        tgt_vocab = data.fields[SeqModel.tgt_field_name].vocab
        pad = tgt_vocab.stoi[data.fields[SeqModel.tgt_field_name].pad_token]
        pred_list = []
        gold_list = []
        with torch.no_grad():
            for batch in batch_iterator:
                input_rate = getattr(batch, SeqModel.src_field_rate)
                input_item_id = getattr(batch, SeqModel.src_field_itemId)
                input_user_id = getattr(batch, SeqModel.src_field_userId)
                target_variables = getattr(batch, SeqModel.tgt_field_name)
                input_variables = [input_user_id, input_item_id, input_rate]
                # input_lengths.to(device)
                (decoder_outputs, decoder_hidden, other), rate_predic = model(input_variables, target_variables)
                # Evaluation
                seqlist = other['sequence']
                pred_list.append(torch.stack(seqlist, dim=1).cpu().numpy().tolist())
                gold_list.append(target_variables.cpu().numpy().tolist())
                for step, step_output in enumerate(decoder_outputs):
                    target = target_variables[:, step + 1]
                    loss.eval_batch(step_output.view(target_variables.size(0), -1), target)

                    non_padding = target.ne(pad)
                    correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().item()
                    match += correct
                    total += non_padding.sum().item()
        metric_result = cal_mt_score(pred_list, gold_list)
        if total == 0:
            accuracy = float('nan')
        else:
            accuracy = match / total

        return loss.get_loss(), accuracy, metric_result
