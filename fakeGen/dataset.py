import torch.utils.data.dataset as dataset
import torch




# also I should pad the sequence
class SeqLabel(dataset):
    def __init__(self, seq_ids, labels):
        super(SeqLabel, self).__init__()
        self.seq_ids = seq_ids
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.seq_ids[item], self.labels[item]