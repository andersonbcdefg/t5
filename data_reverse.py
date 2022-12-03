import torch
import torch.nn.functional as F

class ReverseDataset(torch.utils.data.Dataset):
    def __init__(self, seq_len, max_val, n_samples):
        self.seq_len = seq_len
        self.max_val = max_val
        self.n_samples = n_samples
        self.data = torch.randint(1, max_val, (n_samples, seq_len))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], F.pad(self.data[idx].flip(-1), (1, 0), value=0) # shift right