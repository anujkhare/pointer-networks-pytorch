import numpy as np


class ConvexHullDataset:
    def __init__(self, data: np.ndarray) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError
        if not data.shape[1] == 2:
            raise ValueError('Must have 2 columns: The coordinates of points, the indices of the convex hull')
        
        self.data = data
    
    def __getitem__(self, ix):
        datum = self.data[ix]
        return {
            'sequence': datum[0].astype(np.float32),
            'pointers': datum[1].astype(np.long).reshape(-1, 1),  # FIXME: 1-based
        }

    def __len__(self) -> int:
        return len(self.data)

    
    
# Data loader specific
import torch
def get_padded_tensor_and_lens(list_seqs, pad_constant_value=0):
    # NOTE: DON'T SORT HERE! YOU'LL LOSE THE CORRESPONDANCE B/W SENTENCE PAIRS AND THE LABELS
    # Each sequence is an array of shape seq_len*n_dim
    for seq in list_seqs:
        assert len(seq.shape) == 2, 'Actual shape is: {}'.format(seq.shape)
    lens = np.array([len(x) for x in list_seqs])

    max_len = max(lens)
    data = np.array([
        np.pad(seq, pad_width=[(0, max_len - len(seq)), (0, 0)], mode='constant', constant_values=pad_constant_value)
        for seq in list_seqs
    ])

    return data, lens


def collate_fn(batch):
    sequences, lens1 = get_padded_tensor_and_lens([sample['sequence'] for sample in batch], pad_constant_value=0)
    pointers, lens2 = get_padded_tensor_and_lens([sample['pointers'] for sample in batch], pad_constant_value=-1)
    
    # Sort such that the longest sequence is first. Sort the pointers to match the sequences.
    inds_sorted_desc = np.argsort(lens1)[::-1]
    sequences, lens1 = sequences[inds_sorted_desc, ...], lens1[inds_sorted_desc]
    pointers, lens2 = pointers[inds_sorted_desc, ...], lens2[inds_sorted_desc]
    
    sequences = torch.from_numpy(sequences)
    pointers = torch.from_numpy(pointers)
    
    return {
        'sequence': sequences,
        'sequence_lens': lens1,
        'pointers': pointers,
        'pointer_lens': lens2,
    }