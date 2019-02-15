from typing import List, Dict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import collections
import matplotlib.pyplot as plt
import numpy as np
import os
import torch


EOL_TOKEN = [-1, -1]


def get_lines_from_pointers(points, pointers, return_labels: bool = False) -> List[np.ndarray]:
    """THIS IS THE POST-PROCESSING TO GO FROM OUTPUT TO LINES!"""
    lines = []
    cur_line = []
    
    labels, cur_labels = [], []
    for pointer in pointers:
        point = points[pointer]
        if np.all(point == EOL_TOKEN):
            if len(cur_line) > 0:
                lines.append(np.array(cur_line))
                labels.append(cur_labels)
            cur_line, cur_labels = [], []
            continue

        cur_line.append(point)
        cur_labels.append(pointer)

    if len(cur_line) > 0:
        lines.append(np.array(cur_line))
        labels.append(cur_labels)

    if return_labels:
        return lines, labels
    return lines

def plot_points_and_lines(points, pointers) -> None:
    plt.xlim(0, 1.2)
    plt.ylim(0, 1.2)
    # Plot points
    points_to_plot = points[points[:, 0] != -1]
    plt.scatter(points_to_plot[:, 0], points_to_plot[:, 1], marker='x')
    
    # Mark the numbers
    for pointer in range(len(points)):
        point = points[pointer]
        if point[0] == -1:
            continue
        plt.text(point[0], point[1] + 0.025, str(pointer), ha='center', color='r')

    # Plot lines
    lines = get_lines_from_pointers(points, pointers)

    for line in lines:
        plt.plot(line[:, 0], line[:, 1], '--')

        
# Dataset related
class LineDataset:
    def __init__(self, data, random_shuffle=False):
        self.data = data
        self.random_shuffle = random_shuffle
    
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, ix) -> Dict[str, np.ndarray]:
        datum = self.data[ix]
        
        if self.random_shuffle:
            datum = self.random_shuffle_sequence(datum)
        
        return datum

    @staticmethod
    def random_shuffle_sequence(datum):
        """
        Randomly shuffles the input sequence and the other arrays correspondingly.
        """
        sequence, pointers = datum['sequence'], datum['pointers']
        n = len(sequence)
        
        # Generate new 
        inds_new_order = np.arange(n)
        np.random.shuffle(inds_new_order)
        sequence = sequence[inds_new_order].squeeze()
        
        # map the pointers to the new indices
        inds_reverse = np.zeros(n)
        inds_reverse[inds_new_order] = np.arange(n)
        new_pointers = inds_reverse[pointers].astype(np.long)
        assert np.all(new_pointers.shape == pointers.shape)
        
        return {
            'sequence': sequence,
            'pointers': new_pointers,
        }

# Data loader specific
def get_padded_tensor_and_lens(list_seqs, pad_constant_value=0, n_dim=2):
    lens = np.array([len(x) for x in list_seqs])
    # Each sequence is an array of shape seq_len*n_dim
    for ix in range(len(list_seqs)):
        seq = list_seqs[ix]
        if len(seq) == 0 or len(seq[0]) == 0:
            list_seqs[ix] = np.zeros(n_dim, dtype=np.float32)[np.newaxis, :]
        seq = list_seqs[ix]
        assert len(seq.shape) == 2, 'Actual shape is: {}'.format(seq.shape)
        assert seq.shape[1] == n_dim

    max_len = max(lens)
    data = np.array([
        np.pad(seq, pad_width=[(0, max_len - len(seq)), (0, 0)], mode='constant', constant_values=pad_constant_value)
        for seq in list_seqs
    ])

    return data, lens


def collate_fn(batch):
    sequences, lens1 = get_padded_tensor_and_lens([sample['sequence'] for sample in batch], pad_constant_value=0, n_dim=2)
    pointers, lens2 = get_padded_tensor_and_lens([sample['pointers'][..., np.newaxis] for sample in batch], pad_constant_value=-100, n_dim=1)
    
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