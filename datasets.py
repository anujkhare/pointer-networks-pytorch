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
            'points': datum[0].astype(np.float32),
            'inds_hull': datum[1].astype(np.long),  # FIXME: 1-based
        }

    def __len__(self) -> int:
        return len(self.data)