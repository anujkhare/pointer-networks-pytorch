from typing import List, Dict
import numpy as np


EOL_TOKEN = [-1, -1]

def generate_random_sample(n_lines_min=1, n_lines_max=4, n_points_min=1, n_points_max=10) -> List[np.ndarray]:
    """
    Returns a list of "lines".
    
    Each "line" is an np.ndarray containing between 1-10 2-D points on the same y-coordinate.
    """
    assert 0 < n_lines_min <= n_lines_max <= 10
    assert 0 < n_points_min <= n_points_max <= 10

    n_lines = np.random.choice(np.arange(n_lines_min, n_lines_max + 1))

    # all the lines will be sampled at the discrete intervals - 0.1, 0.2 .... 1.0
    line_pos = np.random.choice(np.arange(10), n_lines, replace=False) + 1
    line_pos = sorted(line_pos, reverse=True)  # NOTE: THIS IS IMPORTANT! ORDER MATTERS: ALWAYS GIVE THE LINES TOP to BOTTOM

    coords = []
    for ix in range(n_lines):
        n_points = np.random.choice(np.arange(n_points_min, n_points_max + 1))
        x = sorted(np.random.random(n_points))
        y = [line_pos[ix] / 10] * n_points
#         print(x, y)
        points = np.array(list(zip(x, y))).astype(np.float)
        coords.append(points)

    return coords

def generate_points_pointers(n_lines_min=1, n_lines_max=4, n_points_min=1, n_points_max=10) -> Dict[str, np.ndarray]:
    sample = generate_random_sample()
    
    # Create the sequence of all points
    sequence = np.array([EOL_TOKEN])
    
    # Create the sequence of points and their pointers
    all_pointers = []
    for ix in range(len(sample)):
        points = sample[ix]

        # Pointers
        offset = len(sequence)
        pointers = np.arange(len(points)) + offset

        all_pointers.extend(pointers)
        all_pointers.append(0)  # For the EOL token pointer

        # Points
        sequence = np.vstack([sequence, points])

    sequence = sequence.astype(np.float32)
    all_pointers = np.array(all_pointers).astype(np.long)

    return {
        'sequence': sequence,
        'pointers': all_pointers,
    }


def create_dataset(n_samples = 10, filename = 'lines_data.npz'):
    data = []
    for ix in range(n_samples):
        datum = generate_points_pointers()
        data.append(datum)

    np.savez_compressed(filename, data)


if __name__ == '__main__':
    pass
    create_dataset(n_samples=100000, filename='lines_data_train.npz')
    create_dataset(n_samples=1000, filename='lines_data_val.npz')