from typing import List, Dict
import numpy as np


EOL_TOKEN = [-1, -1]


def generate_random_sample(
    n_lines_min=1, n_lines_max=4, n_points_min=1, n_points_max=10, add_noise: bool = False,
    n_line_buckets: int = 10,
) -> List[np.ndarray]:
    """
    Returns a list of "lines".
    
    Each "line" is an np.ndarray containing between 1-10 2-D points on the same y-coordinate.
    """
    assert 0 < n_lines_min <= n_lines_max <= n_line_buckets
    assert 0 < n_points_min <= n_points_max <= n_line_buckets

    n_lines = np.random.choice(np.arange(n_lines_min, n_lines_max + 1))

    # all the lines will be sampled at the discrete intervals - 0.1, 0.2 .... 1.0
    line_pos = np.random.choice(np.arange(n_line_buckets), n_lines, replace=False) + 1
    line_pos = sorted(line_pos, reverse=True)  # NOTE: THIS IS IMPORTANT! ORDER MATTERS: ALWAYS GIVE THE LINES TOP to BOTTOM

    coords = []
    for ix in range(n_lines):
        n_points = np.random.choice(np.arange(n_points_min, n_points_max + 1))
        x = sorted(np.random.random(n_points))
        y = np.array([line_pos[ix] / n_line_buckets] * n_points)
        
        if add_noise:
            # +- 30% of the min possible distance between two lines. for >= 50% - the task is NOT well defined..
            noise_max = 0.30 * 1. / n_line_buckets

            y_noise = np.random.rand(n_points) * noise_max
            y += y_noise

        points = np.array(list(zip(x, y))).astype(np.float)
        coords.append(points)

    return coords


def generate_points_pointers(**kwargs) -> Dict[str, np.ndarray]:
    sample = generate_random_sample(**kwargs)
    
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


def create_dataset(n_samples = 10, filename = 'lines_data.npz', random_n_line_buckets: bool = False, **kwargs):
    data = []
    for ix in range(n_samples):
        datum = generate_points_pointers(**kwargs)
        data.append(datum)

    np.savez_compressed(filename, data)


if __name__ == '__main__':
    choice = 'insane'
    
    if choice == 'easy':
        create_dataset(n_samples=100000, filename='lines_data_train.npz', n_lines_max=4)
        create_dataset(n_samples=1000, filename='lines_data_val.npz', n_lines_max=4)

    if choice == 'hard':
        create_dataset(n_samples=500000, filename='lines_data_train.npz', n_lines_max=10, add_noise=True)
        create_dataset(n_samples=5000, filename='lines_data_val.npz', n_lines_max=10, add_noise=True)

    if choice == 'insane':
        create_dataset(n_samples=500000, filename='lines_data_train.npz', n_lines_max=100, n_line_buckets=100)
        create_dataset(n_samples=5000, filename='lines_data_val.npz', n_lines_max=100, n_line_buckets=100)
