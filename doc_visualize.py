from typing import Dict, List
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
EOC_TOKEN = np.array([0, 0, 0, 0])
EOL_TOKEN = [0, 0]


def _draw_bbox(ax, bbox, margin=0, color='r', linestyle='solid', fill=False, **kwargs):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1

    x1 -= margin
    y1 -= margin
    w += 2*margin
    h += 2*margin

    rect = mpatches.Rectangle((x1, y1), w, h, fill=fill, color=color, linestyle=linestyle, **kwargs)
    ax.add_patch(rect)

def plot_word_bboxes_ponters(image, word_bboxes, pointers, figsize=(20, 10)):
    colors = [
        (1, 0, 0, 0.2),
        (1, 1, 0, 0.2),
        (1, 0, 1, 0.2),
        (0.5, 0, 0, 0.2),
        (0, 0, 0.5, 0.2),
        (0, 0.5, 0, 0.2),
        (0.5, 0.5, 0, 0.2),
    ]

    # Plot image
    plt.figure(figsize=figsize)
    ax = plt.subplot(1, 1, 1)
    ax.imshow(image)

    # Draw Words
#     np.random.seed(10)
    color = colors[np.random.randint(len(colors))]
    for pointer in pointers:
        bbox = word_bboxes[pointer]
        if np.all(bbox == EOC_TOKEN):  # THE EOC token
            color = colors[np.random.randint(len(colors))]
            continue

        bbox = word_bboxes[pointer].flatten()
        _draw_bbox(ax, bbox, fill=True, color=color)
        plt.text((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, str(pointer), ha='center', color='r')

def get_lines_from_pointers(points, pointers) -> List[np.ndarray]:
    """THIS IS THE POST-PROCESSING TO GO FROM OUTPUT TO LINES!"""
    lines = []
    cur_line = []
    for pointer in pointers:
        point = points[pointer]
        if np.all(point == EOL_TOKEN):
            if len(cur_line) > 0:
                lines.append(np.array(cur_line))
            cur_line = []
            continue
        cur_line.append(point)
    if len(cur_line) > 0:
        lines.append(np.array(cur_line))

    return lines

def plot_points_and_lines(points, pointers, image=None, scale=None, fontsize=20) -> None:
    inds = np.all(points != EOL_TOKEN, axis=1)
    points_to_plot = points.copy()

    # These are normalized to between 0 and 1
    if scale is not None:
        points_to_plot *= scale
    else:
        plt.xlim(0, 1.1)
        plt.ylim(0, 1.1)
    
    if image is not None:
        plt.imshow(image)

    # Plot points
    plt.scatter(points_to_plot[inds][:, 0], points_to_plot[inds][:, 1], marker='x')
    
    # Mark the numbers
    for pointer in range(len(points)):
        point = points_to_plot[pointer]
        if np.all(point == EOL_TOKEN):
            continue
        plt.text(point[0], point[1], str(pointer), ha='center', color='r', fontsize=fontsize)

    # Plot lines
    pointers = pointers[pointers != -100]
    lines = get_lines_from_pointers(points_to_plot, pointers)

    for line in lines:
        plt.plot(line[:, 0], line[:, 1], '--')