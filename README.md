# Pointer networks

This implements the model from [Pointer Networks](https://arxiv.org/abs/1506.03134) in PyTorch from scratch.

The convex hull data generation code was copied from [here](https://github.com/vshallc/PtrNets).


## Requirements
1. pytorch 1.0.0: install independently
2. Refer requirements.txt


## Supported tasks
1. Convex hull
2. Simple line grouping task
3. Documents: words to container (textline/field/key/value/etc.) using b-boxes

**The actual model is implemented in `pointer_net.py`**


## Usage
For any of the above tasks:
1. Generate data using the `<task>_generator.py` file
2. Train using the `<task>-train.ipynb` notebook
