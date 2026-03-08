# 3-layer_FNN_MNIST

This project implements a 3-layer feedforward neural network (FNN) from scratch using NumPy to classify handwritten digits from the MNIST dataset.

Through three iterative versions (`V1`, `V2`, and `V3`), the codebase evolves from a basic random-split neural network to a more robust version featuring official dataset splitting, validation tracking, and increased capacity.

## Environment Requirements
To run these scripts, you will need a standard Python data science environment. 

* **Python:** 3.7 or higher
* **Required Libraries:**
  * `numpy`: For core matrix operations and mathematical functions.
  * `scikit-learn`: Exclusively used for fetching the MNIST dataset (`fetch_openml`), one-hot encoding labels, and splitting data (in V1).

You can install the necessary dependencies using pip:
```bash
pip install numpy scikit-learn
```
## Usage Steps
1. Ensure that the Python scripts are downloaded and placed in your current working directory:
* `3layer_FNN_Numpy_V1.py`
* `3layer_FNN_Numpy_V2.py`
* `3layer_FNN_Numpy_V3.py`

2. Execute any of the versions directly from your terminal. The first time you run a script, it will download the MNIST dataset via fetch_openml, which may take a minute.

## Acknowledgements
I would like to acknowledge Gemini for assisting with code optimizations throughout this project. 
