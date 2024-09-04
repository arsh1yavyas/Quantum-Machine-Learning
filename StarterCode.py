import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import algorithm_globals

from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN

algorithm_globals.random_seed = 42

# There may be some encoding issues, so we specify a specific encoding. Otherwise, the dataset can not be loaded.
import pandas as pd

data = pd.read_csv("DataCleaned.csv", encoding = "ISO-8859-1", low_memory=False)

dataNoNaN = data.fillna(data.mean())
normalized = 2*(dataNoNaN-dataNoNaN.mean())/(dataNoNaN.max()-dataNoNaN.min())
