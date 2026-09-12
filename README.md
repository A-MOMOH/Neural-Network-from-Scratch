# Neural Network from Scratch

A fully connected neural network implemented from scratch with **NumPy** for multi-class classification. This project explores the core mechanics behind neural network training by manually implementing forward propagation, backpropagation, gradient-based optimization, dropout, and weight initialization without using deep learning frameworks such as PyTorch or TensorFlow.

The network is evaluated on the **UCI Predict Students' Dropout and Academic Success** dataset, where the goal is to classify students as **Dropout**, **Enrolled**, or **Graduate** based on demographic, socioeconomic, and academic features.

## Overview

The main goal of this project is to understand how a neural network learns by implementing its major components directly with NumPy rather than relying on automatic differentiation or pre-built neural network layers.

The implementation includes:

- Fully connected hidden layers
- ReLU activation
- Softmax output activation
- Categorical cross-entropy loss
- Manual forward propagation
- Manual backpropagation
- Gradient-based weight and bias updates
- Batch Gradient Descent
- Stochastic Gradient Descent (SGD)
- Mini-Batch Gradient Descent
- Dropout regularization
- Small Random, Xavier, and He weight initialization
- Configurable hidden-layer sizes
- Training and validation tracking
- Hyperparameter experiments and visualizations

The experiments compare how optimization strategy, dropout, initialization, learning rate, and network size affect convergence and generalization.

---

## Dataset

The project uses the **UCI Predict Students' Dropout and Academic Success** dataset.

The dataset contains:

- **4,424 student records**
- **36 input features**
- **3 target classes:**
  - Dropout
  - Enrolled
  - Graduate

The target labels are mapped to integer class indices and then converted to one-hot encoded vectors for training.

The data is divided into approximately:

| Split | Percentage |
|---|---:|
| Training | 64% |
| Validation | 16% |
| Testing | 20% |

Feature standardization is performed using `StandardScaler`.

The scaler is fit only on the training data and then applied to the validation and test sets, preventing information from the validation or test data from influencing preprocessing.

A fixed random seed of `42` is used for reproducibility.

---

## Neural Network Architecture

The baseline neural network contains two hidden layers:

```text
36 Input Features
        │
        ▼
┌─────────────────────┐
│   Hidden Layer 1    │
│     64 Neurons      │
│        ReLU         │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Hidden Layer 2    │
│     32 Neurons      │
│        ReLU         │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│    Output Layer     │
│      3 Neurons      │
│       Softmax       │
└─────────────────────┘
        │
        ▼
Dropout | Enrolled | Graduate
```

The baseline architecture can therefore be represented as:

```python
[36, 64, 32, 3]
```

The two hidden layers use **ReLU**, while the output layer uses **Softmax** to generate probabilities for the three possible classes.

---

## Forward Propagation

Forward propagation calculates the network's prediction by passing the input through each layer.

For each layer, the linear transformation is:

```text
Z = A_prev @ W + b
```

where:

- `A_prev` contains the activations from the previous layer
- `W` contains the layer's weights
- `b` contains the biases
- `Z` contains the resulting linear combination

For hidden layers, ReLU is applied:

```text
A = ReLU(Z)
```

The output layer instead applies Softmax:

```text
A = Softmax(Z)
```

The Softmax implementation subtracts the largest value in each row before exponentiation to improve numerical stability.

Intermediate values are stored during the forward pass so they can be reused during backpropagation.

---

## Loss Function

The network uses **categorical cross-entropy** to measure the difference between its predictions and the true class labels.

A small constant is added before taking the logarithm to prevent numerical issues caused by evaluating `log(0)`.

The resulting loss provides the error signal that is propagated backward through the network during training.

---

## Backpropagation

Backpropagation is implemented manually with NumPy.

For the Softmax output layer combined with categorical cross-entropy, the initial gradient simplifies to:

```text
dZ = y_pred - y_true
```

The gradients for the weights and biases are then calculated as:

```text
dW = (A_prev.T @ dZ) / n
```

```text
db = sum(dZ) / n
```

where `n` represents the number of examples in the current batch.

The gradient is propagated back toward the previous layer using:

```text
dA_prev = dZ @ W.T
```

For hidden layers, the gradient also passes through the derivative of ReLU.

This process continues backward through the network until gradients have been calculated for every trainable weight and bias.

The parameters are then updated using:

```text
W = W - learning_rate * dW
b = b - learning_rate * db
```

This means the entire learning process—from prediction to gradient calculation to parameter updates—is implemented without automatic differentiation.

---

## Optimization Strategies

Three gradient descent strategies are implemented and compared.

### Batch Gradient Descent

Batch Gradient Descent uses the entire training set to calculate one parameter update per epoch.

It produced smooth and stable learning curves but converged relatively slowly.

### Stochastic Gradient Descent

Stochastic Gradient Descent updates the model after every individual training example.

This creates many more parameter updates, allowing the model to learn aggressively, but also introduces considerably more noise during training.

### Mini-Batch Gradient Descent

Mini-Batch Gradient Descent divides the training data into smaller groups and performs one update for each group.

The experiments included batch sizes such as:

```text
16
32
64
128
```

Mini-Batch Gradient Descent provided the strongest overall balance between convergence speed, stability, and generalization.

---

## Dropout Regularization

Dropout is implemented for the hidden layers as a regularization technique.

During training, a random binary mask is applied to hidden-layer activations, temporarily disabling a fraction of neurons.

The experiments compare dropout rates of:

```text
0.0
0.2
0.5
```

During evaluation, dropout is disabled so the full network is used to generate predictions.

A dropout rate of `0.2` reduced the gap between training and validation performance and improved test accuracy compared with the equivalent baseline configuration without dropout.

---

## Weight Initialization

The project implements and compares three initialization strategies.

### Small Random Initialization

Weights are initialized using small values sampled from a normal distribution.

### Xavier Initialization

Xavier initialization scales the initial weights based on the number of incoming and outgoing connections.

### He Initialization

He initialization scales weights based on the number of incoming connections and is suited to networks using ReLU activations.

The initialization experiments demonstrate that the starting distribution of the weights can affect both convergence and final performance.

---

## Hyperparameter Experiments

Several experiments were conducted to understand how different training choices affect the network.

### Learning Rate

The following learning rates were evaluated:

```text
0.001
0.01
0.1
```

A learning rate of `0.001` resulted in slower learning, while `0.01` provided a stronger balance between convergence and generalization.

A learning rate of `0.1` allowed the model to fit the training data very aggressively but resulted in worse validation and test performance.

### Hidden-Layer Size

Three hidden-layer configurations were compared:

```text
[32, 16]
[64, 32]
[128, 64]
```

Increasing the size of the network improved training accuracy but did not consistently improve validation or test performance.

This demonstrates that additional model capacity can increase overfitting rather than automatically improving generalization.

---

## Experimental Results

| Optimizer | Dropout | Initialization | Learning Rate | Hidden Layers | Batch Size | Train Acc. | Validation Acc. | Test Acc. |
|---|---:|---|---:|---|---:|---:|---:|---:|
| Batch GD | 0.0 | He | 0.01 | [64, 32] | 2831 | 0.6305 | 0.6271 | 0.6271 |
| SGD | 0.0 | He | 0.01 | [64, 32] | 1 | 0.9968 | 0.6992 | 0.7243 |
| Mini-Batch | 0.0 | He | 0.01 | [64, 32] | 32 | 0.8573 | 0.7472 | 0.7412 |
| Mini-Batch | 0.2 | He | 0.01 | [64, 32] | 32 | 0.8022 | 0.7472 | 0.7559 |
| Mini-Batch | 0.5 | He | 0.01 | [64, 32] | 32 | 0.7616 | 0.7274 | 0.7412 |
| Mini-Batch | 0.0 | Small Random | 0.01 | [64, 32] | 32 | 0.7792 | 0.7500 | 0.7458 |
| Mini-Batch | 0.0 | Xavier | 0.01 | [64, 32] | 32 | 0.8569 | 0.7472 | **0.7571** |
| Mini-Batch | 0.0 | He | 0.001 | [64, 32] | 32 | 0.7662 | 0.7133 | 0.7446 |
| Mini-Batch | 0.0 | He | 0.1 | [64, 32] | 32 | 0.9982 | 0.7034 | 0.7186 |
| Mini-Batch | 0.0 | He | 0.01 | [32, 16] | 32 | 0.8241 | **0.7556** | 0.7469 |
| Mini-Batch | 0.0 | He | 0.01 | [128, 64] | 32 | 0.9050 | 0.7218 | 0.7333 |

---

## Key Findings

### Gradient Descent

Batch Gradient Descent produced the smoothest learning behavior but converged slowly.

SGD reached very high training accuracy but showed a large difference between training and validation performance, indicating overfitting.

Mini-Batch Gradient Descent provided a better balance between convergence, stability, and generalization.

### Dropout

Without dropout, the Mini-Batch model achieved:

```text
Training Accuracy:   0.8573
Validation Accuracy: 0.7472
Test Accuracy:       0.7412
```

With a dropout rate of `0.2`:

```text
Training Accuracy:   0.8022
Validation Accuracy: 0.7472
Test Accuracy:       0.7559
```

The lower training accuracy combined with improved test performance suggests that dropout reduced overfitting.

### Weight Initialization

Xavier initialization achieved the highest individual test accuracy in the initialization comparison:

```text
Test Accuracy: 0.7571
```

Small Random initialization achieved `0.7458`, while the baseline He configuration without dropout achieved `0.7412`.

### Learning Rate

The `0.01` learning rate provided the strongest overall balance.

Increasing the learning rate to `0.1` resulted in a training accuracy of `0.9982`, but test accuracy fell to `0.7186`, demonstrating that extremely high training accuracy does not necessarily indicate strong generalization.

### Network Capacity

The largest tested architecture, `[128, 64]`, reached a training accuracy of `0.9050` but a test accuracy of only `0.7333`.

The smaller `[32, 16]` network achieved a lower training accuracy of `0.8241` but a higher test accuracy of `0.7469`.

This demonstrates the tradeoff between model capacity and generalization.

---

## Selected Configuration

The final selected configuration was:

```text
Optimizer:       Mini-Batch Gradient Descent
Batch Size:      32
Hidden Layers:   [64, 32]
Initialization:  He
Learning Rate:   0.01
Dropout Rate:    0.2
Epochs:          100
```

Its final performance was:

```text
Training Accuracy:   0.8022
Validation Accuracy: 0.7472
Test Accuracy:       0.7559
```

Although the Xavier experiment produced a slightly higher individual test accuracy of `0.7571`, the selected model provided a strong overall balance between training performance and generalization while incorporating dropout to reduce overfitting.

---

## Project Structure

```text
Neural-Network-from-Scratch/
│
├── neural_network.py
├── neural_network_notebook.ipynb
├── data.csv
├── requirements.txt
└── README.md
```

### `neural_network.py`

Contains the neural network implementation and the selected training configuration.

### `neural_network_notebook.ipynb`

Contains the full experimental workflow, including training curves and comparisons of gradient descent strategies, dropout rates, initialization methods, learning rates, and hidden-layer configurations.

### `data.csv`

Contains the dataset used for training, validation, and testing.

### `requirements.txt`

Lists the Python packages required to run the project.

---

## Installation

### Requirements

- **Python 3.13**
- **pip**

Python 3.13 was used to develop and run this project.

### 1. Clone the Repository

```bash
git clone https://github.com/A-MOMOH/Neural-Network-from-Scratch.git
cd Neural-Network-from-Scratch
```

### 2. Install the Dependencies

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains:

```text
numpy
pandas
matplotlib
scikit-learn
```

### 3. Run the Neural Network

Run the Python implementation with:

```bash
python neural_network.py
```

The script trains the selected neural network configuration and reports its test performance.

### 4. Explore the Experiments

Open:

```text
neural_network_notebook.ipynb
```

using Jupyter Notebook or JupyterLab to explore the complete experiments, training curves, and model comparisons.

---

## Technologies

**Python 3.13** — primary programming language

**NumPy** — neural network computations, matrix operations, forward propagation, backpropagation, and parameter updates

**pandas** — dataset loading and manipulation

**scikit-learn** — train/test splitting, feature standardization, and evaluation metrics

**Matplotlib** — visualization of training and validation performance

**Jupyter Notebook** — interactive experimentation and analysis

The neural network itself is implemented with **NumPy rather than a machine learning or deep learning framework**.

---

## Conclusion

This project demonstrates the complete training process of a feedforward neural network by implementing its core components from scratch.

Building forward propagation, backpropagation, loss calculation, parameter updates, dropout, and optimization manually provides a deeper understanding of what occurs internally when a neural network learns.

The experiments also show that model performance depends on more than training accuracy alone. Optimization strategy, regularization, initialization, learning rate, and network capacity all influence how well a neural network generalizes to unseen data.
