# Iris Classification Neural Network

This repository contains a Python implementation of a simple neural network to classify the Iris dataset. It includes both binary and standard classification modes, with the ability to choose whether to use hidden layers, non-linear activation functions, and various update modes (batch, mini-batch, or online). The network is trained using different loss functions (quadratic loss or cross-entropy) based on user input.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Features](#features)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Results](#results)
- [License](#license)

## Overview

The Iris dataset is a well-known dataset in the machine learning community, containing three classes of Iris plants: `Iris-setosa`, `Iris-versicolor`, and `Iris-virginica`. The task is to classify these three types of iris flowers based on four features: sepal length, sepal width, petal length, and petal width.

This neural network implementation can perform classification tasks in two modes:
- **Binary classification**: Classifying `Iris-setosa` vs others. This mode is used specifically to observe how the network performs with a linearly separable binary classification problem, where no non-linearity is introduced, particularly to see how well the network can converge in such a simple problem.
- **Standard classification**: Classifying the three Iris classes (`Iris-setosa`, `Iris-versicolor`, `Iris-virginica`).

The neural network uses fully connected layers, with options to include a hidden layer, and uses ReLU activation (or linear activation). The output layer uses either softmax or linear activation.

## Requirements

- Python 3.x
- NumPy
- Matplotlib

Install dependencies with:

- `pip install numpy matplotlib`

## Features

- **Classification types**: Binary (Iris-setosa vs others) or standard (three classes).
- **Hidden layers**: Option to use a hidden layer in the neural network.
- **Activation functions**: ReLU or linear activation for hidden layers and softmax or linear activation for the output layer.
- **Loss functions**: Quadratic loss (MSE) or cross-entropy loss for classification.
- **Update modes**: Batch, mini-batch, or online update modes for training.
- **Data normalization**: Normalization of the Iris dataset for training stability.
- **Visualization**: Plots for training loss and validation accuracy over epochs.

## Usage

To use this neural network, simply run the Python script and follow the interactive prompts.

1. Clone this repository:
- `git clone https://github.com/kevix01/FML_project.git cd FML_project`

2. Run the Python script:
- `python iris_neural_network.py`

3. The script will prompt you for several options, including:
   - Whether you want to perform binary or standard classification.
   - Whether to use a hidden layer in the neural network.
   - Whether to use non-linear activation functions (ReLU) or linear activation.
   - The update mode for training (online, mini-batch, or batch).
   - The loss function to use (quadratic or cross-entropy).

4. The model will train for the specified number of epochs, and you will see the training loss and validation accuracy at each epoch.

5. At the end of the training, the script will display the final training and validation accuracy.

## Training the Model

The training process involves the following steps:
1. Load and preprocess the Iris dataset.
2. Normalize the features for better training performance.
3. Split the dataset into training and validation sets.
4. Initialize the neural network weights and biases.
5. Train the network using the selected update mode (online, mini-batch, or batch).
6. Calculate the loss and update the weights based on the chosen loss function (quadratic or cross-entropy).
7. Output the training and validation accuracy after each epoch.

## Results

After training, the script will display:
- **Training accuracy**: The accuracy of the model on the training set.
- **Validation accuracy**: The accuracy of the model on the validation set.
- **Training loss**: The loss function value during training.
- **Validation accuracy plot**: A plot showing the validation accuracy over the epochs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
