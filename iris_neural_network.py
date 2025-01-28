import numpy as np
import matplotlib.pyplot as plt
import os

# Change the working directory to the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


# Function to load dataset from file
def load_dataset(file):
    """
    Loads the dataset from a file and converts labels to numerical values.

    Args:
        file (str): Path to the dataset file.

    Returns:
        X (numpy.ndarray): Features matrix.
        y (numpy.ndarray): Labels vector.
    """

    def label_conversion_to_number(iris_name):
        """
        Converts iris class names to numerical values based on classification type.

        Args:
            iris_name (str): Name of the iris class.

        Returns:
            numpy.float32: Numerical label.
        """
        if classification_type == 'binary':
            # Classify 'Iris-setosa' as 0, and others as 1
            return np.float32(0) if iris_name == 'Iris-setosa' else np.float32(1)
        else:
            label_map = {'Iris-setosa': np.float32(0), 'Iris-versicolor': np.float32(1),
                         'Iris-virginica': np.float32(2)}
            return label_map[iris_name]

    # Load dataset from file and convert labels
    outputs = np.genfromtxt(file, delimiter=',', dtype=np.float32, encoding="utf-8",
                            converters={4: label_conversion_to_number})

    # Extract features (X) and labels (y)
    X = outputs[:, 0:-1]
    y = (outputs[:, -1]).astype(np.int_)

    # Shuffle rows to randomize the dataset
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    return X, y


# Function to normalize data
def normalization(X):
    """
    Normalizes the features matrix to have zero mean and unit variance.

    Args:
        X (numpy.ndarray): Features matrix.

    Returns:
        X_normalized (numpy.ndarray): Normalized features matrix.
        mean (numpy.ndarray): Mean of the features.
        std (numpy.ndarray): Standard deviation of the features.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std, mean, std


# ReLU activation function
def relu(x):
    """
    Applies the ReLU (Rectified Linear Unit) activation function.

    Args:
        x (numpy.ndarray): Input matrix.

    Returns:
        numpy.ndarray: Output after applying ReLU.
    """
    return np.maximum(0, x)


# Derivative of ReLU function
def relu_derivative(x):
    """
    Computes the derivative of the ReLU function.

    Args:
        x (numpy.ndarray): Input matrix.

    Returns:
        numpy.ndarray: Derivative of ReLU.
    """
    return (x > 0).astype(float)


# Quadratic loss (Mean Squared Error)
def quadratic_loss(y_predicted, y_target):
    """
    Computes the quadratic loss (MSE) between predicted and target values.

    Args:
        y_predicted (numpy.ndarray): Predicted values.
        y_target (numpy.ndarray): Target values.

    Returns:
        float: Quadratic loss.
    """
    return np.mean(np.sum((y_predicted - y_target) ** 2, axis=1))


# Derivative of quadratic loss
def quadratic_loss_derivative(y_predicted, y_target):
    """
    Computes the derivative of the quadratic loss function.

    Args:
        y_predicted (numpy.ndarray): Predicted values.
        y_target (numpy.ndarray): Target values.

    Returns:
        numpy.ndarray: Derivative of quadratic loss.
    """
    return 2 * (y_predicted - y_target) / y_target.shape[0]


# Cross-entropy loss function
def cross_entropy_loss(y_predicted, y_target):
    """
    Computes the cross-entropy loss for classification problems.

    Args:
        y_predicted (numpy.ndarray): Predicted probabilities.
        y_target (numpy.ndarray): Target labels.

    Returns:
        float: Cross-entropy loss.
    """
    m = y_target.shape[0]
    log_likelihood = -np.log(y_predicted[range(m), y_target.astype(int)])
    loss = np.sum(log_likelihood) / m
    return loss


# Derivative of cross-entropy loss
def cross_entropy_loss_derivative(y_predicted, y_target):
    """
    Computes the derivative of the cross-entropy loss function.

    Args:
        y_predicted (numpy.ndarray): Predicted probabilities.
        y_target (numpy.ndarray): Target labels.

    Returns:
        numpy.ndarray: Derivative of cross-entropy loss.
    """
    m = y_target.shape[0]
    grad = y_predicted
    grad[range(m), y_target.astype(int)] -= 1
    grad = grad / m
    return grad


# Function to convert integer targets to one-hot encoding
def to_one_hot(y, num_classes):
    """
    Converts integer labels to one-hot encoded vectors.

    Args:
        y (numpy.ndarray): Integer labels.
        num_classes (int): Number of classes.

    Returns:
        numpy.ndarray: One-hot encoded labels.
    """
    return np.eye(num_classes)[y]


# Function to initialize weights
def initialize_weights(input_size, hidden_size, output_size, use_hidden_layer):
    """
    Initializes the weights and biases for the neural network.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of neurons in the hidden layer.
        output_size (int): Number of output classes.
        use_hidden_layer (bool): Whether to use a hidden layer.

    Returns:
        W1, b1, W2, b2: Weights and biases for the network.
    """
    if use_hidden_layer:
        W1 = np.random.randn(input_size, hidden_size) * 0.01
        b1 = np.zeros((1, hidden_size))
    else:
        W1, b1 = None, None  # No hidden layer

    W2 = np.random.randn(hidden_size if use_hidden_layer else input_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))

    return W1, b1, W2, b2


# Activation function
def activation(x):
    """
    Applies the activation function (ReLU or linear) based on the user's choice.

    Args:
        x (numpy.ndarray): Input matrix.

    Returns:
        numpy.ndarray: Output after applying the activation function.
    """
    if use_non_linear:
        return relu(x)
    else:
        return x  # Linear activation (identity function)


# Output activation function
def output_activation(x):
    """
    Applies the output activation function (Softmax or linear) based on the user's choice.

    Args:
        x (numpy.ndarray): Input matrix.

    Returns:
        numpy.ndarray: Output after applying the output activation function.
    """
    if use_non_linear:
        # Softmax for non-linear activation (classification problems)
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    else:
        # Linear transformation (raw output)
        return x


# Function to train the neural network
def train(X_train, y_train, X_val, y_val, epochs=1000, lr=0.01, batch_size=32, update_mode="", loss_type="",
          use_hidden_layer=""):
    """
    Trains the neural network using the specified parameters.

    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training labels.
        X_val (numpy.ndarray): Validation features.
        y_val (numpy.ndarray): Validation labels.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        batch_size (int): Size of mini-batches.
        update_mode (str): Training mode ('batch', 'mini-batch', 'online').
        loss_type (str): Loss function type ('quadratic', 'cross_entropy').
        use_hidden_layer (bool): Whether to use a hidden layer.

    Returns:
        W1, b1, W2, b2: Trained weights and biases.
        losses_to_draw (list): Training losses over epochs.
        val_accuracies (list): Validation accuracies over epochs.
    """
    input_size = X_train.shape[1]
    hidden_size = 10

    if classification_type == 'binary':
        output_size = 2
    else:
        output_size = 3

    # Initialize weights based on whether or not there is a hidden layer
    W1, b1, W2, b2 = initialize_weights(input_size, hidden_size, output_size, use_hidden_layer)

    losses_to_draw, val_accuracies = [], []

    for e in range(epochs):
        if update_mode == 'batch':
            X_batch = X_train
            y_batch = y_train

            # Forward stage
            if use_hidden_layer:
                L1 = np.dot(X_batch, W1) + b1
                A1 = activation(L1)  # Use the new activation function
                L2 = np.dot(A1, W2) + b2
            else:
                L2 = np.dot(X_batch, W2) + b2  # No hidden layer

            A2 = output_activation(L2)  # Use the new output activation function

            # Compute the loss
            if loss_type == 'quadratic':
                y_batch_one_hot = to_one_hot(y_batch, output_size)
                loss = quadratic_loss(A2, y_batch_one_hot)
                loss_derivative = quadratic_loss_derivative(A2, y_batch_one_hot)
            elif loss_type == 'cross_entropy':
                loss = cross_entropy_loss(A2, y_batch)
                loss_derivative = cross_entropy_loss_derivative(A2, y_batch)

            # Backward stage
            dA2 = loss_derivative
            dL2 = dA2
            dW2 = np.dot(A1.T, dL2) if use_hidden_layer else np.dot(X_batch.T, dL2)
            db2 = np.sum(dL2, axis=0, keepdims=True)

            if use_hidden_layer:
                dA1 = np.dot(dL2, W2.T)
                if use_non_linear:
                    dL1 = dA1 * relu_derivative(A1)
                else:
                    dL1 = dA1  # Derivative of linear activation is 1
                dW1 = np.dot(X_batch.T, dL1)
                db1 = np.sum(dL1, axis=0, keepdims=True)
                W1 -= lr * dW1
                b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2

        elif update_mode == 'mini-batch':
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                # Forward stage
                if use_hidden_layer:
                    L1 = np.dot(X_batch, W1) + b1
                    A1 = activation(L1)  # Use the new activation function
                    L2 = np.dot(A1, W2) + b2
                else:
                    L2 = np.dot(X_batch, W2) + b2  # No hidden layer

                A2 = output_activation(L2)  # Use the new output activation function

                # Compute the loss
                if loss_type == 'quadratic':
                    y_batch_one_hot = to_one_hot(y_batch, output_size)
                    loss = quadratic_loss(A2, y_batch_one_hot)
                    loss_derivative = quadratic_loss_derivative(A2, y_batch_one_hot)
                elif loss_type == 'cross_entropy':
                    loss = cross_entropy_loss(A2, y_batch)
                    loss_derivative = cross_entropy_loss_derivative(A2, y_batch)

                # Backward stage
                dA2 = loss_derivative
                dL2 = dA2
                dW2 = np.dot(A1.T, dL2) if use_hidden_layer else np.dot(X_batch.T, dL2)
                db2 = np.sum(dL2, axis=0, keepdims=True)

                if use_hidden_layer:
                    dA1 = np.dot(dL2, W2.T)
                    if use_non_linear:
                        dL1 = dA1 * relu_derivative(A1)
                    else:
                        dL1 = dA1  # Derivative of linear activation is 1
                    dW1 = np.dot(X_batch.T, dL1)
                    db1 = np.sum(dL1, axis=0, keepdims=True)
                    W1 -= lr * dW1
                    b1 -= lr * db1
                W2 -= lr * dW2
                b2 -= lr * db2

        elif update_mode == 'online':
            for i in range(len(X_train)):
                X_batch = X_train[i:i + 1]
                y_batch = y_train[i:i + 1]

                # Forward stage
                if use_hidden_layer:
                    L1 = np.dot(X_batch, W1) + b1
                    A1 = activation(L1)  # Use the new activation function
                    L2 = np.dot(A1, W2) + b2
                else:
                    L2 = np.dot(X_batch, W2) + b2  # No hidden layer

                A2 = output_activation(L2)  # Use the new output activation function

                # Compute the loss
                if loss_type == 'quadratic':
                    y_batch_one_hot = to_one_hot(y_batch, output_size)
                    loss = quadratic_loss(A2, y_batch_one_hot)
                    loss_derivative = quadratic_loss_derivative(A2, y_batch_one_hot)
                elif loss_type == 'cross_entropy':
                    loss = cross_entropy_loss(A2, y_batch)
                    loss_derivative = cross_entropy_loss_derivative(A2, y_batch)

                # Backward stage
                dA2 = loss_derivative
                dL2 = dA2
                dW2 = np.dot(A1.T, dL2) if use_hidden_layer else np.dot(X_batch.T, dL2)
                db2 = np.sum(dL2, axis=0, keepdims=True)

                if use_hidden_layer:
                    dA1 = np.dot(dL2, W2.T)
                    if use_non_linear:
                        dL1 = dA1 * relu_derivative(A1)
                    else:
                        dL1 = dA1  # Derivative of linear activation is 1
                    dW1 = np.dot(X_batch.T, dL1)
                    db1 = np.sum(dL1, axis=0, keepdims=True)
                    W1 -= lr * dW1
                    b1 -= lr * db1
                W2 -= lr * dW2
                b2 -= lr * db2

        # Compute the accuracy on validation set
        y_val_pred = predict(X_val, W1, b1, W2, b2, use_hidden_layer)
        val_accuracy = np.mean(np.argmax(y_val_pred, axis=1) == y_val) * 100
        val_accuracies.append(val_accuracy)
        losses_to_draw.append(loss)

        # Print the updates during training phase
        print(f"Epoch {e + 1}/{epochs} - Loss: {loss:.4f} - Val Accuracy: {val_accuracy:.2f}%")

    return W1, b1, W2, b2, losses_to_draw, val_accuracies


# Function to make predictions
def predict(X, W1, b1, W2, b2, use_hidden_layer):
    """
    Makes predictions using the trained neural network.

    Args:
        X (numpy.ndarray): Input features.
        W1, b1, W2, b2: Weights and biases of the network.
        use_hidden_layer (bool): Whether to use a hidden layer.

    Returns:
        numpy.ndarray: Predicted probabilities.
    """
    if use_hidden_layer:
        L1 = np.dot(X, W1) + b1
        A1 = activation(L1)  # Use the new activation function
        L2 = np.dot(A1, W2) + b2
    else:
        L2 = np.dot(X, W2) + b2  # No hidden layer

    A2 = output_activation(L2)  # Use the new output activation function
    return A2


# Function to plot training loss and validation accuracy
def plot_loss_and_accuracy(train_losses, val_accuracies):
    """
    Plots the training loss and validation accuracy over epochs.

    Args:
        train_losses (list): Training losses over epochs.
        val_accuracies (list): Validation accuracies over epochs.
    """
    plt.figure(figsize=(12, 6))

    # Loss function
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")

    # Validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title("Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")

    plt.tight_layout()
    plt.show()


# Main script execution
if __name__ == "__main__":
    # Choose between standard classification (on 3 classes) and binary classification (Iris setosa vs others)
    classification_type = input(
        "Do you want the standard classification or the binary one? (standard/binary): ").strip().lower()

    # Load dataset and normalize
    data_X, data_y = load_dataset('iris.data')
    data_X, mean, std = normalization(data_X)

    # Divide dataset into training and validation sets
    train_frac = 0.6
    train_size = int(len(data_X) * train_frac)
    X_train, X_val = data_X[:train_size], data_X[train_size:]
    y_train, y_val = data_y[:train_size], data_y[train_size:]

    # Input from user for whether to use hidden layer or not
    use_hidden_layer = input("Do you want to use the hidden layer? (yes/no): ").strip().lower() == 'yes'

    # Input from user for whether to use non-linear activation functions or not
    use_non_linear = input("Do you want to use non-linear activation functions? (yes/no): ").strip().lower() == 'yes'

    # Input from user for the update mode (online, mini-batch, batch)
    update_mode = input("Choose update mode (online/mini-batch/batch): ").strip().lower()

    if use_non_linear:
        # Input from user for the loss_type (quadratic or cross_entropy)
        loss_type = input("Choose loss function (quadratic/cross_entropy): ").strip().lower()
    else:
        loss_type = 'quadratic'
        print("Linear activation function is used, so the loss function is set to quadratic.")

    # Train the model
    W1, b1, W2, b2, train_losses, val_accuracies = train(X_train, y_train, X_val, y_val, epochs=150, lr=0.1,
                                                         batch_size=15,
                                                         update_mode=update_mode, loss_type=loss_type,
                                                         use_hidden_layer=use_hidden_layer)

    # Plot the results
    plot_loss_and_accuracy(train_losses, val_accuracies)

    # Final prediction and accuracy on the test set
    y_pred_train = predict(X_train, W1, b1, W2, b2, use_hidden_layer)
    y_pred_val = predict(X_val, W1, b1, W2, b2, use_hidden_layer)

    train_accuracy = np.mean(np.argmax(y_pred_train, axis=1) == y_train) * 100
    val_accuracy = np.mean(np.argmax(y_pred_val, axis=1) == y_val) * 100

    print(f"Training accuracy: {train_accuracy:.2f}%")
    print(f"Validation accuracy: {val_accuracy:.2f}%")