import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# ==========================================
# 1. Activation Functions & Loss
# ==========================================
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    # Subtracting max for numerical stability
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    epsilon = 1e-9 # Prevent log(0)
    return -np.sum(y_true * np.log(y_pred + epsilon)) / m

# ==========================================
# 2. Neural Network Class (with Mini-Batch)
# ==========================================
class ThreeLayerNN:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, learning_rate=0.1):
        self.lr = learning_rate
        
        # He initialization
        self.params = {
            'W1': np.random.randn(input_size, hidden1_size) * np.sqrt(2. / input_size),
            'b1': np.zeros((1, hidden1_size)),
            'W2': np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2. / hidden1_size),
            'b2': np.zeros((1, hidden2_size)),
            'W3': np.random.randn(hidden2_size, output_size) * np.sqrt(2. / hidden2_size),
            'b3': np.zeros((1, output_size))
        }

    def forward(self, X):
        self.cache = {'A0': X}
        
        # Layer 1
        self.cache['Z1'] = np.dot(X, self.params['W1']) + self.params['b1']
        self.cache['A1'] = relu(self.cache['Z1'])
        
        # Layer 2
        self.cache['Z2'] = np.dot(self.cache['A1'], self.params['W2']) + self.params['b2']
        self.cache['A2'] = relu(self.cache['Z2'])
        
        # Layer 3 (Output)
        self.cache['Z3'] = np.dot(self.cache['A2'], self.params['W3']) + self.params['b3']
        self.cache['A3'] = softmax(self.cache['Z3'])
        
        return self.cache['A3']

    def backward(self, Y):
        m = Y.shape[0] # dynamically adapts to batch size
        grads = {}
        
        # Layer 3 Gradients
        dZ3 = self.cache['A3'] - Y
        grads['dW3'] = (1 / m) * np.dot(self.cache['A2'].T, dZ3)
        grads['db3'] = (1 / m) * np.sum(dZ3, axis=0, keepdims=True)
        
        # Layer 2 Gradients
        dZ2 = np.dot(dZ3, self.params['W3'].T) * relu_derivative(self.cache['Z2'])
        grads['dW2'] = (1 / m) * np.dot(self.cache['A1'].T, dZ2)
        grads['db2'] = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Layer 1 Gradients
        dZ1 = np.dot(dZ2, self.params['W2'].T) * relu_derivative(self.cache['Z1'])
        grads['dW1'] = (1 / m) * np.dot(self.cache['A0'].T, dZ1)
        grads['db1'] = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Update parameters
        for key in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
            self.params[key] -= self.lr * grads['d' + key]

    def train(self, X, Y, epochs=20, batch_size=128):
        m = X.shape[0]
        
        for epoch in range(epochs):
            # 1. Shuffle the dataset at the start of each epoch
            permutation = np.random.permutation(m)
            X_shuffled = X[permutation]
            Y_shuffled = Y[permutation]
            
            # 2. Iterate over mini-batches
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                Y_batch = Y_shuffled[i:i+batch_size]
                
                # Forward and backward pass on the batch
                self.forward(X_batch)
                self.backward(Y_batch)
            
            # Print progress every 5 epochs
            if epoch % 5 == 0 or epoch == epochs - 1:
                # Calculate metrics on the whole training set to track progress
                y_pred_full = self.forward(X)
                loss = cross_entropy_loss(Y, y_pred_full)
                accuracy = np.mean(np.argmax(y_pred_full, axis=1) == np.argmax(Y, axis=1))
                print(f"Epoch {epoch:2d}/{epochs} | Loss: {loss:.4f} | Training Acc: {accuracy:.4f}")

    def predict(self, X):
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)

# ==========================================
# 3. MNIST Data Loading and Execution
# ==========================================
if __name__ == "__main__":
    print("Downloading MNIST dataset (this may take a minute)...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    
    # Convert to NumPy arrays and normalize
    X = np.array(mnist.data) / 255.0
    y = np.array(mnist.target).astype(int)

    # One-hot encode the labels
    encoder = OneHotEncoder(sparse_output=False)
    Y_onehot = encoder.fit_transform(y.reshape(-1, 1))

    # Split into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=0.2, random_state=42)

    # Initialize network
    print("\nStarting training with mini-batches...")
    nn = ThreeLayerNN(
        input_size=784, 
        hidden1_size=128, 
        hidden2_size=64, 
        output_size=10, 
        learning_rate=0.1
    )
    
    nn.train(X_train, Y_train, epochs=20, batch_size=128)

    # Evaluate on test set
    test_preds = nn.predict(X_test)
    y_test_labels = np.argmax(Y_test, axis=1)
    test_accuracy = np.mean(test_preds == y_test_labels)
    print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
