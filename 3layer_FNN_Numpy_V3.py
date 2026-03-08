import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder

# ==========================================
# 1. Activation Functions & Loss
# ==========================================
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    epsilon = 1e-9
    return -np.sum(y_true * np.log(y_pred + epsilon)) / m

# ==========================================
# 2. Neural Network Class (Version 2.0)
# ==========================================
class ThreeLayerNN:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, learning_rate=0.05):
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
        
        self.cache['Z1'] = np.dot(X, self.params['W1']) + self.params['b1']
        self.cache['A1'] = relu(self.cache['Z1'])
        
        self.cache['Z2'] = np.dot(self.cache['A1'], self.params['W2']) + self.params['b2']
        self.cache['A2'] = relu(self.cache['Z2'])
        
        self.cache['Z3'] = np.dot(self.cache['A2'], self.params['W3']) + self.params['b3']
        self.cache['A3'] = softmax(self.cache['Z3'])
        
        return self.cache['A3']

    def backward(self, Y):
        m = Y.shape[0]
        grads = {}
        
        dZ3 = self.cache['A3'] - Y
        grads['dW3'] = (1 / m) * np.dot(self.cache['A2'].T, dZ3)
        grads['db3'] = (1 / m) * np.sum(dZ3, axis=0, keepdims=True)
        
        dZ2 = np.dot(dZ3, self.params['W3'].T) * relu_derivative(self.cache['Z2'])
        grads['dW2'] = (1 / m) * np.dot(self.cache['A1'].T, dZ2)
        grads['db2'] = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        
        dZ1 = np.dot(dZ2, self.params['W2'].T) * relu_derivative(self.cache['Z1'])
        grads['dW1'] = (1 / m) * np.dot(self.cache['A0'].T, dZ1)
        grads['db1'] = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
        
        for key in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
            self.params[key] -= self.lr * grads['d' + key]

    def train(self, X_train, Y_train, X_val, Y_val, epochs=40, batch_size=128):
        m = X_train.shape[0]
        
        for epoch in range(epochs):
            # Shuffle training data
            permutation = np.random.permutation(m)
            X_shuffled = X_train[permutation]
            Y_shuffled = Y_train[permutation]
            
            # Mini-batch training
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                Y_batch = Y_shuffled[i:i+batch_size]
                self.forward(X_batch)
                self.backward(Y_batch)
            
            # Print metrics every 5 epochs
            if epoch % 5 == 0 or epoch == epochs - 1:
                # Training Metrics (calculated on the whole training set)
                train_preds = self.forward(X_train)
                train_loss = cross_entropy_loss(Y_train, train_preds)
                train_acc = np.mean(np.argmax(train_preds, axis=1) == np.argmax(Y_train, axis=1))
                
                # Validation Metrics (calculated on unseen validation set)
                val_preds = self.forward(X_val)
                val_loss = cross_entropy_loss(Y_val, val_preds)
                val_acc = np.mean(np.argmax(val_preds, axis=1) == np.argmax(Y_val, axis=1))
                
                print(f"Epoch {epoch:2d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    def predict(self, X):
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)

# ==========================================
# 3. Data Loading and Execution
# ==========================================
if __name__ == "__main__":
    print("Downloading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    
    X = np.array(mnist.data) / 255.0
    y = np.array(mnist.target).astype(int)

    encoder = OneHotEncoder(sparse_output=False)
    Y_onehot = encoder.fit_transform(y.reshape(-1, 1))

    # 1. Take the first 60k for training purposes
    X_train_full = X[:60000]
    Y_train_full = Y_onehot[:60000]
    
    # 2. Keep the official last 10k strictly for final testing
    X_test = X[60000:]
    Y_test = Y_onehot[60000:]

    # 3. Create a Validation Set from the 60k training pool
    # First 50k = Training | Last 10k = Validation
    print("Creating Train/Validation split...")
    X_train = X_train_full[:50000]
    Y_train = Y_train_full[:50000]
    
    X_val = X_train_full[50000:]
    Y_val = Y_train_full[50000:]

    # Initialize V2.0 Network (H1=256, LR=0.05)
    print("\nStarting Version 2.0 training...")
    nn = ThreeLayerNN(
        input_size=784, 
        hidden1_size=256,  # Upgraded capacity!
        hidden2_size=64, 
        output_size=10, 
        learning_rate=0.05 # Lower, more stable learning rate
    )
    
    # Train using the validation data
    nn.train(X_train, Y_train, X_val, Y_val, epochs=40, batch_size=128)

    # Final Evaluation
    test_preds = nn.predict(X_test)
    y_test_labels = np.argmax(Y_test, axis=1)
    test_accuracy = np.mean(test_preds == y_test_labels)
    
    print("-" * 45)
    print(f"Final Official Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print("-" * 45)