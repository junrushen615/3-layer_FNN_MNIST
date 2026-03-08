# V1

## 1. Description
V1 implements a 3-layer FNN from scratch using **NumPy**, trained on the **MNIST** (Random 80% training / 20% test split).  
- **Architecture:** Input 784 → Hidden 128 → Hidden 64 → Output 10  
- **Activations:** ReLU (hidden layers), Softmax (output layer)  
- **Loss Function:** Cross-Entropy  
- **Optimizer:** Mini-Batch Stochastic Gradient Descent (SGD)  
- **Mini-Batch Size:** 128  
- **Epochs:** 20  
- **Learning Rate:** 0.1  

## 2. Training Progress
| Epoch | Loss   | Training Accuracy |
|-------|--------|-----------------|
| 0     | 0.2391 | 0.9288          |
| 5     | 0.0915 | 0.9725          |
| 10    | 0.0382 | 0.9900          |
| 15    | 0.0269 | 0.9929          |
| 19    | 0.0149 | 0.9972          |

> **Note:** Loss decreases rapidly and training accuracy approaches ~99.7% by the final epoch.

## 3. Test Results
- **Test Set Accuracy:** 0.9746 (97.46%)

---

# V2

## 1. Description
The main difference between **V1** and **V2** is the **training data split**.  
**V2:** Official MNIST split: first 60,000 images for training, last 10,000 images for testing.

> This version achieved slightly higher test accuracy (~97.89%) compared to V1 (~97.46%).

## 2. Training Progress
| Epoch | Loss   | Training Accuracy |
|-------|--------|-----------------|
| 0     | 0.2367 | 0.9314          |
| 5     | 0.0812 | 0.9764          |
| 10    | 0.0448 | 0.9873          |
| 15    | 0.0261 | 0.9936          |
| 19    | 0.0179 | 0.9961          |

> Loss decreases steadily and training accuracy exceeds 99% by the final epoch.

## 3. Test Results
- **Test Set Accuracy:** 0.9789 (97.89%)  
- Demonstrates improved generalization over V1.

---
# V3

## 1. Description
This version extends V2 with a **validation set** to monitor overfitting and improve training stability.  
Key updates from V2:  
- **Training Split:** 50,000 for training, 10,000 for validation (from the official 60k training images)  
- **Hidden Layer 1 Size:** Increased from 128 → 256 for more capacity  
- **Learning Rate:** Reduced to 0.05 for more stable convergence  
- **Epochs:** 40 to allow more gradual training  

## 2. Training & Validation Progress
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|------------|-----------|----------|---------|
| 0     | 0.3017     | 0.9146    | 0.2758   | 0.9219  |
| 5     | 0.1373     | 0.9608    | 0.1456   | 0.9599  |
| 10    | 0.0737     | 0.9799    | 0.0988   | 0.9737  |
| 15    | 0.0507     | 0.9861    | 0.0880   | 0.9759  |
| 20    | 0.0328     | 0.9927    | 0.0802   | 0.9770  |
| 25    | 0.0247     | 0.9951    | 0.0793   | 0.9768  |
| 30    | 0.0159     | 0.9978    | 0.0762   | 0.9795  |
| 35    | 0.0119     | 0.9987    | 0.0779   | 0.9785  |
| 39    | 0.0096     | 0.9991    | 0.0779   | 0.9792  |

> Training accuracy approaches 99.9%, validation accuracy stabilizes around 97.9%, indicating strong generalization.

## 3. Test Results
- **Test Set Accuracy:** 0.9797 (97.97%)  

> Using a separate validation set helped monitor training, adjust learning, and avoid overfitting.
