# CSCI 682 - Spring 2026
# Logistic Regression Skeleton

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Compute the sigmoid function on a numpy array elementwise
#x - float numpy array - input values on which to apply the sigmoid
#return - float numpy array - the elementwise sigmoid of x
def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def softmax(x):
  # subtract max per row for numerical stability before exponentiating
  e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
  return e_x / np.sum(e_x, axis=1, keepdims=True)



#Compute estimates of P(y=1|x) for each observation given a weight vector and a bias term
#X - numpy array - one row with an entry for each feature for each observation. in particular, this is an observations by features matrix
#w - numpy array - a weight vector with one entry for each feature
#b - float - a bias
#return - numpy array - an estimate of P(y=1|x) for each observation
def predict_proba(X, w, b):
  #check the size and the dimention 
  row_num, col_num = X.shape
  if col_num != w.shape[0]:
    raise ValueError(f"Dimension mismatch: X has {col_num} features but w has {w.shape[0]} entries")
  scores = X @ w + b
  if w.ndim == 1:           # binary: w is (d,)
    return sigmoid(scores)
  else:                     # multiclass: W is (d, K) -> scores is (n, K)
    return softmax(scores)

#Predict binary class labels for each provided observation
#X - numpy array - one row with an entry for each feature for each observation. in particular, this is an observations by features matrix
#w - numpy array - a weight vector with one entry for each feature
#b - float - a bias
#thresh - float - if the estimated probabaility is >= thresh, the predeicted class is 1. defaults to 0.5
#return - numpy array - 0 or 1 for each observation
def predict(X, w, b, thresh=0.5):
  proba = predict_proba(X, w, b)
  if w.ndim == 1:           # binary
    return (proba >= thresh).astype(int)
  else:                     # multiclass: pick class with highest probability
    return np.argmax(proba, axis=1)

  


#Compute the average binary cross-entropy loss
#X - numpy array - the feature matrix
#y - numpy array - true labels
#w - numpy array - weight vector
#b - float - a bias
#return - float - the average binary cross-entropy loss
def loss(X, y, w, b):
  pred = predict_proba(X, w, b)
  eps = 1e-12
  pred = np.clip(pred, eps, 1 - eps)
  if w.ndim == 1:     # binary cross-entropy
    return -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))
  else:               # categorical cross-entropy: y is one-hot (n, K)
    return -np.mean(np.sum(y * np.log(pred), axis=1))

#Compute the gradient of the loss with respect to the model weights and the bias term
#X - numpy array - the feature matrix
#y - numpy array - true labels
#w - numpy array - weight vector
#b - float - a bias
#return - (numpy array, float) - the gradients with respect to w and b
def gradients(X, y, w, b): #= mean of y - y_hat 
  # YOUR CODE HERE
  y_hat   = predict_proba(X, w, b) #generate predictions
  error = y_hat - y  #predc eror for each sample 
  # X.t features are rows and col are sampled observations and X.t @ error is the sum of the errors for each feature
  dw = X.T @ error/ len(y) #divided by number of samples to get the average error for each feature to be consistent with the loss function
  db = np.mean(error, axis=0)
  return dw, db #return the gradients with respect to w and b

#Learn model parameters w and b using gradient descent. Tracks and returns how loss changes from epoch to epoch
#X - numpy array - the feature matrix associated with the traning data
#y - numpy array - true labels
#eta - float - the learning rate, defaults to 0.1
#epochs - int - the number of times to consider the dataset
#batch_size - int - the number of examples to consider in each batch. if None, use all examples
#shuffle - bool - if true, shuffle the data at the start of each epoch
#return - (numpy array, float, [float]) - learned values for w and b along with a list of loss values for each epoch
def fit(X, y, eta=0.1, epochs=50, batch_size=None, shuffle=True, num_classes=None):
  n, d = X.shape

  if num_classes is None or num_classes == 2:  # binary
    w = np.zeros(d)
    b = 0.0
    y_fit = y
  else:                                         # multiclass: one-hot encode y, matrix W
    w = np.zeros((d, num_classes))
    b = np.zeros(num_classes)
    y_fit = np.eye(num_classes)[y]             # one-hot: (n, K)

  losses = []
  if batch_size is None: batch_size = n

  for _ in range(epochs):
    indices = np.arange(n)
    if shuffle: np.random.shuffle(indices)     # shuffle all indices once per epoch

    for start in range(0, n, batch_size):
      batch_idx = indices[start:start + batch_size]   # safe slicing handles last batch
      dw, db = gradients(X[batch_idx], y_fit[batch_idx], w, b)
      w = w - eta * dw
      b = b - eta * db

    losses.append(loss(X, y_fit, w, b)) # full-dataset loss after each epoch

  return w, b, losses

#Generate data based on a random logistic regression model for testing
#n - int - the number of observations
#d - int - the number of dimensions
#scale - float - larger values make the classification problem easier
#num_classes - int or None - None or 2 for binary; >2 for multiclass
#return - numpy array, numpy array, numpy array, float - randomly generated observations along with correct labels and the true underlying w and b
def generateData(n, d, scale=1, num_classes=None):
  X = np.random.normal(0, 1, size=(n, d))
  if num_classes is None or num_classes == 2:  # binary
    w = np.random.normal(0, 1, size=d) * scale
    b = np.random.uniform(-1, 1) * scale
    y = (np.random.uniform(0, 1, size=n) <= sigmoid(X @ w + b)).astype(int)
  else:                                         # multiclass: W is (d, K), b is (K,)
    w = np.random.normal(0, 1, size=(d, num_classes)) * scale
    b = np.random.uniform(-1, 1, size=num_classes) * scale
    probs = softmax(X @ w + b)                 # true class probabilities (n, K)
    y = np.array([np.random.choice(num_classes, p=probs[i]) for i in range(n)])

  return X, y, w, b



#An empty main where code can be added to test the functions above
if __name__=='__main__':

  learning_rates = [0.01, 0.05, 0.1]
  epochs = 100
  fig, axes = plt.subplots(1, 2, figsize=(14, 5))

  #Binary logistic regression example
  X, y, w_true, b_true = generateData(n=200, d=10, scale=1)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  for eta in learning_rates:
    w, b, losses = fit(X_train, y_train, eta=eta, epochs=epochs)
    axes[0].plot(losses, label=f'eta = {eta}')
  axes[0].set_title('Binary Logistic Regression')
  axes[0].set_xlabel('Epoch')
  axes[0].set_ylabel('Loss')
  axes[0].legend()

  preds = predict(X_test, w, b)
  accuracy = np.mean(preds == y_test)
  print(f"Binary accuracy (eta={learning_rates[-1]}): {accuracy:.3f}")

  #Multiclass logistic regression example
  K = 4  # number of classes
  X_m, y_m, w_true_m, b_true_m = generateData(n=400, d=10, scale=1, num_classes=K)
  X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_m, y_m, test_size=0.2, random_state=42)

  for eta in learning_rates:
    w_m, b_m, losses_m = fit(X_train_m, y_train_m, eta=eta, epochs=epochs, num_classes=K)
    axes[1].plot(losses_m, label=f'eta = {eta}')
  axes[1].set_title(f'Multiclass Softmax Regression ({K} classes)')
  axes[1].set_xlabel('Epoch')
  axes[1].set_ylabel('Loss')
  axes[1].legend()

  preds_m = predict(X_test_m, w_m, b_m)
  accuracy_m = np.mean(preds_m == y_test_m)
  print(f"Multiclass accuracy (eta={learning_rates[-1]}): {accuracy_m:.3f}")

  plt.tight_layout()
  plt.savefig('loss_vs_epoch.png')
  plt.show()


