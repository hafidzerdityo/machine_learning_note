import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the dataset
df = pd.DataFrame([
    {'tumor_size': 0, 'target': 0},
    {'tumor_size': 0.82, 'target': 0},
    {'tumor_size': 1, 'target': 0},
    {'tumor_size': 2, 'target': 0},
    {'tumor_size': 2.81, 'target': 0},
    {'tumor_size': 2.71, 'target': 1},
    {'tumor_size': 3, 'target': 1},
    {'tumor_size': 3.8, 'target': 1},
    {'tumor_size': 4.1, 'target': 1},
    {'tumor_size': 5.1, 'target': 1},
])

df = df.sort_values(by='tumor_size')

x_train = df[['tumor_size']].values
y_train = df['target'].values

# Initialize parameters
w = np.zeros(1)  # weight
b = 0             # bias
lr = 0.01         # learning rate
iterations = 50000

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Perform gradient descent
losses = []
fig, ax = plt.subplots()
line, = ax.plot([], [], color='red', lw=2)
ax.scatter(x_train, y_train, color='blue')

def update(frame):
    global w, b, losses
    for _ in range(1000):
        z = np.dot(x_train, w) + b
        y_pred = sigmoid(z)
        loss = -np.mean(y_train * np.log(y_pred) + (1 - y_train) * np.log(1 - y_pred))
        dw = np.dot(x_train.T, (y_pred - y_train)) / len(df)
        db = np.mean(y_pred - y_train)
        w -= lr * dw
        b -= lr * db
        losses.append(loss)
    line.set_data(x_train, sigmoid(np.dot(x_train, w) + b))
    ax.set_title(f'Iteration {frame * 1000}')
    return line,

ani = FuncAnimation(fig, update, frames=range(10), blit=True)
plt.show()
