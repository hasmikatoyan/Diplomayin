import numpy as np
import matplotlib.pyplot as plt


# 1. Target function

def target_fn(x):
    # օրինակ: sine + linear + cosine
    return np.sin(2*np.pi*x) + 0.3*(x-0.5) + 0.5*np.cos(3*x)


# 2. Generate 10000 points

N = 10000
x_all = np.random.rand(N,1)
y_all = target_fn(x_all).reshape(-1,1)

# Split: 6000 train / 4000 test
idx = np.random.permutation(N)
x_train = x_all[idx[:6000]]
y_train = y_all[idx[:6000]]
x_test  = x_all[idx[6000:]]
y_test  = y_all[idx[6000:]]


# 3. Simple MLP with 1 hidden layer

class MLP:
    def __init__(self, n_input=1, n_hidden=64, n_output=1):
        rng = np.random.RandomState(0)
        self.W1 = rng.randn(n_input, n_hidden) * np.sqrt(2/n_input)
        self.b1 = np.zeros((1,n_hidden))
        self.W2 = rng.randn(n_hidden, n_output) * np.sqrt(2/n_hidden)
        self.b2 = np.zeros((1,n_output))

    def forward(self, x):
        self.z1 = x.dot(self.W1) + self.b1
        self.a1 = np.tanh(self.z1)           # activation
        self.z2 = self.a1.dot(self.W2) + self.b2
        return self.z2

    def predict(self, x):
        return self.forward(x)

    def mse(self, y_pred, y_true):
        return np.mean((y_pred - y_true)**2)

    def step(self, xb, yb, lr=0.01):
        # forward
        y_pred = self.forward(xb)
        m = len(xb)
        dL = (2/m)*(y_pred - yb)              # gradient wrt output

        # W2, b2
        dW2 = self.a1.T.dot(dL)
        db2 = np.sum(dL, axis=0, keepdims=True)

        # backprop a1
        da1 = dL.dot(self.W2.T)
        dz1 = da1 * (1 - self.a1**2)          # tanh derivative

        dW1 = xb.T.dot(dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # update
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1


# 4. Train

model = MLP()
epochs = 500
batch_size = 64
lr = 0.01

for ep in range(epochs):
    perm = np.random.permutation(len(x_train))
    x_sh = x_train[perm]
    y_sh = y_train[perm]

    for i in range(0,len(x_train), batch_size):
        xb = x_sh[i:i+batch_size]
        yb = y_sh[i:i+batch_size]
        model.step(xb, yb, lr=lr)

    if (ep+1) % 100 == 0 or ep==0:
        pred_test = model.predict(x_test)
        print(f"Epoch {ep+1:3d}, Test MSE = {model.mse(pred_test, y_test):.6f}", flush=True)


# 5. Compute accuracy ±epsilon

epsilon = 0.05
pred_test = model.predict(x_test)
good = np.sum(np.abs(pred_test - y_test) < epsilon)
accuracy = good / len(y_test) * 100
print(f"\nGood predictions: {good}/4000   Accuracy = {accuracy:.2f}%")

# 6. Plot function vs approximation

xs_plot = np.linspace(0,1,500).reshape(-1,1)
ys_true = target_fn(xs_plot)
ys_pred = model.predict(xs_plot)

plt.figure(figsize=(8,5))
plt.plot(xs_plot, ys_true, label='Original function', linewidth=2)
plt.plot(xs_plot, ys_pred, label='MLP approximation', linestyle='--', linewidth=2)
plt.scatter(x_test, y_test, s=6, alpha=0.2, label='Test samples')
plt.legend()
plt.title('Function approximation with 1-hidden-layer MLP')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

