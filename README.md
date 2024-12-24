import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('/content/Advertising.csv')
X = df[['TV', 'radio', 'newspaper']]
Y = df['sales']

# Standardize the data
Y = np.array((Y - Y.mean()) / Y.std())
X = X.apply(lambda rec: (rec - rec.mean()) / rec.std(), axis=0)

# Initialize bias and weights
import random
def initialize(dim):
    b = random.random()
    theta = np.random.rand(dim)
    return b, theta

b, theta = initialize(3)
print("Bias: ", b, "Weights: ", theta)

# Predict function
def predict_Y(b, theta, X):
    return b + np.dot(X, theta)

Y_hat = predict_Y(b, theta, X)
print("Initial Predictions: ", Y_hat[:10])

# Cost function
def get_cost(Y, Y_hat):
    Y_resd = Y - Y_hat
    return np.sum(np.dot(Y_resd.T, Y_resd)) / len(Y)

# Update weights and bias
def update_theta(X, Y, Y_hat, b_0, theta_0, learning_rate):
    db = (np.sum(Y_hat - Y) * 2) / len(Y)
    dw = (np.dot((Y_hat - Y), X) * 2) / len(Y)
    b_1 = b_0 - learning_rate * db
    theta_1 = theta_0 - learning_rate * dw
    return b_1, theta_1

print("After initialization - Bias: ", b, "Weights: ", theta)
Y_hat = predict_Y(b, theta, X)
b, theta = update_theta(X, Y, Y_hat, b, theta, 0.01)
print("After first update - Bias: ", b, "Weights: ", theta)
print("Cost after update: ", get_cost(Y, Y_hat))

# Gradient Descent
def run_gradient_descent(X, Y, alpha, num_iterations):
    b, theta = initialize(X.shape[1])
    iter_num = 0
    gd_iterations_df = pd.DataFrame(columns=['iteration', 'cost'])
    for each_iter in range(num_iterations):
        Y_hat = predict_Y(b, theta, X)
        this_cost = get_cost(Y, Y_hat)
        b, theta = update_theta(X, Y, Y_hat, b, theta, alpha)
        # Save cost for plotting
        gd_iterations_df.loc[each_iter] = [each_iter, this_cost]
    return gd_iterations_df, b, theta

# Run gradient descent
gd_iterations_df, b, theta = run_gradient_descent(X, Y, alpha=0.001, num_iterations=200)
print(gd_iterations_df.head())

# Plot cost vs iterations
plt.plot(gd_iterations_df['iteration'], gd_iterations_df['cost'])
plt.xlabel("Number of iterations")
plt.ylabel("Cost or MSE")
plt.title("Cost vs Iterations")
plt.show()

# Test with different learning rates
alpha_df_1, b1, theta1 = run_gradient_descent(X, Y, alpha=0.01, num_iterations=2000)
alpha_df_2, b2, theta2 = run_gradient_descent(X, Y, alpha=0.001, num_iterations=2000)

# Plotting different learning rates
plt.plot(alpha_df_1['iteration'], alpha_df_1['cost'], label="alpha=0.01")
plt.plot(alpha_df_2['iteration'], alpha_df_2['cost'], label="alpha=0.001")
plt.legend()
plt.ylabel('Cost')
plt.xlabel('Number of iterations')
plt.title('Cost vs Iterations for different alpha values')
plt.show()





