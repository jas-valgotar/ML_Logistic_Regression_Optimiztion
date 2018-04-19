import math
import numpy as np
import matplotlib.pyplot as plt


# Plots Graph with Separating line on 0.5 based on given Coefficients  and Data X, Y
def plot_reg(X, Y, fitted_values):
    i = 0
    xp = []
    xn = []
    # Separating positive and negative data
    for x in X:
        if (Y[i] == 0):
            xp.append([x[1], x[2]])
        else:
            xn.append([x[1], x[2]])
        i += 1

    xp = np.asarray(xp)
    xn = np.asarray(xn)

    # Creating Boundary
    ex1 = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
    ex2 = -(fitted_values[1] * ex1 + fitted_values[0]) / fitted_values[2]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(xp[:, 0], xp[:, 1], s=30, c='green', marker="s", label='Positive')
    ax1.scatter(xn[:, 0], xn[:, 1], s=30, c='red', marker="o", label='Negative')
    plt.plot(ex1, ex2, color='black', label='decision boundary');
    plt.legend(loc='upper left')
    return plt


# Implementation of Sigmoid function
def logistic_func(theta, x):
    return float(1) / (1 + math.e**(-x.dot(theta)))


# returns Gradient of given theta, x, y
def gradient(theta, x, y):
    first_calc = logistic_func(theta, x) - y
    final_calc = first_calc.T.dot(x)
    return final_calc


# Function Returns Cost for given theta,X,Y
def cost_func(theta, x, y):
    log_func_v = logistic_func(theta,x)
    y = np.squeeze(y)
    step1 = y * np.log(log_func_v)
    step2 = (1-y) * np.log(1 - log_func_v)
    final = -step1 - step2
    return np.sum(final)


# Implementation of Gradient Descent Method
def grad_desc(theta_values, X, y, lr=.001, converge_change=1e-8):
    #theta_values=np.zeros(X[1].shape)
    #setup cost iter
    cost_iter = []
    cost = cost_func(theta_values, X, y)
    cost_iter.append([0, cost])
    change_cost = 100000
    i = 1
    while(change_cost > converge_change):
        old_cost = cost
        theta_values = theta_values - (lr * gradient(theta_values, X, y))
        cost = cost_func(theta_values, X, y)
        cost_iter.append([i, cost])
        change_cost = old_cost - cost
        i+=1
    return theta_values, np.array(cost_iter)


# Implementation of Finding Hessian Matrix
def hessian(theta, X):
    ans = np.zeros((len(X[0]),len(X[0])))
    for xi in X:
        xit = xi.reshape(len(xi),1)
        ans = ans + logistic_func(theta, xi)*(1-logistic_func(theta, xi))*xi*xit
    return ans


# Implementation of Newton's Method
def newtons(theta_values,X,Y,converge_change=1e-8):
    # setup cost iter
    cost_iter = []
    cost = cost_func(theta_values, X, Y)
    cost_iter.append([0, cost])
    change_cost = 100000.0
    i = 1
    while (change_cost > converge_change):
        old_cost = cost
        h = hessian(theta_values, X)
        theta_values = theta_values - np.dot(np.linalg.inv(h),gradient(theta_values,X,Y))
        cost = cost_func(theta_values, X, Y)
        cost_iter.append([i, cost])
        change_cost = old_cost - cost
        i += 1
    return theta_values, np.array(cost_iter)


# function used to Predict
def pred_values(theta, X, hard=True):
    pred_prob = logistic_func(theta, X)
    pred_value = np.where(pred_prob >= .5, 1, 0)
    if hard:
        return pred_value
    return pred_prob


X = np.genfromtxt('q1x.dat')
Y = np.genfromtxt('q1y.dat')
intercept = np.ones((X.shape[0], 1))
X = np.hstack((intercept, X))
betas = np.zeros(X[1].shape)


# Applying GD to optimize theta
fitted_values, cost_iter = grad_desc(betas, X, Y)
print "For Gradient Descent Method"
print "Intercept\tCoef1\t\tCoef2"
print(fitted_values)," Iteration To Converge", len(cost_iter)
# Plot Graph for GD
plt = plot_reg(X, Y, fitted_values)
plt.title("Boundary using Coefficient of Gradient Descent Method")
plt.savefig("GD Method")


# Applying NR Method for optimization
fitted_values, cost_iter = newtons(betas, X, Y)
print "For NR Method"
print "Intercept\tCoef1\t\tCoef2"
print(fitted_values)," Iteration To converge", len(cost_iter)
# Plot Graph for Newton's
plt = plot_reg(X, Y, fitted_values)
plt.title("Boundary using Coefficient of NR Method")
plt.savefig("NR Method")


# Prediction Of each data in X
predicted_y = pred_values(fitted_values, X)
print predicted_y
