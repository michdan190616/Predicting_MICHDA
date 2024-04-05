import numpy as np

def compute_loss(y, tx, w):

    """
    A function that takes as inputs:
    - y: the n-dimensional np.array of dependent variables
    - tx: the (n x d)-dimensional np.array of feature observations
    - w: the d-dimensional np.array of weights

    And gives as output:
    - loss: the quadratic loss computed on y,tx and w

    """
    
    err = y-tx.dot(w)

    loss = (1/(2*len(err)))*np.linalg.norm(err)**2

    return loss

def compute_gradient(y, tx, w):
    """
    A function that takes as inputs:
    - y: the n-dimensional np.array of dependent variables
    - tx: the (n x d)-dimensional np.array of feature observations
    - w: the d-dimensional np.array of weights

    And gives as output:
    -grad: the gradient of the Gradient Descent method computed on y,tx and w

    """

    N = len(y)
    
    err = y-tx.dot(w)

    grad = -(1/N)*(tx.T).dot(err)

    return grad

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    A function that takes as inputs:
    - y: the n-dimensional np.array of dependent variables
    - tx: the (n x d)-dimensional np.array of feature observations
    - initial_w: a d-dimensional np.array, initial guess on the optimal weights
    - max_iters: a type int variable, maximum numbers of iteration for the loop in the function
    - gamma: a type int variable, learning rate of the model

    And gives as output:
    - loss: the quadratic loss of the Gradient Descent method at the final iteration
    - w: the d-dimensional array of optimal weights

    """

    w = initial_w

    loss = compute_loss(y,tx, w)

    n_iter = 0

    while(n_iter<max_iters):
        
        grad = compute_gradient(y,tx,w)
        
        w = w-gamma*grad

        loss = compute_loss(y,tx, w)

        if (n_iter % 100 == 0):
            print(
                "GD iter. {bi}/{ti}: loss={l}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss
                )
            )

        n_iter += 1

    return w, loss

#############------------------------------#############

### implementation of mean_squared_error_sgd
def compute_stoch_gradient(y, tx, w, batch_size):
    """
    A function that takes as inputs:
    - y: the n-dimensional np.array of dependent variables
    - tx: the (n x d)-dimensional np.array of feature observations
    - w: the d-dimensional np.array of weights
    - batch_size: a type int variable, the dimension of the random batch used by the method

    And gives as output:
    -grad: the gradient of the Stochastic Gradient Descent method computed on y,tx and w

    """
    
    idx = np.random.randint(0, len(y), size=batch_size)

    grad = compute_gradient(y[idx], tx[idx, :], w)

    return grad

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, batch_size = 1):
    """
    A function that takes as inputs:
    - y: the n-dimensional np.array of dependent variables
    - tx: the (n x d)-dimensional np.array of feature observations
    - initial_w: a d-dimensional np.array, initial guess on the optimal weights
    - max_iters: a type int variable, maximum numbers of iteration for the loop in the function
    - gamma: a type int variable, learning rate of the model
    - batch_size: a type int variable, the dimension of the random batch used by the method. Set at 1

    And gives as output:
    - loss: the quadratic loss of the Stochastic Gradient Descent method at the final iteration
    - w: the d-dimensional array of optimal weights

    """

    w = initial_w

    loss = compute_loss(y,tx, w)

    n_iter = 0

    while(n_iter<max_iters):
        
        grad = compute_stoch_gradient(y,tx,w,batch_size)
        
        w = w-gamma*grad

        loss = compute_loss(y,tx, w)

        if (n_iter % 500 == 0):
            print(
                "GD iter. {bi}/{ti}: loss={l}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss
                )
            )

        n_iter += 1

    return w, loss

#############------------------------------#############

### implementation of least squares
def least_squares(y, tx):
    """
    A function that takes as inputs:
    - y: the n-dimensional np.array of dependent variables
    - tx: the (n x d)-dimensional np.array of feature observations

    And gives as output:
    - loss: the quadratic loss of the Least Squares method computed on y,tx and w
    - w: the d-dimensional np.array of optimal weights 

    """

    w = np.linalg.solve(((tx.T)@(tx)),(tx.T)@y)
    
    loss = compute_loss(y, tx, w)

    return w, loss

#############------------------------------#############

### implementation of ridge regression

def ridge_regression(y, tx, lambda_):
    """
    A function that takes as inputs:
    - y: the n-dimensional np.array of dependent variables
    - tx: the (n x d)-dimensional np.array of feature observations
    - lambda_: a type int variable, the regularization parameter used by Ridge Regression

    And gives as output:
    - loss: the quadratic loss of the Ridge Regression computed on y,tx and w
    - w: the d-dimensional np.array of optimal weights 

    """

    lambda_1 = lambda_*2*len(y)

    I = np.eye(tx.shape[1])

    w = np.linalg.solve(((tx.T)@tx + lambda_1*I),(tx.T)@y)

    loss = compute_loss(y,tx,w)

    return w, loss

#############------------------------------#############

### implementation of logistic regression

def logistic(x):
    """
    A function that takes as inputs:
    - x: a n-dimensional np.array 

    And gives as output:
    - sigmoid: the sigmoid function computed on x

    """
    sigmoid= np.exp(x)/(np.exp(x)+1)

    return sigmoid

def compute_logistic_loss(y, tx, w):
    """
    A function that takes as inputs:
    - y: the n-dimensional np.array of dependent variables
    - tx: the (n x d)-dimensional np.array of feature observations
    - w: the d-dimensional np.array of weights

    And gives as output:
    - loss: the loss of the Logistic Regression computed on y,tx and w

    """
    
    
    N = len(y)
    h = logistic(tx @ w)

    loss = (-1 / N) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))

    return loss

def compute_logistic_gradient(y, tx, w):
    """
    A function that takes as inputs:
    - y: the n-dimensional np.array of dependent variables
    - tx: the (n x d)-dimensional np.array of feature observations
    - w: the d-dimensional np.array of weights

    And gives as output:
    -grad: the gradient of the Logistic Regression method computed on y,tx and w

    """

    N = len(y)

    h = logistic(tx @ w)

    grad = (1 / N) * tx.T @ (h - y)

    return grad

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    A function that takes as inputs:
    - y: the n-dimensional np.array of dependent variables
    - tx: the (n x d)-dimensional np.array of feature observations
    - initial_w: a d-dimensional np.array, initial guess on the optimal weights
    - max_iters: a type int variable, maximum numbers of iteration for the loop in the function
    - gamma: a type int variable, learning rate of the model

    And gives as output:
    - loss: the loss of the Logistic Regression at the final iteration
    - w: the d-dimensional array of optimal weights

    """
    
    w = initial_w

    loss = compute_logistic_loss(y, tx, w)

    n_iter = 0

    while(n_iter<max_iters):
        
        grad = compute_logistic_gradient(y, tx, w)
        
        w = w-gamma*grad

        loss = compute_logistic_loss(y, tx, w)

        if (n_iter % 100 == 0):
            print(
                "GD iter. {bi}/{ti}: loss={l}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss
                )
            )

        n_iter += 1

    return w, loss

#############------------------------------##############

### implementation of regularized logistic regression

def compute_reg_logistic_loss(y, tx, w, lambda_):
    """
    A function that takes as inputs:
    - y: the n-dimensional np.array of dependent variables
    - tx: the (n x d)-dimensional np.array of feature observations
    - w: the d-dimensional np.array of weights
    - lambda_: a type int variable, regularization parameter used by Regularized Logistic Regression

    And gives as output:
    - loss: the loss of the Regularized Logistic Regression computed on y,tx and w

    """

    log_loss = compute_logistic_loss(y, tx, w)

    loss = log_loss + lambda_*np.linalg.norm(w)**2

    return loss

def compute_reg_logistic_gradient(y, tx, w, lambda_):
    """
    A function that takes as inputs:
    - y: the n-dimensional np.array of dependent variables
    - tx: the (n x d)-dimensional np.array of feature observations
    - w: the d-dimensional np.array of weights
    - lambda_: a type int variable, regularization parameter used by Regularized Logistic Regression

    And gives as output:
    - grad: the gradient of the Regularized Logistic Regression method computed on y,tx, w and lambda

    """

    N = len(y)

    log_grad = compute_logistic_gradient(y, tx, w)

    grad = log_grad + 2*lambda_*w

    return grad

def reg_logistic_regression(y, tx,lambda_ ,initial_w, max_iters, gamma ):
    """
    A function that takes as inputs:
    - y: the n-dimensional np.array of dependent variables
    - tx: the (n x d)-dimensional np.array of feature observations
    - initial_w: a d-dimensional np.array, initial guess on the optimal weights
    - max_iters: a type int variable, maximum numbers of iteration for the loop in the function
    - gamma: a type int variable, learning rate of the model
    - lambda_: a type int variable, regularization parameter used by Regularized Logistic Regression

    And gives as output:
    - loss: the loss of the Regularized Logistic Regression at the final iteration
    - w: the d-dimensional array of optimal weights

    """

    w = initial_w

    loss = compute_logistic_loss(y, tx, w)

    n_iter = 0

    while(n_iter<max_iters):
        
        grad = compute_reg_logistic_gradient(y, tx, w, lambda_)
        
        w = w-gamma*grad

        loss = compute_logistic_loss(y, tx, w)

        if (n_iter % 100 == 0):
            print(
                "GD iter. {bi}/{ti}: loss={l}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss
                )
            )

        n_iter += 1

    return w, loss






