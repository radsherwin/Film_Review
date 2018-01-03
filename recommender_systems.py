import numpy as np
import scipy.io as io
import pandas as pd
from scipy.optimize import minimize
import io as iosys

def main():
    Y = pd.read_csv('./YMatrix_test.csv').as_matrix()
    R = pd.read_csv('./RMatrix_test.csv').as_matrix()
    R = R.astype(bool)
    Ynorm, Ymean = normalize_ratings(Y, R)

    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = 19

    X = np.random.rand(num_movies, num_features)
    Theta = np.random.rand(num_users, num_features)
    initial_parameters = np.hstack((X.T.flatten(), Theta.T.flatten()))
    lambda_var = 10

    costFunc = lambda p: cost_function(p, Ynorm, R, num_users, num_movies, num_features, lambda_var)[0]
    gradFunc = lambda p: cost_function(p, Ynorm, R, num_users, num_movies, num_features, lambda_var)[1]

    result = minimize(costFunc, initial_parameters, method='CG', jac=gradFunc,
                      options={'disp': True, 'maxiter': 1000.0})
    theta = result.x
    cost = result.fun

    # Unfold the returned theta back into U and W
    X = theta[:num_movies * num_features].reshape(num_movies, num_features)
    Theta = theta[num_movies * num_features:].reshape(num_users, num_features)

    p = X.dot(Theta.T)
    my_predictions = p[:, 0] + Ymean

    movie_csv = pd.read_csv('./data/movie_rating_data/test_data.csv')
    movie_id = np.array(movie_csv['movieId'], dtype=pd.Series)
    tmdb_id = np.array(movie_csv['tmdbId'], dtype=pd.Series)

    pre = np.array([[idx, p] for idx, p in enumerate(my_predictions)])
    post = pre[pre[:, 1].argsort()[::-1]]
    r = post[:, 1]
    ix = post[:, 0]


    print('\nTop recommendations for you:')
    for i in range(91):
        j = int(ix[i])
        print('Predicting rating %.1f for movie %s\n' % (my_predictions[j], tmdb_id[j]), movie_id[j])

def normalize_ratings(Y, R):
    m, n = Y.shape
    Ymean = np.zeros(m)
    Ynorm = np.zeros(Y.shape)

    for i in range(n):
        idx = (R[i,:]==1).nonzero()[0]
        if len(idx):
            Ymean[i] = np.mean(Y[i, idx])
            Ynorm[i, idx] = Y[i, idx] - Ymean[i]
        else:
            Ymean[i] = 0.0
            Ynorm[i,idx] = 0.0

    return Ynorm, Ymean

def cost_function(params, Y, R, num_users, num_movies, num_features, lambda_var):
    # Unfold the U and W matrices from params
    X = np.array(params[:num_movies * num_features]).reshape(num_features, num_movies).T.copy()
    Theta = np.array(params[num_movies * num_features:]).reshape(num_features, num_users).T.copy()

    J = 0
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)


    squared_error = np.power(np.dot(X, Theta.T) - Y, 2)
    # for cost function, sum only i,j for which R(i,j)=1
    J = (1 / 2.) * np.sum(squared_error * R)

    X_grad = np.dot((np.dot(X, Theta.T) - Y) * R, Theta)

    Theta_grad = np.dot(((np.dot(X, Theta.T) - Y) * R).T, X)

    J = J + (lambda_var / 2.) * (np.sum(np.power(Theta, 2)) + np.sum(np.power(X, 2)))

    X_grad = X_grad + lambda_var * X
    Theta_grad = Theta_grad + lambda_var * Theta

    grad = np.hstack((X_grad.T.flatten(), Theta_grad.T.flatten()))
    return J, grad


if __name__ == '__main__':
    main()