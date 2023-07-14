import numpy as np

def HT( v, k ):
    t = np.zeros_like( v )
    if k < 1:
        return t
    else:
        ind = np.argsort( abs( v ) )[ -k: ]
        t[ ind ] = v[ ind ]
        return t
################################
# Non Editable Region Starting #
################################
def my_fit( X_trn, y_trn ):
################################
#  Non Editable Region Ending  #
################################
    # Compute the number of samples (n) and the number of features (D)
    n, D = X_trn.shape

    # Set the learning rate, initial weights, maximum iterations, and convergence tolerance
    lr = 1.266206144716182
    model = np.linalg.lstsq(X_trn, y_trn, rcond=None)[0]  # Initialize w0 by solving the least squares problem
    S = 512  # Maximum number of non-zero elements in the weight vector
    max_iter = 40
    eps = 1e-6

    for x in range(max_iter):

        # Compute the update step using gradient descent
        w_dash = model - lr * np.dot(X_trn.T, np.dot(X_trn, model) - y_trn) / n

        # Apply hard thresholding to obtain the new weight vector
        w_new = HT(w_dash, S)

        # Compute active_indices, input_data_active, and weights_active using a for loop
        active_indices = []
        for i in range(len(w_new)):
            if w_new[i] != 0:
                active_indices.append(i)
        X_active = X_trn[:, active_indices]  # Prepare a sub-matrix with active columns of X
        w_active = np.linalg.lstsq(X_active, y_trn, rcond=None)[0]  # Solve the least squares problem

        # Update weights_new using the solution of the least squares problem
        w_new[active_indices] = w_active

        if np.linalg.norm(w_new - model) < eps:  # Check convergence criterion
            break

        model = w_new  # Update the weight vector

    return model



	# Use this method to train your model using training CRPs
	# Youe method should return a 2048-dimensional vector that is 512-sparse
	# No bias term allowed -- return just a single 2048-dim vector as output
	# If the vector your return is not 512-sparse, it will be sparsified using hard-thresholding
	# Return the trained model

