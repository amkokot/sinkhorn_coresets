import numpy as np
import numpy.linalg

import scipy


def nystrom(A, r):
    n = A.shape[0]
    test_mat = np.random.normal(0, 1, (n, 2 * r))

    test_mat, _ = np.linalg.qr(test_mat)

    nu = np.sqrt(n) * max(np.linalg.norm(A), 1) * 2.2 *  10 ** (-7)

    Y = A @ test_mat + nu * test_mat

    B = test_mat.T @ Y

    C = numpy.linalg.cholesky((B + B.T)/2)

    Cinv = scipy.linalg.solve_triangular(C, np.eye(2*r), lower="True")

    E = Y @ Cinv

    U, D, _ = np.linalg.svd(E)

    U1, D1, _ = np.linalg.svd(A)

    U = U[:, :r]
    D = np.diag(D[:r] ** 2) - nu * np.eye(r)

    D = np.diag(D)
    D = np.sqrt(D)

    return U @ D @ U.T, np.diag(D), U




def nystrom_evec(A, r):
    n = A.shape[0]
    test_mat = np.random.normal(0, 1, (n, 2 * r))
    nu = max(np.linalg.norm(A), 1) * 2.2 *  10 ** (-3)
    #nu = 0

    Y = A @ test_mat + nu * test_mat

    B = test_mat.T @ Y

    C = numpy.linalg.cholesky((B + B.T) / 2)

    Cinv = scipy.linalg.solve_triangular(C, np.eye(2*r), lower="True")

    E = Y @ Cinv

    U, D, _ = np.linalg.svd(E)

    U = U[:, :r]

    return D[:r] ** 2 - nu * np.ones(r), U




def pivoted_cholesky(A, rank=None, tol=1e-10):
    """
    Performs a pivoted Cholesky decomposition of a symmetric positive semidefinite matrix A.
    
    Parameters:
    A (np.ndarray): The symmetric positive semidefinite matrix.
    rank (int): Rank of the approximation (number of iterations). If None, iterate until tol is reached.
    tol (float): Tolerance for stopping criterion based on the residual norm.
    
    Returns:
    L (np.ndarray): The lower triangular matrix from the pivoted Cholesky decomposition.
    pivots (np.ndarray): The pivot indices used during decomposition.
    """
    n = A.shape[0]
    if rank is None:
        rank = n
    
    L = np.zeros((n, rank))
    pivots = np.arange(n)
    residual = A.copy()
    diag_residual = np.diag(residual)
    
    for k in range(rank):
        # Find the pivot index: the largest diagonal element in the residual
        pivot_index = np.argmax(diag_residual)
        
        #if diag_residual[pivot_index] < tol:
            # Stop if the largest diagonal is below the tolerance
        #    break
        
        # Swap rows and columns to move the pivot element to the current position
        if pivot_index != k:
            A[[k, pivot_index]] = A[[pivot_index, k]]
            A[:, [k, pivot_index]] = A[:, [pivot_index, k]]
            pivots[[k, pivot_index]] = pivots[[pivot_index, k]]
            L[[k, pivot_index]] = L[[pivot_index, k]]
        
        # Cholesky update
        L[k, k] = np.sqrt(diag_residual[pivot_index])
        L[(k+1):, k] = (A[(k+1):, k] - L[(k+1):, :k] @ L[k, :k]) / L[k, k]
        
        # Update the residual matrix
        residual = A - L @ L.T
        diag_residual = np.diag(residual)
    
    return L[:, :k+1], pivots



def modify_eigenvalues(A):
    """
    Performs spectral decomposition of a symmetric matrix A,
    sets negative eigenvalues to 0, and reconstructs a new matrix B.

    Args:
        A: Symmetric matrix (n x n).

    Returns:
        B: Matrix with the same eigenvectors as A but non-negative eigenvalues.
    """
    # Step 1: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    # Step 2: Set negative eigenvalues to zero
    eigenvalues_modified = np.where(eigenvalues < 0, 0, eigenvalues)

    # Step 3: Reconstruct the matrix B with modified eigenvalues
    B = eigenvectors @ np.diag(eigenvalues_modified) @ eigenvectors.T

    return B

def nystrom_gauss(A, r, theta = 2, nu = 10 ** (-4)):
    n = A.shape[0]
    d = A.shape[1]
    Omega = np.random.normal(0, 1, (n, theta*r))
    Q, _ = np.linalg.qr(Omega)
    
    #S, E = np.linalg.eig(A)
    #print(S)
    #print(E[S<0, :])
    Y = A @ Q + nu * Q

    B = Q.T @ Y

    B = (B + B.T)/2
    
    #B = modify_eigenvalues(B)
    #S, _ = np.linalg.eig(B)
    #`print(S)

    #B = B + nu * np.eye(theta * r)

    #C, _ = pivoted_cholesky(B, rank = theta * r)
    C = np.linalg.cholesky(B)
    Conj = np.linalg.solve(C.T, Y.T)

    Conj = Conj.T

    U, S, Vh = np.linalg.svd(Conj)

    return U, S - nu





