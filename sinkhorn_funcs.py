import numpy as np
import cvxpy as cp
import ot
from sklearn.metrics.pairwise import euclidean_distances
import math
import scipy
import torch
import geomloss as gm
import pykeops
#from pykrylov.symmlq import SYMMLQ as KSolver
from minresQLP import MinresQLP
from scipy.sparse.linalg import cg
from goodpoints.jax.sliceable_points import SliceablePoints
import jax.numpy as jnp

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
tensor = lambda x: torch.tensor(np.array(x, copy=True), device=device, dtype=torch.float32)

# Compute entropy

def emp_entropy(P1, P2):
    logP = np.log(P2)
    logP[P2 == 0] = 0
    return sum(sum(P1 * logP))

# Compute distance between data

# euclidean_distances(X,Y)

# Compute EOT Loss

def EOT_loss(X,Y, reg, mu = 1, nu = 1):
    if type(mu) is float and type(nu) is float:
        P = ot.bregman.empirical_sinkhorn(X, Y, reg, numIterMax=20000)
    else:
        P = ot.bregman.empirical_sinkhorn(X, Y, float(reg), a = mu, b = nu, numIterMax=20000)
    C = euclidean_distances(X, Y)

    Pscale = P /nu
    Pscale = Pscale.T / mu

    Pscale = Pscale.T

    Pscale[mu == 0, :] = 0
    Pscale[:, nu == 0] = 0

    C = C ** 2
    return np.einsum('ij,ij->', P, C) + reg * emp_entropy(P, Pscale)

# Compute sinkhorn divergence between measures

def sinkhorn_loss(X,Y, reg, mu = 1, nu = 1):
    A = EOT_loss(X,Y, reg, mu, nu)
    B = EOT_loss(X, X, reg, mu, mu)
    C = EOT_loss(Y, Y, reg, nu, nu)
    return A - 0.5 * (B + C)




# Compute sinkhorn kernel

def sink_density(X):
    P = ot.bregman.empirical_sinkhorn(X, X, reg)
    n = X.shape[0]
    return P * n

def pot_to_mat(f, g, X, reg):
    n = X.shape[0]
    C = euclidean_distances(X, X) ** 2
    A = np.exp(-C/reg)
    B = f.reshape((n,1)) + g.reshape((1,n))
    A *= np.exp(B/reg)
    return (A + A.T)/2

def sink_kernel_dens(X, reg, mu):
    n = X.shape[0]
    Loss = gm.SamplesLoss("sinkhorn",
            cost = "SqDist(X,Y)",
            truncate = None,
            potentials = True,
            blur = np.sqrt(reg),
            backend = "online",
            scaling = 0.8,
            debias = False)

    Wass = Loss(tensor(mu), tensor(X), tensor(mu), tensor(X))

    F = Wass[0].cpu().numpy()

    A = pot_to_mat(F, F, X, reg)
    A /= n

    return A


def sink_kernel_func(X, reg):
    n = X.shape[0]

    Loss = gm.SamplesLoss("sinkhorn",
            cost="SqDist(X,Y)",
            truncate=None,
            potentials=True,
            blur=np.sqrt(reg),
            backend="online",
            scaling=0.8,
            debias=False)

    one = np.ones(n)
    Wass = Loss(tensor(one/n), tensor(X), tensor(one/n), tensor(X))
    F = np.exp(Wass[0].cpu().numpy() / reg).reshape((n, 1))


    def eval_F(x):
        """
        Computes the normalization function for input points x.
        Supports batch input: if x is shape (m, d), returns (m, 1).
        """
        x = jnp.asarray(x)  # Ensure x is a NumPy array
        if x.ndim == 1:
            x = x.reshape(1, -1)  # Ensure (1, d) shape if x is a single point

        y = jnp.asarray(X)  # Reference dataset
        m = x.shape[0]
        n = y.shape[0]

        val = jnp.zeros(m)

        for i in range(m):
            C_row = jnp.sum((x[i] - y) ** 2, axis=-1)  # Compute one row of C at a time
            A_row = jnp.exp(-C_row / reg)  # Compute corresponding A values
            val = val.at[i].set(jnp.sum(A_row * F) / n)  # Accumulate result

        return 1 / val


    def sink_ker(x, y):

        x = jnp.asarray(x)  # Ensure x is a NumPy array
        if x.ndim == 1:
            x = x.reshape((1, -1))
        if y.ndim == 1:
            y = y.reshape((1, - 1))
        # Compute normalization factors for each input set
        Fx = eval_F(x).reshape((-1,)) # Shape (n, 1)
        Fy = eval_F(y).reshape((-1,))  # Shape (1, m), transposed for broadcasting
        # Compute squared Euclidean distances between all pairs
        if x.shape[0] > 1 and y.shape[0] > 1:
            Cxy = jnp.sum((x - y) ** 2, axis=-1)
            #Cxy = jnp.diagonal(Cxy)
        else:
            Cxy = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
            Cxy = Cxy.reshape((-1,))
        out = Fx * Fy * jnp.exp(-Cxy/reg)
        return out  # Shape (n, m)

    return sink_ker


def sink_kernel(X, reg):
    n = X.shape[0]
    Loss = gm.SamplesLoss("sinkhorn", 
        cost = "SqDist(X,Y)", 
        truncate = None, 
        potentials = True, 
        blur = np.sqrt(reg),
        backend = "online",
        scaling = 0.8,
        debias = False)


    one = np.ones(n)

    Wass = Loss(tensor(one/n), tensor(X), tensor(one/n), tensor(X))

    F = Wass[0].cpu().numpy()

    A = pot_to_mat(F, F, X, reg)
    A /= n
    #eig_vec = np.ones(n)/np.sqrt(n)

    #num_err = np.dot(eig_vec, A @ eig_vec)

    #eig_mat = np.outer(eig_vec, eig_vec)

    #A += - num_err * eig_mat
    '''
    #S, E = np.linalg.eig(A)
    #print(S)
    #print("Row Sums: ", np.sum(A, axis = 0))
    I = np.eye(n)
    Asq = np.linalg.matrix_power(A, 2)

    Asq = (Asq + Asq.T)/2

    Denom = I - Asq

    eig_vec = np.ones(n)/np.sqrt(n)

    num_err = np.dot(eig_vec, Denom @ eig_vec)

    eig_mat = np.outer(eig_vec, eig_vec)

    Denom += - num_err * eig_mat

    #S, E = np.linalg.eig(Denom)
    #print(S)

    #K = scipy.linalg.solve(Denom, A * n + 2 * n * Asq, assume_a='pos')

    Num = A * n + 2 * n * Asq

    K = np.zeros((n,n))

    for i in range(1):
        K[i,:] = MinresQLP(Denom, Num[i, :], 10 ** (-7), 2 * n)[0].reshape((n,))
    K = (K + K.T) / 2
    '''
    return A #K




def estimate_diag(A, num_samples=100):
    """
    Estimate the diagonal of B^{-1}A efficiently using Hutchinson's method,
    where B = I - A^2.

    Parameters:
    A (ndarray): The psd matrix A.
    num_samples (int): Number of random vectors for estimation.

    Returns:
    ndarray: Estimated diagonal of B^{-1}A.
    """
    n = A.shape[0]
    B = np.eye(n) - np.dot(A, A)  # Construct B (consider sparse if applicable)

    epsilon = 1e-7  # Small regularization parameter
    B = B + epsilon * np.eye(B.shape[0])

    diagonal_estimate = np.zeros(n)

    for _ in range(num_samples):
        # Generate a random vector with entries ±1 (Rademacher distribution)
        v = np.random.choice([-1, 1], size=n)

        # Solve for B^{-1}v using Conjugate Gradient
        B_inv_v, _ = cg(B, v)

        # Estimate the diagonal contribution from this vector
        diagonal_estimate += v * (np.dot(A, B_inv_v))

    # Average over the number of samples
    diagonal_estimate /= num_samples

    return diagonal_estimate




def diff_maps(X, reg):
    C = euclidean_distances(X, X) ** 2
    W = np.exp(-C/reg)
    D = np.sum(W, axis = 1)
    n = C.shape[0]
    return   (np.eye(n) - ((D ** (-1)) * W.T).T) / reg

def get_potential(X,Y,reg, mu = 1, nu = 1, sym = 0):
    if type(mu) is int and type(nu) is int:
        A = ot.bregman.empirical_sinkhorn(X, Y, reg, numIterMax=20000)
        n = A.shape[0]
        mu = np.ones(n)/n
        nu = np.ones(n)/n
    else:
        A = ot.bregman.empirical_sinkhorn(X, Y, reg, a= mu, b = nu, numIterMax=20000)

    C = euclidean_distances(X, Y) ** 2
    K = np.exp(-C / reg)

    potentials = A / K

    potentials = np.linalg.svd(potentials)

    p1 = potentials.U[:, 0]
    p2 = potentials.Vh[0, :]

    p1 = np.abs(p1)
    p2 = np.abs(p2)

    alpha = np.sqrt(A[0, 0] / (p1[0] * p2[0]))

    p1 = alpha * p1
    p2 = alpha * p2

    p1 /= mu
    p2 /= nu

    p1 = np.log(p1)/reg
    p2 = np.log(p2)/reg

    if sym:
        return p1
    else:
        return (p1, p2)

def sink_potentials(X, Y, reg, mu, nu):
    p = get_potential(Y, Y, reg, sym=1)
    p1 = get_potential(X, X, reg, mu=mu, nu=mu, sym=1)
    p2 = get_potential(X, Y, reg, mu=mu, nu=nu)
    p3 = p2[1]
    p2 = p2[0]

    return (p3 - p, p2 - p1)

def sink_potentials_2(X, Y, reg, mu, nu):
    p = get_potential(Y, Y, reg, sym=1)
    p1 = get_potential(X, X, reg, mu=mu, nu=mu, sym=1)
    p2 = get_potential(X, Y, reg, mu=mu, nu=nu)
    p3 = p2[1]
    p2 = p2[0]

    return (p, p1, p2, p3)




