from recombination import *
import matplotlib.pyplot as plt
from rp_cholesky import *
from scipy.stats import qmc
import goodpoints as gp
from goodpoints.jax.dtc import lsr
import jax.numpy as jnp
import jax
from functools import partial
from goodpoints.herding import herding
import cvxpy as cp
import numpy as np

def q_opt(K, indices):
    K = (K + K.T)/2
    n = K.shape[0]
    K = K + 1e-6 * np.eye(n)
    one_n = np.ones(n) / n
    
    # Compute relevant submatrix and vector
    K_sub = K[np.ix_(indices, indices)]
    one_n_K = one_n @ K  # Compute (1/n)^T K
    one_n_K_sub = one_n_K[indices]  # Restrict to relevant indices
    
    # Define the optimization variable
    w_sub = cp.Variable(len(indices))
    
    # Define the objective function
    objective = cp.quad_form(w_sub, K_sub) - 2 * one_n_K_sub @ w_sub
    
    # Constraints: w_sub >= 0, sum(w_sub) = 1
    constraints = [w_sub >= 0, cp.sum(w_sub) == 1]
    
    # Solve the optimization problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve()
    
    # Construct the full weight vector
    w = np.zeros(n)
    w[indices] = w_sub.value
    
    return w


n = 25000

d = 10

r_list = [10, 25, 50, 100, 250, 500, 1000]



X = np.random.uniform(0, 1, (n, d))
reg = 2 * d
sink_ker = reg * sink_kernel(X, reg)



error_mat = np.zeros((len(r_list), 5))

r = r_list[-1]

U, svs = nystrom_gauss(sink_ker, r, theta = 3)

sampler = qmc.Halton(d=d, scramble=True, optimization="random-cd")


Loss = gm.SamplesLoss("sinkhorn",
            cost = "SqDist(X,Y)",
            truncate = None,
            potentials = False,
            blur = np.sqrt(reg),
            backend = "online",
            scaling = 0.8,
            debias = True)



for j in range(len(r_list)):

    r = r_list[j]

    print(r)

    idx_star, w_star = recombination(np.arange(n), r - 1, U[:, :(r - 1)], svs[:(r - 1)], np.diag(sink_ker))

    w_vec = np.zeros(n)
    w_vec[idx_star] = w_star


    rng = np.random.default_rng()
    arr = np.arange(n)
    rng.shuffle(arr)

    random_samp = arr[:r]

    uni = np.ones(n)/n
    rand_uni = np.ones(r)/r

    w_star = q_opt(sink_ker, idx_star)


    Wass = Loss(tensor(uni), tensor(X), tensor(w_star[idx_star]), tensor(X[idx_star,:]))

    err = Wass.cpu().numpy()

    error_mat[j, 0] = err

    Wass = Loss(tensor(uni), tensor(X), tensor(rand_uni), tensor(X[random_samp,:]))

    err = Wass.cpu().numpy()


    error_mat[j, 1] = err


    halton_samp = sampler.random(n = r)

    Wass = Loss(tensor(uni), tensor(X), tensor(rand_uni), tensor(halton_samp))

    err = Wass.cpu().numpy()

    error_mat[j, 2] = err



    indices_herd = herding(X, r, sink_ker, unique = True)

    w_herd = q_opt(sink_ker, indices_herd)

    Wass = Loss(tensor(uni), tensor(X), tensor(w_herd[indices_herd]), tensor(X[indices_herd,:]))

    err = Wass.cpu().numpy()

    error_mat[j, 4] = err



np.savetxt("cube.npy", error_mat, delimiter = ",")
