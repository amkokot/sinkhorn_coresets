import math
import numpy as np
import copy
from sklearn.decomposition import TruncatedSVD
from linalg_algs import *
from sinkhorn_funcs import *




# Given k, eigen_funcs and eigen_values externally,
# the recombination function outputs a kernel quadrature formula

def recombination(
    pts_rec,  # random sample for recombination
    num_pts,  # number of quadrature nodes
    evecs,    # input desired eigenfunctions
    svs,      # input corresponding eigenvalues
    obj,      # input diagonal of the kernel matrix
    use_obj=True,  # whether or not using objective
):
    return rc_mercer(pts_rec, num_pts, evecs, svs, obj, use_obj=use_obj)


def rc_mercer(samp, s, evecs, svs, diag_mat, use_obj=True):
    X = evecs.T
    evs = svs
    obj = copy.deepcopy(diag_mat)
    sur_evs = np.reshape(np.sqrt(evs), (-1, 1))
    if use_obj:
        N = len(samp)
        rem = N - s * (N // s)
        for i in range(N//s):
            mat = X[:, s*i:s*(i+1)]
            mat = np.multiply(mat, sur_evs)
            obj[s*i:s*(i+1)] -= np.sum(mat**2, axis=0)
        if rem:
            mat = X[:, N-rem:N]
            mat = np.multiply(mat, sur_evs)
            obj[N-rem:N] -= np.sum(mat**2, axis=0)

    w_star, idx_star = Mod_Tchernychova_Lyons(
        X, obj, use_obj=use_obj)

    if use_obj:
        # final sparsification
        Xp = X[:, idx_star]
        Xp = np.append(Xp, np.ones((1, len(idx_star))), axis=0)
        _, _, w_null = np.linalg.svd(Xp)
        w_null = w_null[-1]
        if np.dot(obj[idx_star], w_null) < 0:
            w_null = -w_null

        lm = len(w_star)
        plis = w_null > 0
        alpha = np.zeros(lm)
        alpha[plis] = w_star[plis] / w_null[plis]
        idx = np.arange(lm)[plis]
        idx = idx[np.argmin(alpha[plis])]
        w_star = w_star-alpha[idx]*w_null
        w_star[idx] = 0.

        idx_ret = idx_star[w_star > 0]
        w_ret = w_star[w_star > 0]
        return idx_ret, w_ret

    else:
        return idx_star, w_star

# Mod_Tchernychova_Lyons is modification of Tcherynychova_Lyons from https://github.com/FraCose/Recombination_Random_Algos/blob/master/recombination.py


def Mod_Tchernychova_Lyons(X, obj=0, mu=0, use_obj=True, DEBUG=False):
    n, N = X.shape
    if use_obj:
        n += 1

    # tic = timeit.default_timer()

    number_of_sets = 2*(n+1)

    if np.all(obj == 0) or len(obj) != N:
        obj = np.zeros(N)
    if np.all(mu == 0) or len(mu) != N or np.any(mu < 0):
        mu = np.ones(N)/N

    idx_story = np.arange(N)
    idx_story = idx_story[mu != 0]
    remaining_points = len(idx_story)

    while True:

        if remaining_points <= n+1:
            idx_star = np.arange(len(mu))[mu > 0]
            w_star = mu[idx_star]
            # toc = timeit.default_timer()-tic
            return w_star, idx_star
            # return w_star, idx_star, X[idx_star], toc, ERR, np.nan, np.nan # original
        elif n+1 < remaining_points <= number_of_sets:
            X_mat = X[:, idx_story]
            if use_obj:
                X_mat = np.append(X_mat, np.reshape(
                    obj[idx_story], (1, -1)), axis=0)
            w_star, idx_star, x_star, _, ERR, _, _ = Tchernychova_Lyons_CAR(
                np.transpose(X_mat), np.copy(mu[idx_story]), DEBUG)
            idx_story = idx_story[idx_star]
            mu[:] = 0.
            mu[idx_story] = w_star
            idx_star = idx_story
            w_star = mu[mu > 0]
            # toc = timeit.default_timer()-tic
            return w_star, idx_star
            # return w_star, idx_star, x_star, toc, ERR, np.nan, np.nan

        # remaining points at the next step are = remaining_points/card*(n+1)

        # number of elements per set
        number_of_el = int(remaining_points/number_of_sets)
        # WHAT IF NUMBER OF EL == 0??????
        # IT SHOULD NOT GET TO THIS POINT GIVEN THAT AT THE END THERE IS A IF

        X_tmp = np.zeros((number_of_sets, n))
        # mu_tmp = np.empty(number_of_sets)

        idx = idx_story[:number_of_el*number_of_sets].reshape(number_of_el, -1)
        for i in range(number_of_el):
            X_mat = X[:, idx_story[i*number_of_sets:(i+1)*number_of_sets]]
            if use_obj:
                X_mat = np.append(X_mat, np.reshape(
                    obj[idx_story[i*number_of_sets:(i+1)*number_of_sets]], (1, -1)), axis=0)
            X_tmp += np.multiply(np.transpose(X_mat), mu[idx_story[i *
                                 number_of_sets:(i+1)*number_of_sets], np.newaxis])
        tot_weights = np.sum(mu[idx], 0)

        idx_last_part = idx_story[number_of_el*number_of_sets:]

        if len(idx_last_part):
            X_mat = X[:, idx_last_part]
            if use_obj:
                X_mat = np.append(X_mat, np.reshape(
                    obj[idx_last_part], (1, -1)), axis=0)
            X_tmp[-1] += np.multiply(np.transpose(X_mat),
                                     mu[idx_last_part, np.newaxis]).sum(axis=0)
            tot_weights[-1] += np.sum(mu[idx_last_part], 0)

        X_tmp = np.divide(X_tmp, tot_weights[np.newaxis].T)

        w_star, idx_star, _, _, ERR, _, _ = Tchernychova_Lyons_CAR(
            X_tmp, np.copy(tot_weights))

        idx_tomaintain = idx[:, idx_star].reshape(-1)
        idx_tocancel = np.ones(idx.shape[1]).astype(bool)
        idx_tocancel[idx_star] = 0
        idx_tocancel = idx[:, idx_tocancel].reshape(-1)

        mu[idx_tocancel] = 0.
        mu_tmp = np.multiply(mu[idx[:, idx_star]], w_star)
        mu_tmp = np.divide(mu_tmp, tot_weights[idx_star])
        mu[idx_tomaintain] = mu_tmp.reshape(-1)

        idx_tmp = idx_star == number_of_sets-1
        idx_tmp = np.arange(len(idx_tmp))[idx_tmp != 0]
        # if idx_star contains the last barycenter, whose set could have more points
        if len(idx_tmp) > 0:
            mu_tmp = np.multiply(mu[idx_last_part], w_star[idx_tmp])
            mu_tmp = np.divide(mu_tmp, tot_weights[idx_star[idx_tmp]])
            mu[idx_last_part] = mu_tmp
            idx_tomaintain = np.append(idx_tomaintain, idx_last_part)
        else:
            idx_tocancel = np.append(idx_tocancel, idx_last_part)
            mu[idx_last_part] = 0.

        idx_story = np.copy(idx_tomaintain)
        remaining_points = len(idx_story)
        # remaining_points = np.sum(mu>0)


# Tchernychova_Lyons_CAR is taken from https://github.com/FraCose/Recombination_Random_Algos/blob/master/recombination.py


def Tchernychova_Lyons_CAR(X, mu, DEBUG=False):
    # this functions reduce X from N points to n+1

    # com = np.sum(np.multiply(X,mu[np.newaxis].T),0)
    X = np.insert(X, 0, 1., axis=1)
    if np.isnan(X[0,-1]):
        X[:,-1] = 0
    N, n = X.shape
    U, Sigma, V = np.linalg.svd(X.T)
    # np.allclose(U @ np.diag(Sigma) @ V, X.T)
    U = np.append(U, np.zeros((n, N-n)), 1)
    Sigma = np.append(Sigma, np.zeros(N-n))
    Phi = V[-(N-n):, :].T
    cancelled = np.array([], dtype=int)

    for _ in range(N-n):

        #alpha = mu/Phi[:, 0]
        lm = len(mu)
        plis = Phi[:, 0] > 0
        alpha = np.zeros(lm)
        alpha[plis] = mu[plis] / Phi[plis, 0]
        idx = np.arange(lm)[plis]
        idx = idx[np.argmin(alpha[plis])]
        cancelled = np.append(cancelled, idx)
        mu[:] = mu-alpha[idx]*Phi[:, 0]
        mu[idx] = 0.

        if DEBUG and (not np.allclose(np.sum(mu), 1.)):
            # print("ERROR")
            print("sum ", np.sum(mu))

        Phi_tmp = Phi[:, 0]
        Phi = np.delete(Phi, 0, axis=1)
        Phi = Phi - np.matmul(Phi[idx, np.newaxis].T,
                              Phi_tmp[:, np.newaxis].T).T/Phi_tmp[idx]
        Phi[idx, :] = 0.

    w_star = mu[mu > 0]
    idx_star = np.arange(N)[mu > 0]
    return w_star, idx_star, np.nan, np.nan, 0., np.nan, np.nan
"""
n = 10000
d = 10
r = 100

X = np.random.uniform(0, 1, (n, d))

reg = 0.1

A = ot.bregman.empirical_sinkhorn(X, X, reg)

Nys = nystrom(A, r)

svs, U = nystrom_evec(A, r)

idx_star, w_star = recombination(np.arange(n), r, U, svs, np.diag(A))

# test to verify it worked

w_vec = np.zeros(n)
w_vec[idx_star] = w_star

check1 = U.T @ (np.ones(n)/n - w_vec)

D = np.diag(A - Nys)


check2 = np.dot(D, np.ones(n)/n - w_vec)

"""