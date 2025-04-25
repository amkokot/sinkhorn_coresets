from recombination import *
import matplotlib.pyplot as plt
from rp_cholesky import *

n = 25000

r = 16

k = 8

d = 2

reg = 0.5

eps = 1


mu = np.array([[3, 3], [3, -3], [-3,3], [-3,-3], [0, 6], [6, 0], [-6, 0], [0, -6]])


draws = np.random.multinomial(n, np.ones(k)/k)


X = np.zeros((n, 2))
tot = 0

for i in range(k):
    X[tot:(tot + draws[i]),:] = np.random.multivariate_normal(mu[i], np.eye(2), draws[i])
    tot += draws[i]


sink_ker = reg * sink_kernel(X, reg)

U, svs = nystrom_gauss(sink_ker, r, theta=10)

idx_star, w_star = recombination(np.arange(n), r-1, U[:,:(r-1)], svs[:(r-1)], np.diag(sink_ker))

# test to verify it worked

w_vec = np.zeros(n)
w_vec[idx_star] = w_star


rng = np.random.default_rng()
arr = np.arange(n)
rng.shuffle(arr)

random_samp = arr[:r]


uni = np.ones(n)/n
rand_uni = np.zeros(n)
rand_uni[random_samp] = 1/r


coreset = X[idx_star, :]


np.savetxt("gauss_nodes.npy", coreset, delimiter = ",")
np.savetxt("gauss_weights.npy", w_star, delimiter = ",")
