from recombination import *
import matplotlib.pyplot as plt
from rp_cholesky import *
import sys



job_id = int(sys.argv[1])

n_list = [5000, 10000, 15000, 25000]

d_list = [2, 5, 10]

r_list = [10, 25, 50, 100, 250, 500, 1000]

#reps = 1000

#job_id += 1000
#job_id += 2000

n = n_list[job_id % 4]
d = d_list[job_id % 3]


X = np.random.normal(0, 1, (n, d))
reg = 2 * d
sink_ker = reg * sink_kernel(X, reg)

error_mat = np.zeros((len(r_list), 4))

r = r_list[-1]

U, svs = nystrom_gauss(sink_ker, r, theta = 3)


for j in range(len(r_list)):

    r = r_list[j]

    idx_star, w_star = recombination(np.arange(n), r - 1, U[:, :(r - 1)], svs[:(r - 1)], np.diag(sink_ker))

    w_vec = np.zeros(n)
    w_vec[idx_star] = w_star


    rng = np.random.default_rng()
    arr = np.arange(n)
    rng.shuffle(arr)

    random_samp = arr[:r]

    uni = np.ones(n)/n
    rand_uni = np.zeros(n)
    rand_uni[random_samp] = 1/r

    error_mat[j,0] = np.dot((w_vec - uni), sink_ker @ (w_vec - uni))/2
    error_mat[j,1] = np.dot((rand_uni - uni), sink_ker @ (rand_uni - uni))/2

    Loss = gm.SamplesLoss("sinkhorn",
            cost = "SqDist(X,Y)",
            truncate = None,
            potentials = False,
            blur = np.sqrt(reg),
            backend = "online",
            scaling = 0.8,
            debias = True)

    Wass = Loss(tensor(uni), tensor(X), tensor(w_star), tensor(X[idx_star,:]))

    err = Wass.cpu().numpy()

    error_mat[j, 2] = err


    Wass = Loss(tensor(uni), tensor(X), tensor(np.ones(r)/r), tensor(X[random_samp,:]))

    err = Wass.cpu().numpy()

    error_mat[j, 3] = err




np.savetxt("out/gauss/gauss_1_kern_%d.npy" % job_id, error_mat, delimiter = ",")
