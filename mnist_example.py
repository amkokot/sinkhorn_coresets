from recombination import *
from sklearn.preprocessing import StandardScaler
from rp_cholesky import *
import sys
from sklearn.decomposition import PCA

job_id = int(sys.argv[1])
data_set = scipy.io.loadmat('mnist-original.mat')
data = data_set['data']
label = data_set['label']
'''
# Step 1: Identify rows with any NaN values
nan_cols = np.isnan(data).any(axis=0)

# Step 2: Select rows that do not contain NaN values
data = data[:, ~nan_cols]
label = label[:, ~nan_cols]
'''
n = 70000

#r_list = [500, 1000, 1500, 2000]

#r_list = [100, 1000, 2500]

#r = r_list[job_id % len(r_list)]

r = 100

#reg_list = [15000, 20000, 25000, 30000, 40000, 50000, 100000]


#reg_list = [500, 1000, 1500, 2000, 2500, 5000]

#reg = reg_list[job_id % len(reg_list)]

reg = 15000

rng = np.random.default_rng()
arr = np.arange(70000)
rng.shuffle(arr)

random_samp = arr[:n]



label = label.reshape((70000))
label = label[random_samp]

X = data[:, random_samp]
X = X.T

X = X.astype("float")

X_init = X
#X /= np.amax(X)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X = X_scaled
#k = 25
#pca = PCA(n_components=k)
#X = pca.fit_transform(X_scaled)
#print("X: ", X)

error_mat = np.zeros((6,))

sink_ker = sink_kernel(X, reg)

U, svs = nystrom_gauss(sink_ker, r, theta = 5)

#diag = estimate_diag(sink_ker, num_samples=100)

#idx_star, w_star = recombination(np.arange(n), r - 1, U[:, 1:r], svs[1:r]/(1-svs[1:r]**2), diag)

idx_star, w_star = recombination(np.arange(n), r - 1, U[:, 1:r], svs[1:r], np.diagonal(sink_ker))

# test to verify it worked

w_vec = np.zeros(n)
w_vec[idx_star] = w_star



subset_label = label[idx_star]

max_num = max(label)
min_num = min(label)

rng = np.random.default_rng()
arr = np.arange(n)
rng.shuffle(arr)

random_samp = arr[:r]

rand_label = label[random_samp]

prop1 = np.zeros((int(max_num+1 - min_num)))
prop2 = np.zeros((int(max_num+1 - min_num)))
prop3 = np.zeros((int(max_num+1 - min_num)))


for i in range(int(min_num), int(max_num+1)):
    prop1[i-1] = sum(label[:n] == i)/n
    prop2[i-1] = sum(w_vec[label[:n] == i])
    prop3[i-1] = sum(rand_label == i) / r

error_mat[0] = sum(abs(prop1 - prop2))
error_mat[1] = sum(abs(prop1 - prop3))

Loss = gm.SamplesLoss("sinkhorn",
            cost = "SqDist(X,Y)",
            truncate = None,
            potentials = False,
            blur = np.sqrt(reg),
            backend = "online",
            scaling = 0.8,
            debias = True)

uni = np.ones(n)/n
rand_uni = np.zeros(n)
rand_uni[random_samp] = 1/r

error_mat[2] = np.dot(w_vec, sink_ker @ w_vec)/2
error_mat[3] = np.dot(rand_uni, sink_ker @ rand_uni)/2


Wass = Loss(tensor(uni), tensor(X), tensor(w_star), tensor(X[idx_star,:]))

err = Wass.cpu().numpy()

error_mat[4] = err


Wass = Loss(tensor(uni), tensor(X), tensor(np.ones(r)/r), tensor(X[random_samp,:]))

err = Wass.cpu().numpy()

error_mat[5] = err



#np.save("out/mnist/mnist_og_fulldata_reps_%d.npy" % job_id, error_mat)
np.save("out/mnist/mnist_samp_100_%d.npy" % job_id, np.hstack((X_init[idx_star,:], label[idx_star].reshape((r,1)), w_star.reshape((r,1)))))

