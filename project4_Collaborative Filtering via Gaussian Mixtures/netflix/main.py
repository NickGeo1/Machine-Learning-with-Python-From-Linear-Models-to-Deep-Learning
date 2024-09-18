import numpy as np
import kmeans
import common
import naive_em
import em

#X = np.loadtxt("toy_data.txt")
X = np.loadtxt("netflix_incomplete.txt")

# TODO: Your code here
#for s in range(5):
gm, post = common.init(X, 12, 1)
final_gm, final_post, ll = em.run(X, gm, post)
print(f"LL for seed = {1} and K = {12}: {ll}")
X_gold = np.loadtxt("netflix_complete.txt")
X_pred = em.fill_matrix(X, final_gm)
print(X_pred)
print(f"RMSE: {common.rmse(X_pred, X_gold)}")
#common.plot(X, final_gm, final_post, f"Kmeans seed = {0} K = {k}")
