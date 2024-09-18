import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm




# Initialize parameters
mu1_hat, sigma1_hat = 6, 1
mu2_hat, sigma2_hat = 7, 2
pi1_hat, pi2_hat = 0.5, 0.5

X = np.array([-1, 0, 4, 5, 6])

# Perform EM algorithm for 20 epochs
num_epochs = 20
log_likelihoods = []
mu1 = []
mu2 = []
s1 = []
s2 = []
p1 = []
p2 = []

for epoch in range(num_epochs):
    # E-step: Compute responsibilities
    gamma1 = pi1_hat * norm.pdf(X, mu1_hat, sigma1_hat)
    gamma2 = pi2_hat * norm.pdf(X, mu2_hat, sigma2_hat)
    total = gamma1 + gamma2
    gamma1 /= total
    gamma2 /= total

    # M-step: Update parameters
    mu1_hat = np.sum(gamma1 * X) / np.sum(gamma1)
    mu2_hat = np.sum(gamma2 * X) / np.sum(gamma2)
    sigma1_hat = np.sqrt(np.sum(gamma1 * (X - mu1_hat) ** 2) / np.sum(gamma1))
    sigma2_hat = np.sqrt(np.sum(gamma2 * (X - mu2_hat) ** 2) / np.sum(gamma2))
    pi1_hat = np.mean(gamma1)
    pi2_hat = np.mean(gamma2)

    # Compute log-likelihood
    log_likelihood = np.sum(np.log(pi1_hat * norm.pdf(X, mu1_hat, sigma1_hat)
                                   + pi2_hat * norm.pdf(X, mu2_hat, sigma2_hat)))
    log_likelihoods.append(log_likelihood)

    #store params
    mu1.append(mu1_hat)
    mu2.append(mu2_hat)
    s1.append(sigma1_hat)
    s2.append(sigma2_hat)
    p1.append(pi1_hat)
    p2.append(pi2_hat)

print(np.array(mu1))
print(np.array(mu2))
print(np.array(s1)**2)
print(np.array(s2)**2)
print(np.array(p1))
print(np.array(p2))

# Plot log-likelihood values over epochs
plt.plot(range(1, num_epochs + 1), log_likelihoods)
plt.xlabel('Epoch')
plt.ylabel('Log-Likelihood')
plt.title('Log-Likelihood vs. Epoch')
plt.show()
