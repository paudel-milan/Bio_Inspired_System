import numpy as np


expected_returns = np.array([0.12, 0.18, 0.15])  
cov_matrix = np.array([
    [0.0064, 0.0008, 0.0011],
    [0.0008, 0.0025, 0.0014],
    [0.0011, 0.0014, 0.0036]
])  
num_assets = len(expected_returns)
risk_free_rate = 0.03


num_particles = 30
num_iterations = 100
w_inertia = 0.5
c1 = 1.5
c2 = 1.5


particles = np.random.dirichlet(np.ones(num_assets), size=num_particles)
velocities = np.zeros_like(particles)

def sharpe_ratio(weights):
    weights = np.maximum(weights, 0)
    weights /= np.sum(weights)
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return (portfolio_return - risk_free_rate) / portfolio_vol


pbest = particles.copy()
pbest_scores = np.array([sharpe_ratio(p) for p in particles])
gbest = pbest[np.argmax(pbest_scores)]
gbest_score = np.max(pbest_scores)

print("Starting optimization...\n")

for i in range(num_iterations):
    for j in range(num_particles):
        r1, r2 = np.random.rand(), np.random.rand()
        velocities[j] = (
            w_inertia * velocities[j]
            + c1 * r1 * (pbest[j] - particles[j])
            + c2 * r2 * (gbest - particles[j])
        )
        particles[j] += velocities[j]
        particles[j] = np.maximum(particles[j], 0)
        particles[j] /= np.sum(particles[j])  

        score = sharpe_ratio(particles[j])
        if score > pbest_scores[j]:
            pbest_scores[j] = score
            pbest[j] = particles[j].copy()

    best_particle_idx = np.argmax(pbest_scores)
    if pbest_scores[best_particle_idx] > gbest_score:
        gbest_score = pbest_scores[best_particle_idx]
        gbest = pbest[best_particle_idx].copy()
    

    if (i + 1) % 10 == 0:
        print(f"Iteration {i+1}/{num_iterations}")
        print(f"Current Best Sharpe Ratio: {gbest_score:.4f}")
        print(f"Current Best Weights: {gbest}\n")

print("Optimization completed!\n")
print("Final Results:")
print("Optimal Portfolio Weights:")
for i, w in enumerate(gbest):
    print(f"Stock {i+1}: {w:.4f}")

portfolio_return = np.dot(gbest, expected_returns)
portfolio_vol = np.sqrt(np.dot(gbest.T, np.dot(cov_matrix, gbest)))
sharpe = (portfolio_return - risk_free_rate) / portfolio_vol

print(f"\nExpected Portfolio Return: {portfolio_return:.4f}")
print(f"Portfolio Risk (Volatility): {portfolio_vol:.4f}")
print(f"Sharpe Ratio: {sharpe:.4f}")
