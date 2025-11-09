
# Data-Driven SKU Recommendation Engine with Synthetic Control Evaluation (Supply Chain)

## 📌 Project Overview
This project focuses on building an intelligent SKU (Stock Keeping Unit) recommendation system to maximize revenue by selecting optimal product combinations. It integrates reinforcement learning (Q-learning, SARSA, DQN, PPO), genetic algorithms, and bandit strategies with a Multinomial Logit (MNL) framework and simulated environments. The end goal is to recommend the best SKU combinations per store and evaluate uplift using synthetic control methods.

---

## 📁 Project Modules

### 1. **Reinforcement Learning Algorithms**
#### a. **Vanilla Q-Learning**
- Tabular method using discrete binary state encoding for SKU selection.
- Explores and exploits over product combinations with revenue-based rewards.

#### b. **SARSA (On-Policy Q-learning)**
- On-policy method for learning action-value functions.
- Updates Q-values using the next action chosen by the policy.
- Includes reward history tracking and episode-wise reward plots.

#### c. **Deep Q-Network (DQN)**
- Neural network-based agent to approximate Q-values.
- Uses experience replay and target network.
- Learns optimal SKU selection policy via epsilon-greedy exploration.

#### d. **PPO (Proximal Policy Optimization)**
- Actor-Critic based neural architecture with policy clipping.
- Trains on episodes with binary SKU combinations.
- Tracks best combinations and plots reward trajectories.

#### e. **Final PPO (Enhanced)**  
- Binary product selection using sigmoid outputs from actor network.
- Uses Monte Carlo advantage estimation with entropy regularization.
- Tracks best revenue and product combo across episodes.

---

### 2. **Multi-Armed Bandit (MAB) – ε-Greedy**
- Lightweight agent that selects products based on expected reward.
- Balances exploration and exploitation to estimate best SKU using simple reward rule (utility × share).

---

### 3. **Genetic Algorithm (GA)**
- Encodes SKU combinations as binary chromosomes.
- Uses crossover, mutation, and elitism for evolution.
- Optimizes revenue-based fitness function over generations.

---

## 📊 Dataset (Synthetic)
Each SKU contains:
- `utility`: A proxy for customer preference.
- `price`: Selling price of SKU.
- `volume`: Volume sold used to compute relative share.
- `share`: Normalized share based on volume.

All algorithms simulate on randomly initialized SKU states (`n_skus = 5`) to test logic, convergence, and learning dynamics.

---

## 🎯 Revenue Objective Function
For all agents and optimizers, the reward signal is computed using:

```python
revenue = np.sum(state * utility * price * share)
```

This ensures SKU combinations are chosen to maximize expected revenue under the constraints of SKU utility, price, and market share.

---

## 📈 Visualizations
- **Reward vs Episodes Plot**: SARSA, PPO, and final PPO include matplotlib plots showing learning trajectory.
- **Best Combination Tracking**: Final PPO and SARSA retain and log the best SKU combination based on maximum revenue.

---

## ✅ Key Highlights
- Multi-agent approach for evaluating different RL policies.
- Combination of stochastic methods (bandits, genetic algorithms) with neural agents (DQN, PPO).
- Realistic revenue formulation simulating business value optimization.
- Modular design allows swapping of agents, reward schemes, or datasets.

---

## 🔧 Dependencies
- `numpy`
- `pandas`
- `torch`
- `matplotlib`

Install via:

```bash
pip install numpy pandas torch matplotlib
```

---

## 📁 File Structure
```
sku_recommender/
│
├── vanilla_q_learning.py
├── sarsa_agent.py
├── deep_q_learning.py
├── ppo_agent.py
├── final_ppo_agent.py
├── genetic_algorithm.py
├── multi_armed_bandit.py
├── utils.py (optional helper utilities)
└── README.md
```

---

## 🧠 Application & Use Case
This system can be deployed in:
- Retail chains to optimize product placement and assortment.
- CPG brands to determine product combos per store cluster.
- Dynamic shelf planning engines with reinforcement optimization.

---

## 👤 Author
Abhishek Prithvi Teja Angadala

---

## 📬 Contact
Feel free to connect for enhancements, enterprise integration, or optimization discussions.

---
