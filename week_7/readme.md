# Week 7 Resources & Task: Reinforcement Learning for Fantasy Team Selection (Dream11)

Welcome to **Week 7**, where we dive into **Reinforcement Learning (RL)** applied to tabular data—inspired by the paper [_Optimizing Fantasy Sports Team Selection with Deep Reinforcement Learning_](https://arxiv.org/html/2412.19215v1).

---

## Resources

### 1. Core Reading

- **[Research Paper: Optimizing Fantasy Sports Team Selection with Deep RL](https://arxiv.org/html/2412.19215v1)**  
  This paper is now the setup to the remainder for our project as we move from CI/CD for MLOps to instead completing a full-fledged exploration of MLOps with as many models as we can.

---

### 2. Reinforcement Learning Foundations
- **[Intro to Q-Learning – GeeksForGeeks](https://www.geeksforgeeks.org/q-learning-in-python/)**  
  Simple implementation of Q-Learning for discrete environments.

- **[RL Course by David Silver (UCL / DeepMind)](https://www.davidsilver.uk/teaching/)**  
  Especially useful: Lecture 2 (MDPs), Lecture 4 (Planning by Dynamic Programming), Lecture 6 (Value Function Approximation).

- **[Blog on RL for Tabular Data](https://medium.com/@tom.kaminski01/reinforcement-learning-for-f9a28632914f)**
A blog on implementation with some explanations provided too

- **[Reinforcement Learning with Tabular Q-Learning – Towards Data Science](https://towardsdatascience.com/reinforcement-learning-with-tabular-q-learning-in-python-6646cbd28ee0)**  
  Good practical guide for implementing tabular Q-learning in Python.

- **[Paper on RL for Large Datasets and Action Spaces](https://arxiv.org/pdf/2405.10310)**  
You are not expected to read this paper in detail, but it provides insights into RL challenges with large datasets and action spaces. Kind of relevant for the 


---

## 📝 This Week’s Task

### Main Assignment

**Build a basic DQN agent, as described in the DREAM11 paper:**

- **Environment:** Simulate a Dream11 team selection process with a fixed pool of players (22 or more), each with features like position, credit, recent form, average points.
- **Agent:** Use **Deep Q-Networks (DQN)** to train an agent that learns to select the best 11-player team based on rewards (fantasy points).
- **Reward:** Total fantasy score of the selected team. Penalize invalid teams (budget exceeded, too many players from same team, etc.).
- **Output:** Display the selected team after training + cumulative reward.
---

## 💡 Tips
- Treat each player selection as a step in the episode. The episode ends when 11 players are picked.
- Action space: all unselected players at that point.
- Use epsilon-greedy strategy for exploration during training.

