# Module 09: Deep Reinforcement Learning and Alignment

Five notebooks (run in order) and the Python modules they were built from. CPU only, no downloads, no gym package: the environments are written in plain numpy.

| File | Purpose |
|---|---|
| `mdp.py` | A 4 by 4 slipping grid as a tabular MDP; exact policy evaluation, value iteration, tabular Q-learning, an exact policy gradient and the REINFORCE estimate of it |
| `bandit.py` | The ten-armed bandit and five exploration strategies (greedy, optimistic, epsilon-greedy, UCB, softmax) |
| `cartpole.py` | CartPole in numpy (same physics as the classic task, 200-step cap) |
| `dqn.py` | Deep Q-network with a replay buffer and a target network, each of which can be switched off |
| `pg.py` | REINFORCE (with or without a baseline) and PPO with the clipped objective and GAE |
| `rlhf.py` | Preference world, Bradley-Terry reward model, KL-regularized optimum, DPO loss and training |
| `test_modules.py` | Checks the modules against the claims in the learning content |
| `figdata.py` | Regenerates the curves quoted in the learning content (same seeds as the notebooks) |

| Notebook | Covers |
|---|---|
| Part 1, MDPs, returns and value functions | Discounting, exact policy evaluation against simulation, value iteration, how slip and the discount factor change the best policy |
| Part 2, Exploration and exploitation | Greedy failure on 200 bandits, optimism, epsilon-greedy, UCB, softmax, the cost of epsilon, tabular Q-learning with and without exploration |
| Part 3, Deep Q-networks | A DQN on CartPole; five settings (full, window of 64, online, replay with batch 1, no target); value blow-up without a target network |
| Part 4, Policy gradients and PPO | REINFORCE against the exact gradient, the effect of a baseline, REINFORCE vs PPO vs DQN, what the clip buys |
| Part 5, RLHF and DPO | Reward model from comparisons, the KL-regularized optimum, proxy against true reward, an exploitable reward model, DPO against the two-stage route |

The notebooks are self-contained and were executed, so the outputs are real. The experiments use small networks and tiny tasks, so every number can be reproduced, but they show mechanics and not how large systems behave. Several results are reported as they came out and may differ from what you expect: removing the replay buffer in the form of a 64-step window did not hurt on CartPole (Part 3), PPO's clip made no measurable difference at ordinary step sizes (Part 4), and with independent reward noise the true reward does not fall when the policy over-optimizes (Part 5). Results are noisy across seeds; three seeds show large effects only.

Setup: `pip install torch numpy matplotlib scipy jupyter`, then `python test_modules.py` (about 25 seconds). Executing Part 3 takes about 7 minutes on two CPU cores and the other notebooks one to three minutes each.
