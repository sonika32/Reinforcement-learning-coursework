# DQN for CartPole

This is my implementation of a Deep Q-Network to solve the CartPole-v1 environment from Gymnasium, done as part of my Reinforcement Learning coursework at the University of Bayreuth.

The goal is simple: keep a pole balanced on a moving cart for as long as possible. The agent gets +1 reward for every step it survives, and the episode ends if the pole tips too far, the cart goes off track, or it hits 500 steps.

## What I built
I implemented the DQN itself, a replay buffer, and a target network in PyTorch, then spent most of my time tuning hyperparameters and the exploration strategy to actually get it to learn well — the baseline version barely improves over time.

Things I tuned:
- network architecture (hidden layers/size)
- learning rate
- replay buffer size
- epsilon-greedy decay schedule

After tuning, the agent consistently reaches a mean return above 100 over 10 training runs, which was a big jump from the untuned version.

![Learning curve](learning_curve.png)

I also visualized the learned Q-values and policy to see what the agent actually picked up — plotting pole angle against angular velocity to see how it decides to push left or right.

![Q-values](q_value_heatmap.png)

## Files
- `coursework2.ipynb` — training, tuning, and plots
- `utils.py` — DQN, replay buffer, epsilon-greedy, loss function

## Running it
pip install torch gymnasium numpy matplotlib
Then just open `coursework2.ipynb` and run all cells. It'll train the agent across 10 runs (takes a few minutes), then plot the learning curve and the Q-value/policy visualizations at the end.

## Notes
The exploration schedule ended up mattering more than I expected — tweaking the network alone wasn't enough to get consistent results, the epsilon decay rate made a much bigger difference than I thought it would.

## Acknowledgements
Base structure of `utils.py` was provided as starter code for the RL coursework at the University of Bayreuth.

## Author
Sonika Kumari
