import numpy as np
import random

class QTable:

    def __init__(self, state_dim, action_dim, gamma, alpha, mode='qlearning', epsilon=1.0, epsilon_final=0.01, epsilon_decay=0.995):
        self.state_dim   =  state_dim
        self.action_dim  =  action_dim

        self.mode  =  mode

        self.gamma  =  gamma
        self.alpha  =  alpha
        self.Q      =  None

        self.epsilon        =  epsilon
        self.epsilon_final  =  epsilon_final
        self.epsilon_decay  =  epsilon_decay
        self.reward_window  =  []


    def reset(self, mode='qlearning'):
        self.mode = mode
        self.epsilon = 1.0
        self.reward_window = []
        self.Q = np.zeros(shape=(self.state_dim, self.action_dim))

    def update(self, s, s_, a, a_=0, reward=0, done=False):
        # Update q-table
        if self.mode == 'qlearning':
            self.Q[s, a]  =  self.Q[s, a] + self.alpha * (reward + self.gamma * np.amax(self.Q[s_, :]) - self.Q[s, a])
        else:
            if done:
                self.Q[s, a] += self.alpha * ( reward  - self.Q[s, a] )
            else:
                self.Q[s, a] += self.alpha * ( reward + (self.gamma * self.Q[s_, a_] ) - self.Q[s, a] )

        self.reward_window.append(reward)
        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay

    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1.)

    def epsilon_greedy(self, s):
        if random.random() <= self.epsilon:
            # Explore: randomly choose an action
            action = random.randrange(self.action_dim)
        else:
            if self.mode == 'qlearning':
                # Exploit: Find action with highest rewards
                rewards  =  self.Q[s, :]
                max_reward = [i for i, e in enumerate(rewards) if e == max(rewards)]

                # Select the best arm
                if len( max_reward ) == 1:
                    # Best action for current state
                    action = max_reward[0]
                else:
                    # randomly choose an action
                    action = random.choice(max_reward)
            else:
                action = np.argmax(self.Q[s, :])

        return action
