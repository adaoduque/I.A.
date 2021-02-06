import numpy as np
import random

class QTable:

    def __init__(self, state_dim, action_dim, gamma, alpha, epsilon=1.0, epsilon_final=0.01, epsilon_decay=0.995):
        self.state_dim   =  state_dim
        self.action_dim  =  action_dim

        self.gamma  =  gamma
        self.alpha  =  alpha

        self.epsilon        =  epsilon
        self.epsilon_final  =  epsilon_final
        self.epsilon_decay  =  epsilon_decay

        self.q_table = np.zeros(shape=(state_dim, action_dim))

    def update_q_table(self, s, next_s, a, r):
        # Update q-table
        self.q_table[s, a]  =  self.q_table[s, a] + self.alpha * (r + self.gamma * np.amax(self.q_table[next_s, :]) - self.q_table[s, a])

        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay

    def select_action(self, s):
        if random.random() <= self.epsilon:
            # Explore: randomly choose an action
            action = random.randrange(self.action_dim)
        else:
            # Exploit: Find action with highest rewards
            rewards  =  self.q_table[s, :]
            max_reward = [i for i, e in enumerate(rewards) if e == max(rewards)]

            # Select the best arm
            if len( max_reward ) == 1:
                # Best action for current state
                action = max_reward[0]
            else:
                # randomly choose an action
                action = random.choice(max_reward)

        return action
