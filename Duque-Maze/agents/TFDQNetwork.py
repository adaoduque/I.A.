import random
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import numpy as np
import tensorflow as tf
import os

from collections import deque

class TFDQNetwork():
    def __init__(self, state_dim, action_dim, replay_size=100000, learn_rate=0.001, discount_rate=0.99, gamma=0.999, loss_fun="mse", mode='train'):
        self.mode           =  mode
        self.state_dim      =  state_dim
        self.action_dim     =  action_dim
        self.learn_rate     =  learn_rate
        self.discount_rate  =  discount_rate
        self.gamma          =  gamma
        self.loss_fun       =  loss_fun
        self.epsilon        =  1.0
        self.epsilon_final  =  0.01
        self.epsilon_decay  =  0.995

        self.replay         =  deque(maxlen=replay_size)
        self.modelActor     =  None
        self.modelCritic    =  None
        self.reward_window  =  []
        self.last_actions   =  []

    def createModelActor(self):
        self.modelActor = tf.keras.models.Sequential()
        self.modelActor.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(5, )))
        self.modelActor.add(tf.keras.layers.Dropout(0.2))
        self.modelActor.add(tf.keras.layers.Dense(units=self.action_dim, activation="softmax"))
        self.modelActor.compile(loss=self.loss_fun, optimizer=tf.keras.optimizers.Adam(lr=self.learn_rate))
        # self.modelActor.summary()

    def createModelCritic(self):
        self.modelCritic = tf.keras.models.Sequential()
        self.modelCritic.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(5, )))
        self.modelCritic.add(tf.keras.layers.Dropout(0.2))
        self.modelCritic.add(tf.keras.layers.Dense(units=self.action_dim, activation="softmax"))
        self.modelCritic.compile(loss=self.loss_fun, optimizer=tf.keras.optimizers.Adam(lr=self.learn_rate))
        # self.modelCritic.summary()

    def set_epsilon(self, epsilon, reset=False):
        if self.epsilon < 0.2:
            print("Epsilon reajustado 0")
            self.epsilon = epsilon
        elif reset:
            print("Epsilon reajustado 1")
            self.epsilon = epsilon
        else:
            print("Epsilon valor: "+str(self.epsilon))

    def select_action(self, s):
        if self.mode == 'train':
            if random.random() <= self.epsilon:
                action = random.randrange(self.action_dim)
                return action
        actions = self.modelActor.predict(s)
        action  =  np.argmax(actions[0])

        self.last_actions = np.append(self.last_actions, [action])
        return action

    def learning(self, batch_size):
        batch = []

        choices  =  random.sample(range(len(self.replay)), batch_size)
        for i in choices:
            batch.append(self.replay[i])

        for state, next_state, action, reward, done in batch:
            target = self.modelActor.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.modelCritic.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(t)

            self.modelActor.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay

    def store(self, cur_state, next_state, action, reward, done):
        # cur_state = np.array([cur_state]).reshape(-1, 1)
        # next_state = np.array([next_state]).reshape(-1, 1)
        self.replay.append((cur_state, next_state, action, reward, done))

        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]

    def update(self):
        # Check if replay memory is greater than or equal to 1000
        if len(self.replay) >= 1000:
            # Train the neural network with data in replay memory
            self.learning(batch_size=50)

    def save_model_actor(self, model_path="models/brain0-actor.h5"):
        self.modelActor.save(model_path)

    def save_model_critic(self, model_path="models/brain0-critic.h5"):
        self.modelCritic.save(model_path)

    def load_model_actor(self, model_path="models/brain0-actor.h5"):
        self.modelActor  =  tf.keras.models.load_model(model_path)

    def load_model_critic(self, model_path="models/brain0-critic.h5"):
        self.modelCritic  =  tf.keras.models.load_model(model_path)

    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1.)

    def save_memory(self):
        if os.path.isfile('./logs/memory.csv'):
            os.remove('./logs/memory.csv')

        file = open('./logs/memory.csv', mode="a", encoding="utf-8")

        for state, next_state, action, reward, done in self.replay:
            state = np.array2string(state, precision=2, separator=',', suppress_small=True)
            next_state = np.array2string(next_state, precision=2, separator=',', suppress_small=True)

            file.write( state+";"+next_state+";"+str(action)+";"+str(reward)+";"+str(done)+";\n" )
        file.close()