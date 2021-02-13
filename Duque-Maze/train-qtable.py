import numpy as np
import matplotlib.pyplot as plt
from environment.Maze import Maze
from agents.QTable import QTable

def showScores(scores0, scores1, name, save=True):
    plt.figure(0)
    plt.clf()
    plt.title('Learning')
    plt.xlabel('Sessions')
    plt.ylabel('Rewards')
    plt.plot(scores0, color='C1')
    plt.plot(scores1, color='C2')
    plt.pause(0.001)
    if save:
        plt.savefig(name)

def train():
    # Episodes to run
    episodes = config["episodes"]

    # Shape of array
    shape = config["environment"].shape

    # Length for Q-Table (All possible states for this environment)
    state_dim = shape[0] * shape[1]

    # Instance Q-Table
    Q = QTable(state_dim=state_dim, action_dim=4, gamma=0.999, alpha=0.8)
    Q.reset()

    # Initialize reward with zero
    r = 0

    scores0 = []
    scores1 = []

    # Loop episodes
    for episode in range(1, episodes + 1):

        # Print current episode
        print("\nEpisode: {}/{}".format(episode, episodes))

        # Reset game
        appMaze.reset()

        # Done False, is init
        done = False

        # Get current state
        s = appMaze.get_observable()

        # Play
        while not done:

            # Get action by q_table or epsilon pseudo random
            a = Q.epsilon_greedy(s)

            # Execute action
            s_, r, done = appMaze.step(a)

            # Update Q-Table
            Q.update(s, s_, a, reward=r)

            scores0.append(Q.score())

            # Set current state now
            s = s_

        if r > 0:
            print("Won")
        else:
            print("Lose")


    Q.reset('sarsa')

    # Initialize reward with zero
    r = 0
    scores1 = []

    # Loop episodes
    for episode in range(1, episodes + 1):

        # Print current episode
        print("\nEpisode: {}/{}".format(episode, episodes))

        # Reset game
        appMaze.reset()

        # Done False, is init
        done = False

        # Get current state
        s = appMaze.get_observable()

        a  = Q.epsilon_greedy(s)

        # Play
        while not done:

            # Execute action
            s_, r, done = appMaze.step(a)

            a_ =  Q.epsilon_greedy(s_)

            # Update Q-Table
            Q.update(s, s_, a, a_, r, done)

            scores1.append(Q.score())

            # Set current state now
            s, a = s_, a_

        if r > 0:
            print("Won")
        else:
            print("Lose")

    # showScores(scores0, scores1, 'comparativo.png')

    # Finish
    print("Finished")


if __name__ == "__main__":
    # Config for environment
    config = {
        "height": 600,  # Height for canvas
        "width": 600,  # Width for canvas
        "widthSquares": 100,  # Width and height for square
        "episodes": 300, # Run this number of episodes
        "animate": False, # Animate action
        "rewardPositive": 10,
        "rewardNegative": -10,
        "rewardEachStep": -0.01,
        "rewardInvalidStep": -1,
        "image_dim": (64, 64, 2),
        "environment": np.array([
            [[' '], [' '], [' '], [' '], [' '], [' ']],
            [[' '], ['X'], ['X'], [' '], ['X'], ['X']],
            [['I'], ['X'], [' '], [' '], [' '], [' ']],
            [[' '], ['X'], ['X'], ['X'], ['X'], ['+']],
            [[' '], [' '], [' '], [' '], [' '], [' ']],
            [['-'], ['-'], ['-'], ['-'], ['-'], ['-']]
        ])
    }

    # Instance Game Maze
    appMaze = Maze(config, mode='train-qtable')

    # Run train
    appMaze.after(3000, train)

    # Show Tkinter GUI
    appMaze.mainloop()



