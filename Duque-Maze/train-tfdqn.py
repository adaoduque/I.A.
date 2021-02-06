import numpy as np
import matplotlib.pyplot as plt
from environment.Maze import Maze
from agents.TFDQNetwork import TFDQNetwork

def showScores(scores, name, color='C1', figure=0, save=True):
    plt.figure(figure)
    plt.clf()
    plt.title('Learning')
    plt.xlabel('Sessions')
    plt.ylabel('Rewards')
    plt.plot(scores, color=color)
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

    # Instance neural network with Tensorflow (mode == train|empty)
    brain0  =  TFDQNetwork(state_dim=state_dim, action_dim=4, replay_size=100000, learn_rate=0.001, mode='train')

    # Create model actor
    brain0.createModelActor()

    # Create model Critic
    brain0.createModelCritic()

    # Initialize reward with zero
    reward = 0

    # Max action per episode
    maxActions =  1000

    # To plot performance
    scores0   =  []

    # Loop episodes
    for episode in range(1, episodes + 1):

        # Print current episode
        print("\nEpisode: {}/{}".format(episode, episodes))

        # Reset game
        appMaze.reset()

        # Done False, start game.
        done = False

        # Get current state
        state = appMaze.get_observable()

        # Actions executed
        actExecuted =  0

        # Play
        while not done:
            # Get action by neural network or epsilon pseudo random
            action = brain0.select_action(state)

            # Execute action
            next_state, reward, done = appMaze.step(action)

            # Store data to train neural network
            brain0.store(state, next_state, action, reward, done )

            # Update scores
            scores0.append(brain0.score())

            # Set current state now
            state = next_state

            # Prevent overfit - Restore epsilon to explore the environment
            if actExecuted > maxActions:
                print("More than "+str(maxActions)+" actions")
                brain0.set_epsilon(1.0)
                reward = -10
                break
            actExecuted += 1

            # Train neural network
            brain0.update()

        # Winner or Lose ?
        if reward > 0:
            print("Won")
        else:
            print("Lose")

    # Save model pre-trained critical
    brain0.save_model_critic()

    # Save model pre-trained actor
    brain0.save_model_actor()

    # Plot scores
    showScores(scores0, name="logs/Brain0.png")

    # Finish
    print("Finished")


if __name__ == "__main__":

    # Config for environment
    config = {
        "height": 600,  # Height for canvas
        "width": 600,  # Width for canvas
        "widthSquares": 100,  # Width and height for square
        "episodes": 1000, # Run this number of episodes
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
    appMaze = Maze(config, mode='train-state')

    # Run train
    appMaze.after(2, train)

    # Show Tkinter GUI
    appMaze.mainloop()

