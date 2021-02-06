import numpy as np
from environment.Maze import Maze
from agents.QTable import QTable


def train():
    # Episodes to run
    episodes = config["episodes"]

    # Shape of array
    shape = config["environment"].shape

    # Length for Q-Table (All possible states for this environment)
    state_dim = shape[0] * shape[1]

    # Instance Q-Table
    q_table = QTable(state_dim=state_dim, action_dim=4, gamma=0.999, alpha=0.8)

    # Initialize reward with zero
    reward = 0

    # Loop episodes
    for episode in range(1, episodes + 1):

        # Print current episode
        print("\nEpisode: {}/{}".format(episode, episodes))

        # Reset game
        appMaze.reset()

        # Done False, is init
        done = False

        # Get current state
        state = appMaze.get_observable()

        # Play
        while not done:
            # Get action by q_table or epsilon pseudo random
            action = q_table.select_action(state)

            # Execute action
            next_state, reward, done = appMaze.step(action)

            # Update Q-Table
            q_table.update_q_table(state, next_state, action, reward)

            # Set current state now
            state = next_state

        if reward > 0:
            print("Won")
        else:
            print("Lose")

    # Finish
    print("Finished")


if __name__ == "__main__":
    # Config for environment
    config = {
        "height": 600,  # Height for canvas
        "width": 600,  # Width for canvas
        "widthSquares": 100,  # Width and height for square
        "episodes": 100000, # Run this number of episodes
        "animate": True, # Animate action
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
    appMaze.after(2, train)

    # Show Tkinter GUI
    appMaze.mainloop()



