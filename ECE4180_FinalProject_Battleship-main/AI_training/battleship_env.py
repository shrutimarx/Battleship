import gymnasium as gym
import numpy as np

class battleshipEnv(gym.Env):
    """
    A custom Gymnasium environment for the game of Battleship.
    The agent learns to fire at an 8x8 grid to sink all hidden ships.
    Ships are randomly placed at the start of each episode.
    """

    def __init__(self, already_shot=-15.0, hit=6.0, miss=-4.0, size=8):
        # Reward values are passed as arguments so they can be tuned externally
        self.already_shot = already_shot  # penalty for firing at a cell already fired at
        self.hit = hit                    # reward for hitting a ship cell
        self.miss = miss                  # reward for missing (hitting empty water)

        self.size = size  # board is size x size (8x8 is default)

        self.board = np.zeros((self.size, self.size), dtype=np.int8)

        # What the agent has fired at so far
        self.shot_board = np.zeros((self.size, self.size), dtype=np.int8)

        self.num_ships = 4  # num ships to place on the board each episode

        # NumPy random number generator for ship placement
        self.rng = np.random.default_rng()

        # obv space: the agent sees only shot_board (not the hidden board)
        # vals are 0 (unknown), 1 (miss), or 2 (hit)
        self.observation_space = gym.spaces.Dict(
            {
                "shot_board": gym.spaces.Box(0, 2, shape=(size, size), dtype=np.int8)
            }
        )

        # action_space: one integer per cell (0 to size*size - 1)
        self.action_space = gym.spaces.Discrete(self.size * self.size)

    def _action_translation(self, action):
       # convert a flat action integer into (row, col) grid coordinates.
        return action // self.size, action % self.size

    def _get_obs(self):
        return {"shot_board": self.shot_board}

    def shipCoords(self, length, orientation):
        # Generate a valid random starting coordinate for a ship.
        if orientation:  # vertical
            x = self.rng.integers(low=0, high=self.size)
            y = self.rng.integers(low=0, high=(self.size - length + 1))
        else:  # horizontal
            x = self.rng.integers(low=0, high=(self.size - length + 1))
            y = self.rng.integers(low=0, high=self.size)
        return x, y

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # clear boards
        self.shot_board = np.zeros((self.size, self.size), dtype=np.int8)
        self.board = np.zeros((self.size, self.size), dtype=np.int8)

        for _ in range(self.num_ships):
            length = self.rng.integers(low=1, high=5)       # ship length: 1, 2, 3, or 4
            orientation = self.rng.integers(low=0, high=2)  # 0=horizontal, 1=vertical

            x, y = self.shipCoords(length, orientation)

            if orientation:  # vertical
                while (self.board[x, y:y+length] == 1).any():
                    x, y = self.shipCoords(length, orientation)
                self.board[x, y:y+length] = 1  # mark ship cells

            else:  # horizontal
                while (self.board[x:x+length, y] == 1).any():
                    x, y = self.shipCoords(length, orientation)
                self.board[x:x+length, y] = 1  # mark ship cells

        observation = self._get_obs()
        return observation, {}

    def step(self, action):
        # Execute one action (fire at a cell) and return the result.
        # Convert to grid coordinates
        x, y = self._action_translation(action)

        # Check if this cell has already been fired at (before updating)
        already_shot = self.shot_board[x, y]

        hit = False

        # Update shot_board based on whether there's a ship at (x, y)
        if self.board[x, y] == 1:
            self.shot_board[x, y] = 2  # mark as hit
            hit = True
        else:
            self.shot_board[x, y] = 1  # mark as miss

        # calculate reward for action
        if already_shot:
            reward = self.already_shot  
        elif hit:
            reward = self.hit
        else:
            reward = self.miss

        # observe board after action for next turn
        observation = self._get_obs()

        # check if game is over (all ships are shot)
        terminated = np.all((self.board == 1) == (self.shot_board == 2))

        # reward for beating the game
        if terminated:
            reward = reward + 10

        return observation, reward, terminated, False, {}


# Register the environment with Gymnasium so it can be created with gym.make()
gym.register(
    id="gymnasium_env/battleship",
    entry_point=battleshipEnv,
    max_episode_steps=100
)

