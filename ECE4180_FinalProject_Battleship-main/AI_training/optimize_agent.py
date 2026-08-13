import gymnasium as gym
import tensorflow as tf
print("TF version:", tf.__version__)

# check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if tf.config.list_physical_devices('GPU'):
    print("TensorFlow is using the GPU!")
else:
    print("TensorFlow is still on the CPU. Check CUDA/cuDNN installation.")

import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras import backend
import optuna
from optuna.trial import TrialState
import numpy as np
import battleship_env  # our custom gym environment


def objective(trial: optuna.trial.Trial):
    backend.clear_session()

    # CNN-based policy network
    # input: 8x8x3 one-hot encoded shot board (0=unknown, 1=miss, 2=hit)
    model = keras.Sequential(
        [
            layers.Input(shape=(8, 8, 3)),
            layers.Conv2D(32, (3, 3), strides=2, activation='relu', padding='same'),  # extract spatial features
            layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(64, activation='linear')  # one output per cell, no activation (logits)
        ],
    )

    @tf.function  # compile to graph for faster execution
    def select_actions(states):
        logits = model(states, training=False)
        actions = tf.random.categorical(logits, 1)
        return tf.squeeze(actions, axis=-1)

    @tf.function
    def train_step(states, actions, returns, loss_fn, optimizer):
        with tf.GradientTape() as tape:
            logits = model(states, training=True)

            # reinforce loss: cross entropy weighted by discounted returns
            cross_entropy = loss_fn(actions, logits)

            # entropy bonus to encourage exploration
            probs = tf.nn.softmax(logits)
            entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1)

            # subtract entropy term to encourage exploration
            loss = tf.reduce_mean(cross_entropy * returns - 0.01 * entropy)

        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)  # clip to prevent exploding gradients
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    def compute_returns_vectorized(rewards, dones, gamma):
        returns = np.zeros_like(rewards, dtype=np.float32)
        running_return = np.zeros(num_envs, dtype=np.float32)

        for t in reversed(range(rewards.shape[0])):
            running_return = rewards[t] + gamma * running_return * (1.0 - dones[t])
            returns[t] = running_return

        return returns

    global num_envs
    num_envs = 32  # run 32 environments in parallel to collect experience faster

    # hyperparameters being tuned by optuna
    lr = trial.suggest_float("lr", 1e-5, 1e-3)
    gamma = trial.suggest_float("gamma", 0.0, 1.0)               # discount factor
    already_shot = trial.suggest_float("already_shot", -40.0, 0.0, step=5)  # penalty for repeat shots
    hit = trial.suggest_float("hit", 0.0, 10.0, step=1)          # reward for hitting a ship
    miss = trial.suggest_float("miss", -10.0, 0.0, step=1)       # reward for missing
    n_steps = trial.suggest_int("n_steps", 1, 4)                 # steps to collect before each update

    # vectorized env- runs num_envs copies of the game simultaneously
    env = gym.make_vec("gymnasium_env/battleship",
                       already_shot=already_shot,
                       hit=hit,
                       miss=miss,
                       num_envs=num_envs,
                       vectorization_mode="sync")

    print(f"Learning Rate: {lr}")
    print(f"Gamma: {gamma}")
    print(f"Already Shot: {already_shot}")
    print(f"Hit: {hit}")
    print(f"Miss: {miss}")
    print(f"Num steps: {n_steps}")

    # exponential decay on learning rate so training stabilizes over time
    lr = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=10000,
        decay_rate=0.9
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE
    )

    num_episodes = 70000
    obs, info = env.reset()

    # track stats across episodes
    episode_lengths = np.zeros(num_envs, dtype=np.int32)
    episode_rewards = np.zeros(num_envs, dtype=np.float32)
    completed_lengths = []
    completed_rewards = []
    rehits = 0
    avg_turns = 100.0  # default before any episodes complete

    for episode in range(num_episodes):
        states, actions, rewards, dones = [], [], [], []

        for _ in range(n_steps):
            shot_board = obs["shot_board"]

            state = tf.one_hot(shot_board, depth=3)

            action = select_actions(state).numpy()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = np.logical_or(terminated, truncated)

            # count how many re-hits happened this step - note: not perfectly accurate
            rehits_this_step = np.sum(reward == already_shot)
            rehits += rehits_this_step

            episode_lengths += 1
            episode_rewards += reward

            # log completed episodes and reset their counters
            for i, d in enumerate(done):
                if d:
                    completed_lengths.append(episode_lengths[i])
                    completed_rewards.append(episode_rewards[i])
                    episode_lengths[i] = 0
                    episode_rewards[i] = 0.0

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            obs = next_obs

        # compute discounted returns for the collected batch
        np_rewards = np.array(rewards)
        np_dones = np.array(dones)
        returns = compute_returns_vectorized(np_rewards, np_dones, gamma)

        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        flat_states = tf.reshape(tf.stack(states), (-1, 8, 8, 3))
        flat_actions = tf.reshape(tf.stack(actions), (-1,))
        flat_returns = tf.reshape(tf.convert_to_tensor(returns, dtype=tf.float32), (-1,))

        loss = train_step(flat_states, flat_actions, flat_returns, loss_fn, optimizer)

        if episode % 100 == 0:
            avg_turns = np.mean(completed_lengths) if completed_lengths else avg_turns
            avg_reward = np.mean(completed_rewards) if completed_rewards else float('nan')
            print(f"Update {episode}/{num_episodes} | Loss: {loss:.4f} | Avg Turns/Game: {avg_turns:.1f} | Avg Reward: {avg_reward:.2f} | Re-hits: {rehits}")
            completed_lengths.clear()
            completed_rewards.clear()
            rehits = 0

            # report to optuna- allows it to prune bad trials early
            trial.report(avg_turns, episode)
            if trial.should_prune():
                env.close()
                raise optuna.TrialPruned()

    env.close()
    return avg_turns  # optuna minimizes this (fewer turns = better agent)


if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(multivariate=True)

    # load existing study if it exists so we can resume interrupted runs
    study = optuna.create_study(
        study_name="battleship",
        direction="minimize",  # we want to minimize avg turns to win
        load_if_exists=True,
        sampler=sampler
    )

    # run up to 1000 trials or 15 hours, whichever comes first
    study.optimize(objective, n_trials=1000, timeout=54000)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study Statistics:")
    print("Finished: ", len(study.trials))
    print("Pruned: ", len(pruned_trials))
    print("Completed: ", len(complete_trials))

    best = study.best_trial
    print("Best Trial: ")
    print("Average: ", best.value)
    print("Parameters: ")
    for key, value in best.params.items():
        print("{}: {}".format(key, value))
