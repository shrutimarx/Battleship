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
import scipy.signal
import numpy as np
import battleship_env  # our custom gym environment

# run 32 environments in parallel
num_envs = 32
env = gym.make_vec("gymnasium_env/battleship", num_envs=num_envs, vectorization_mode="async")

model = keras.Sequential(
    [
        layers.Input(shape=(8, 8, 3)),
        layers.Conv2D(32, (3, 3), strides=2, activation='relu', padding='same'),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(64, activation='linear')  
    ],
)

lr = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0007038791930658126,
    decay_steps=10000,
    decay_rate=0.9  # lr decays by 10% every 10000 steps
)

# define standard optimzer and loss for classification model
optimizer = keras.optimizers.Adam(learning_rate=lr)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,
    reduction=tf.keras.losses.Reduction.NONE
)

@tf.function  # compiled to graph for speed
def select_actions(states):
    logits = model(states, training=False)
    # sample from distribution over cells rather than always picking argmax
    actions = tf.random.categorical(logits, 1)
    return tf.squeeze(actions, axis=-1)

@tf.function
def train_step(states, actions, returns):
    with tf.GradientTape() as tape:
        logits = model(states, training=True)

        cross_entropy = loss_fn(actions, logits)

        # entropy bonus to encourage exploration
        probs = tf.nn.softmax(logits)
        entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1)

        loss = tf.reduce_mean(cross_entropy * returns - 0.01 * entropy)

    grads = tape.gradient(loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 1.0)  # prevent exploding gradients
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def compute_returns_vectorized(rewards, dones, gamma=0.99):
    # work backwards to accumulate discounted returns
    returns = np.zeros_like(rewards, dtype=np.float32)
    running_return = np.zeros(num_envs, dtype=np.float32)

    for t in reversed(range(rewards.shape[0])):
        running_return = rewards[t] + gamma * running_return * (1.0 - dones[t])
        returns[t] = running_return

    return returns

num_episodes = 100000
gamma = 0.027566378686344206  # best gamma found by optuna (very low — mostly immediate rewards)
obs, info = env.reset()
n_steps = 2  # collect 2 steps per env before each gradient update

# track stats per env
episode_lengths = np.zeros(num_envs, dtype=np.int32)
episode_rewards = np.zeros(num_envs, dtype=np.float32)
completed_lengths = []
completed_rewards = []
rehits = 0  # count of times agent fired at an already-shot cell

for episode in range(num_episodes):
    states, actions, rewards, dones = [], [], [], []

    # collect n_steps of experience across all envs
    for _ in range(n_steps):
        shot_board = obs["shot_board"]

        # one-hot encode shot board: 0→[1,0,0], 1→[0,1,0], 2→[0,0,1]
        state = tf.one_hot(shot_board, depth=3)

        action = select_actions(state).numpy()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = np.logical_or(terminated, truncated)

        rehits += np.sum(reward == -20.0)

        episode_lengths += 1
        episode_rewards += reward

        # log completed episodes and reset their per-env counters
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

    loss = train_step(flat_states, flat_actions, flat_returns)

    if episode % 100 == 0:
        avg_turns = np.mean(completed_lengths) if completed_lengths else float('nan')
        avg_reward = np.mean(completed_rewards) if completed_rewards else float('nan')
        print(f"Update {episode}/{num_episodes} | Loss: {loss:.4f} | Avg Turns/Game: {avg_turns:.1f} | Avg Reward: {avg_reward:.2f} | Re-hits: {rehits}")
        completed_lengths.clear()
        completed_rewards.clear()
        rehits = 0

print("Training Complete!")

# save the trained model in SavedModel format
model.export("saved_model_dir")



def representative_data_gen():
    # feed 100 random samples so the converter can calibrate quantization ranges
    for _ in range(100):
        dummy_state = np.random.choice([0, 1], size=(1, 8, 8, 3)).astype(np.float32)
        yield [dummy_state]

converter = tf.lite.TFLiteConverter.from_saved_model("saved_model_dir")
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # enable quantization
converter.representative_dataset = representative_data_gen

# full integer quantization— weights and activations both become int8
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8   # input tensor will be int8
converter.inference_output_type = tf.int8  # output tensor will be int8

tflite_model_quant = converter.convert()

# save the quantized model
with open("battleship_agent_int8.tflite", "wb") as f:
    f.write(tflite_model_quant)

print(f"New INT8 model size: {len(tflite_model_quant) / 1024:.1f} KB")

