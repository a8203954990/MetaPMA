import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
import datetime

# Import CSMA environment
from MetaPMA_adaptation import CSMAEnv


class MetaCSMAEnv(CSMAEnv):
    """Extended CSMA environment supporting dynamic number of CSMA nodes"""

    def __init__(self, csma_num=None):
        super().__init__()
        if csma_num is not None:
            self.CSMA_num = csma_num
            # Reinitialize variables related to the number of CSMA nodes
            self.max_csma_nodes = max(self.CSMA_num)
            self.reset()

    def update_csma_num(self, new_csma_num):
        """Dynamically update the number of CSMA nodes"""
        self.CSMA_num = new_csma_num
        self.max_csma_nodes = max(self.CSMA_num)
        # Reinitialize related variables
        self.csma_backoff = [[np.random.randint(0, self.CW_min) for _ in range(num)] for num in self.CSMA_num]
        self.csma_state = [[0 for _ in range(num)] for num in self.CSMA_num]
        self.csma_retry = [[0 for _ in range(num)] for num in self.CSMA_num]
        self.csma_transmission_counter = [[0 for _ in range(num)] for num in self.CSMA_num]
        self.success_num_CSMA = [[0 for _ in range(num)] for num in self.CSMA_num]


def create_env_with_csma_num(csma_num):
    """Create environment with specific number of CSMA nodes"""
    return MetaCSMAEnv(csma_num)


# Create base environment
base_env = DummyVecEnv([lambda: MetaCSMAEnv()])

# Define PPO model
model = PPO('MlpPolicy', base_env, verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01)

# MAML parameters
meta_lr = 0.0001  # Meta learning rate
inner_lr = 0.001  # Inner loop learning rate
meta_iterations = 100  # Meta training iterations
meta_batch_size = 20  # Number of tasks per meta update
num_inner_updates = 10  # Inner loop updates per task
steps_per_update = 2048  # Sampling steps per update

# TensorBoard logger
writer = SummaryWriter(log_dir='./logs_csma_meta')


def generate_random_csma_num():
    """Generate random CSMA node configuration"""
    return [
        np.random.randint(1, 50),  # Node count range for channel 1
        np.random.randint(1, 40),  # Node count range for channel 2
        np.random.randint(1, 30)  # Node count range for channel 3
    ]


def clone_model(model):
    """Clone model"""
    model_clone = PPO('MlpPolicy', base_env, verbose=0)
    model_clone.set_parameters(model.get_parameters())
    return model_clone


def inner_loop_update(model, env, inner_lr, num_inner_updates):
    """Inner loop update"""
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=inner_lr)
    losses = []
    rewards_list = []

    for update in range(num_inner_updates):
        # Collect 2048 steps of data
        observations = []
        actions = []
        rewards = []
        total_rewards = 0

        obs = env.reset()
        for step in range(steps_per_update):  # Use 2048 steps
            action, _ = model.predict(obs)
            next_obs, reward, done, info = env.step(action)

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            total_rewards += reward

            obs = next_obs
            if done:
                obs = env.reset()

        # Calculate loss and update
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, requires_grad=True)
        loss = -torch.mean(rewards_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        rewards_list.append(total_rewards / steps_per_update)

    return np.mean(losses), np.std(losses), np.mean(rewards_list), np.std(rewards_list)


# Record start time before meta training loop
start_time = time.time()
total_updates = meta_iterations * meta_batch_size

# Create progress bar using tqdm
with tqdm(total=total_updates, desc="Meta-Training") as pbar:
    for meta_iteration in range(meta_iterations):
        meta_gradients = None
        meta_losses = []
        meta_rewards = []

        for task in range(meta_batch_size):
            # Calculate current progress
            current_update = meta_iteration * meta_batch_size + task

            # Update progress bar
            pbar.update(1)

            # Calculate estimated remaining time
            elapsed_time = time.time() - start_time
            updates_completed = current_update + 1
            time_per_update = elapsed_time / updates_completed
            remaining_updates = total_updates - updates_completed
            estimated_remaining_time = remaining_updates * time_per_update

            # Format time display
            elapsed = str(datetime.timedelta(seconds=int(elapsed_time)))
            remaining = str(datetime.timedelta(seconds=int(estimated_remaining_time)))

            # Update progress bar description
            pbar.set_description(
                f"Meta-Training: {current_update}/{total_updates} "
                f"[{elapsed}<{remaining}] "
                f"Loss: {np.mean(meta_losses):.3f} "
                f"Reward: {np.mean(meta_rewards):.3f}"
            )

            # Generate new CSMA node configuration for each task
            new_csma_num = generate_random_csma_num()
            task_env = DummyVecEnv([lambda: MetaCSMAEnv(new_csma_num)])

            # Record CSMA configuration
            writer.add_scalar(f'CSMA_Config/Channel1', new_csma_num[0], current_update)
            writer.add_scalar(f'CSMA_Config/Channel2', new_csma_num[1], current_update)
            writer.add_scalar(f'CSMA_Config/Channel3', new_csma_num[2], current_update)

            # Clone model
            model_clone = clone_model(model)

            # Perform inner loop update
            mean_loss, std_loss, mean_reward, std_reward = inner_loop_update(
                model_clone, task_env, inner_lr, num_inner_updates)

            # Record inner loop metrics
            writer.add_scalar(f'Inner_Loop_Mean_Loss/Task_{task}', mean_loss, current_update)
            writer.add_scalar(f'Inner_Loop_Mean_Reward/Task_{task}', mean_reward, current_update)

            # Calculate meta gradient
            obs = task_env.reset()
            total_loss = 0
            total_rewards = 0

            for step in range(100):
                action, _ = model_clone.predict(obs)
                obs, rewards, dones, info = task_env.step(action)
                rewards = torch.tensor(rewards, dtype=torch.float32, requires_grad=True)
                loss = -torch.mean(rewards)
                total_loss += loss.item()
                total_rewards += rewards.sum().item()
                loss.backward()

                if dones:
                    obs = task_env.reset()

            meta_losses.append(total_loss / 100)
            meta_rewards.append(total_rewards / 100)

            # Accumulate meta gradients
            if meta_gradients is None:
                meta_gradients = [param.grad.clone() if param.grad is not None else torch.zeros_like(param)
                                  for param in model.policy.parameters()]
            else:
                for i, param in enumerate(model.policy.parameters()):
                    if param.grad is not None:
                        meta_gradients[i] += param.grad.clone()

        # Apply meta gradients
        with torch.no_grad():
            for param, meta_grad in zip(model.policy.parameters(), meta_gradients):
                param -= meta_lr * meta_grad / meta_batch_size

        # Record meta loop metrics
        writer.add_scalar('Meta_Loop_Mean_Loss', np.mean(meta_losses), meta_iteration)
        writer.add_scalar('Meta_Loop_Mean_Reward', np.mean(meta_rewards), meta_iteration)

        # Update statistics
        if meta_losses:
            writer.add_scalar('Training/Average_Loss', np.mean(meta_losses), meta_iteration)
            writer.add_scalar('Training/Average_Reward', np.mean(meta_rewards), meta_iteration)

# Print total time after training
total_time = time.time() - start_time
print(f"\nTotal training time: {str(datetime.timedelta(seconds=int(total_time)))}")

# Save model
model.save("ppo_csma_meta")