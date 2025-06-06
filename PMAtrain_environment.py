import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import deque
import random
import torch
from stable_baselines3.common.callbacks import BaseCallback
import math


class CSMAEnv(gym.Env):
    def __init__(self):
        super(CSMAEnv, self).__init__()

        # Define action space: 8 actions representing transmission on channels 1,2,3
        self.action_space = spaces.Discrete(8)  # 8 combinations for transmission decisions on channels 1,2,3

        # Define observation space: [3 channel states (busy/idle), queue length, head packet delay]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1, 100, 1000], dtype=np.float32),
            dtype=np.float32
        )

        # Initial state
        self.current_slot = 0  # Current slot index

        # CSMA parameters
        self.cut_stage = 6  # Control parameter
        self.CW_min = 16  # Control parameter (minimum contention window)
        self.CW_max = self.CW_min * (2 ** self.cut_stage)  # Maximum contention window

        # Number of CSMA nodes on each channel
        self.CSMA_num = [20, 20, 20]  # Channel 1:20, channel 2:20, channel 3:20 nodes  # Control parameter
        # self.CSMA_num = [12, 8, 4]  # Small-scale scenario
        self.max_csma_nodes = max(self.CSMA_num)  # For state space dimension design

        # Initialize backoff counters and states for each CSMA node
        self.csma_backoff = [[np.random.randint(0, self.CW_min) for _ in range(num)] for num in self.CSMA_num]
        self.csma_state = [[0 for _ in range(num)] for num in self.CSMA_num]  # 0: idle, 1: transmitting
        self.csma_retry = [[0 for _ in range(num)] for num in self.CSMA_num]  # Transmission retry count
        self.csma_transmission_counter = [[0 for _ in range(num)] for num in self.CSMA_num]  # Transmission duration counter

        # Transmission duration per channel (in slots)
        # self.channel_duration = [39, 72, 136]  # Slots required for channels 1,2,3
        self.channel_duration = [136, 72, 39]  # Reversed environment  # Control parameter
        # self.channel_duration = [197, 103, 54]  # 64QAM
        self.transmission_counter = [0, 0, 0]  # Track remaining slots for agent's transmission

        # Success transmission counters
        self.success_num_CSMA = [[0 for _ in range(num)] for num in self.CSMA_num]
        self.success_num_agent = [0, 0, 0]

        # Collision duration parameters
        # self.channel_collision_duration = [37, 70, 134]  # Collision duration per channel
        self.channel_collision_duration = [134, 70, 37]  # Collision duration per channel  # Control parameter
        # self.channel_collision_duration = [195, 101, 52]
        self.channel_collision_counter = [0, 0, 0]  # Track remaining collision time per channel

        # VR traffic parameters
        self.fps = 90  # Control parameter (frames per second)
        self.bps = 150  # Mbps  # Control parameter (bandwidth)
        # self.fps = 72
        # self.bps = 100  # Mbps
        self.u_interval = 1 / self.fps  # Seconds
        self.b_interval = 0.00065  # Seconds
        # self.b_interval = 0.00098  # Seconds
        self.slot_duration = 9e-6  # 1 slot = 9 microseconds
        self.u_slot = self.u_interval / self.slot_duration
        self.b_slot = self.b_interval / self.slot_duration

        self.u_frame_length = (1.02 * self.bps + 0.28) * 1e6 / (8 * self.fps)
        self.b_frame_length = 124.9 * self.bps + 3080.6
        # self.b_frame_length = 98.3 * self.bps + 1980.1

        # VR traffic queue
        self.VR_queue = deque()
        self.packet_timestamps = deque()  # Record slot when each packet was generated
        self.packet_num = 0
        self.next_burst_time = 0  # Next burst generation time
        self.remaining_burst_packets = 0  # Packets remaining in current burst

        # Track collided packets and timestamps
        self.collision_packets = [[] for _ in range(3)]  # List of collided packets per channel
        self.collision_timestamps = [[] for _ in range(3)]  # Corresponding timestamps

        # Reward parameters
        self.base_reward = 2.0  # Base reward for successful transmission
        self.delay_weight = 0.5  # Weight for delay penalty
        self.collision_penalty = -2.0  # Penalty for collisions
        self.empty_transmission_penalty = 0  # Penalty for empty transmission attempts

        # Maximum acceptable packet delay
        self.max_acceptable_delay = 1500  # Maximum acceptable delay threshold
        # Record packet delays
        self.delay_records = []

        self.collision_num = 0  # Collision counter

        # Reward centering variables
        self.reward_mean = 0
        self.reward_count = 0
        self.alpha = 0  # Smoothing factor for historical rewards

        # Channel busy rate statistics
        self.busy_window = 1000  # Window size for busy rate calculation
        self.channel_busy_history = [deque([0] * self.busy_window, maxlen=self.busy_window) for _ in range(3)]

        # Transmission attempt counters
        self.attempt_num_agent = [0, 0, 0]  # Transmission attempts per channel
        self.success_num_agent = [0, 0, 0]  # Successful transmissions per channel

    def reset(self):
        """Reset environment to initial state"""
        self.current_slot = 0
        self.VR_queue.clear()
        self.packet_timestamps.clear()
        self.packet_num = 0
        self.next_burst_time = np.random.randint(0, 100)  # Initialize burst time
        self.remaining_burst_packets = 0

        # Reset all CSMA node states
        self.csma_backoff = [[np.random.randint(0, self.CW_min) for _ in range(num)] for num in self.CSMA_num]
        self.csma_state = [[0 for _ in range(num)] for num in self.CSMA_num]
        self.csma_retry = [[0 for _ in range(num)] for num in self.CSMA_num]
        self.csma_transmission_counter = [[0 for _ in range(num)] for num in self.CSMA_num]
        self.transmission_counter = [0, 0, 0]

        # Reset success counters
        self.success_num_CSMA = [[0 for _ in range(num)] for num in self.CSMA_num]
        self.success_num_agent = [0, 0, 0]

        # Reset collision counters
        self.channel_collision_counter = [0, 0, 0]

        self.delay_records = []  # Reset delay records

        self.VR_queue.clear()
        self.packet_timestamps.clear()
        self.collision_packets = [[] for _ in range(3)]
        self.collision_timestamps = [[] for _ in range(3)]

        self.collision_num = 0

        # Reset reward centering variables
        self.reward_mean = 0
        self.reward_count = 0

        # Reset channel busy history
        self.channel_busy_history = [deque([0] * self.busy_window, maxlen=self.busy_window) for _ in range(3)]

        # Reset counters
        self.attempt_num_agent = [0, 0, 0]
        self.success_num_agent = [0, 0, 0]

        return np.array(self._get_observation())

    def step(self, action):
        """Execute one environment step"""
        done = False
        rewards = [0, 0, 0]  # Per-channel rewards

        # Update VR traffic generation
        self.next_burst_time, self.remaining_burst_packets = self.generate_VR(
            self.current_slot, self.next_burst_time, self.remaining_burst_packets,
            self.u_slot, self.b_slot, self.u_frame_length, self.b_frame_length
        )

        # Add new packets if generated
        if self.remaining_burst_packets > 0:
            self.VR_queue.append(1)  # Add packet
            self.packet_timestamps.append(self.current_slot)  # Record generation time
            self.remaining_burst_packets -= 1

        # Decode PPO action into individual channel decisions
        ppo_actions = self._decode_action(action)

        # Phase 1: Update transmission states and collision counters
        for i in range(3):
            # Update collision counter
            if self.channel_collision_counter[i] > 0:
                self.channel_collision_counter[i] -= 1
                # When collision ends, return collided packets to queue head
                if self.channel_collision_counter[i] == 0 and self.collision_packets[i]:
                    # Return packets in LIFO order
                    while self.collision_packets[i]:
                        packet = self.collision_packets[i].pop()
                        timestamp = self.collision_timestamps[i].pop()
                        self.VR_queue.appendleft(packet)
                        self.packet_timestamps.appendleft(timestamp)

            # Update agent's transmission state
            if self.transmission_counter[i] > 0:
                self.transmission_counter[i] -= 1

            # Update CSMA nodes' transmission states
            for j in range(self.CSMA_num[i]):
                if self.csma_transmission_counter[i][j] > 0:
                    self.csma_transmission_counter[i][j] -= 1
                    if self.csma_transmission_counter[i][j] == 0:
                        self.csma_state[i][j] = 0
                        # Reset backoff after transmission
                        self.csma_retry[i][j] = 0
                        self.csma_backoff[i][j] = np.random.randint(0, self.CW_min)

        # Phase 2: Process backoff and transmission attempts
        for i in range(3):
            # Check if channel is busy
            channel_busy = (self.transmission_counter[i] > 0 or  # Agent transmission
                            any(self.csma_transmission_counter[i][j] > 0
                                for j in range(self.CSMA_num[i])) or  # CSMA transmission
                            self.channel_collision_counter[i] > 0)  # Collision state

            if not channel_busy:  # Only process when channel is idle
                total_channel_duration = sum(self.channel_duration)
                channel_diff = (total_channel_duration - self.channel_duration[i]) / total_channel_duration  # Channel quality metric
                # Collect CSMA nodes ready to transmit (backoff=0)
                ready_nodes = []
                for j in range(self.CSMA_num[i]):
                    if self.csma_state[i][j] == 0:  # Node is idle
                        if self.csma_backoff[i][j] == 0:  # Backoff finished
                            ready_nodes.append(j)
                        else:  # Decrement backoff counter
                            self.csma_backoff[i][j] -= 1

                # Agent's transmission decisions
                ppo_attempt = ppo_actions[i] == 1 and len(self.VR_queue) > 0
                empty_attempt = ppo_actions[i] == 1 and len(self.VR_queue) == 0

                # Count agent attempts
                if ppo_attempt:
                    self.attempt_num_agent[i] += 1

                # Case 1: Only agent transmits
                if ppo_attempt and not ready_nodes:
                    packet = self.VR_queue.popleft()
                    waiting_delay = self.current_slot - self.packet_timestamps.popleft()
                    transmission_delay = self.channel_duration[i]
                    total_delay = waiting_delay + transmission_delay

                    self.delay_records.append(total_delay)
                    # Set transmission state
                    self.transmission_counter[i] = self.channel_duration[i]
                    self.success_num_agent[i] += 1

                    # Reward calculation
                    # 1. Base transmission reward
                    rewards[i] = self.base_reward

                    # 2. Delay penalty
                    delay_penalty = self.delay_weight * (total_delay / self.max_acceptable_delay)
                    rewards[i] -= delay_penalty

                    # 3. Channel quality bonus
                    rewards[i] += channel_diff

                # Case 2: Only one CSMA node transmits
                elif not ppo_attempt and len(ready_nodes) == 1:
                    node = ready_nodes[0]
                    self.csma_state[i][node] = 1
                    self.csma_transmission_counter[i][node] = self.channel_duration[i]
                    self.success_num_CSMA[i][node] += 1

                    # Penalize agent if packets are waiting
                    if len(self.VR_queue) > 0:
                        oldest_packet_delay = self.current_slot - self.packet_timestamps[0]
                        urgency_factor = min(1.0, oldest_packet_delay / self.max_acceptable_delay)
                        if urgency_factor > 0.5:  # Penalty only for urgent packets
                            rewards[i] = -urgency_factor

                # Case 3: Collision occurs (agent + CSMA or multiple CSMA)
                elif (ppo_attempt and ready_nodes) or len(ready_nodes) > 1:
                    # Set collision duration
                    self.channel_collision_counter[i] = self.channel_collision_duration[i]

                    # Process agent packet if involved
                    if ppo_attempt:
                        # Save collided packet
                        packet = self.VR_queue.popleft()
                        timestamp = self.packet_timestamps.popleft()
                        self.collision_packets[i].append(packet)
                        self.collision_timestamps[i].append(timestamp)

                        # Calculate packet urgency
                        packet_delay = self.current_slot - timestamp
                        urgency_factor = min(1.0, packet_delay / self.max_acceptable_delay)

                        # Dynamic collision penalty adjustment
                        # Higher penalty for non-urgent packets (urgency_factor near 0)
                        # Lower penalty for urgent packets (urgency_factor near 1)
                        dynamic_penalty = self.collision_penalty * (1 - urgency_factor)
                        rewards[i] = dynamic_penalty

                        self.collision_num += 1

                    # Update CSMA backoff parameters
                    for node in ready_nodes:
                        self.csma_retry[i][node] = min(self.csma_retry[i][node] + 1, self.cut_stage)
                        cw = min(self.CW_max, self.CW_min * (2 ** self.csma_retry[i][node]))
                        self.csma_backoff[i][node] = np.random.randint(0, cw)

                # Case 4: Agent attempts transmission with empty queue
                elif empty_attempt:
                    rewards[i] = self.empty_transmission_penalty

                # Case 5: Idle channel but agent doesn't transmit when packets are waiting
                elif not ready_nodes and not ppo_attempt and len(self.VR_queue) > 0:
                    oldest_packet_delay = self.current_slot - self.packet_timestamps[0]
                    urgency_factor = min(0.5, oldest_packet_delay / self.max_acceptable_delay)
                    rewards[i] = -(urgency_factor + channel_diff)

        # Global reward adjustment
        if len(self.VR_queue) > 0 and all(r < 0 for r in rewards):
            # Calculate head packet urgency
            oldest_packet_delay = self.current_slot - self.packet_timestamps[0]
            urgency_factor = min(1.0, oldest_packet_delay / self.max_acceptable_delay)
            # Extra penalty for missed transmission opportunity
            total_reward = sum(rewards) - 0.5 * urgency_factor
        else:
            total_reward = sum(rewards)

        # Implement reward centering
        self.reward_count += 1
        self.reward_mean = (1 - self.alpha) * self.reward_mean + self.alpha * total_reward
        centered_reward = total_reward - self.reward_mean

        # Update slot and check termination
        self.current_slot += 1
        if self.current_slot >= 20000:
            done = True

        # Update channel busy history
        for i in range(3):
            is_busy = 1 if (self.transmission_counter[i] > 0 or  # Agent transmission
                            any(self.csma_transmission_counter[i][j] > 0
                                for j in range(self.CSMA_num[i])) or  # CSMA transmission
                            self.channel_collision_counter[i] > 0) else 0  # Collision state
            self.channel_busy_history[i].append(is_busy)

        return np.array(self._get_observation()), centered_reward, done, {}

    def _get_observation(self):
        """Get the current observable state"""
        # Get current channel states (busy=1, idle=0)
        channel_states = [
            1 if (self.transmission_counter[i] > 0 or
                  any(self.csma_transmission_counter[i][j] > 0
                      for j in range(self.CSMA_num[i])) or
                  self.channel_collision_counter[i] > 0)
            else 0 for i in range(3)
        ]

        # Calculate channel busy rates (0.0-1.0)
        channel_busy_rates = [
            sum(history) / len(history)
            for history in self.channel_busy_history
        ]

        # Queue information
        queue_length = len(self.VR_queue)

        # Head packet delay (waiting time)
        head_packet_delay = 0
        if self.packet_timestamps:
            head_packet_delay = self.current_slot - self.packet_timestamps[0]

        # Normalization
        normalized_queue = min(1.0, queue_length / 100.0)  # Queue length scaled to [0,1]
        normalized_delay = min(1.0, head_packet_delay / self.max_acceptable_delay)  # Delay scaled to [0,1]

        return np.array(
            channel_states +
            channel_busy_rates +  # Include channel busy rates
            [normalized_queue, normalized_delay],
            dtype=np.float32
        )

    def _decode_action(self, action):
        """Decode action space into transmission state list"""
        return [(action >> i) & 1 for i in range(3)]

    def generate_VR(self, current_t, next_burst_time, remaining_burst_packets, u_slot, b_slot, u_frame_length, b_frame_length):
        """Generate VR traffic at scheduled times"""
        if current_t == next_burst_time:
            frame_length = max(1, int(np.random.laplace(u_frame_length, b_frame_length)))
            remaining_burst_packets = int(np.ceil(frame_length / 16384))  # 2^14 byte packets
            # remaining_burst_packets = int(np.ceil(frame_length / 1500))  # 1500 BYTES
            interval = max(1, np.random.laplace(u_slot, b_slot))
            interval = int(np.ceil(interval))
            next_burst_time = current_t + interval

        return next_burst_time, remaining_burst_packets

    def render(self, mode="human"):
        """Render environment (not implemented)"""
        pass

    def seed(self, seed=None):
        """Set random seed for reproducibility"""
        np.random.seed(seed)
        return [seed]


class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.ep_rewards_mean = []  # Store mean episode rewards

        # New: Add file path parameter (5.13 update)
        self.save_path = "episode_rewards.txt"

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            # Get mean reward from episode buffer
            mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            self.ep_rewards_mean.append(mean_reward)

            # New: Append to file in real-time (5.13 update)
            with open(self.save_path, 'a') as f:
                f.write(f"{mean_reward}\n")
        return True

    def plot_rewards(self):
        """Plot training reward curve"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.ep_rewards_mean, linewidth=1.5)
        plt.title('Training Reward Curve')
        plt.xlabel('Update Step')
        plt.ylabel('Mean Episode Reward')

        # Set x-axis range
        plt.xlim(0, len(self.ep_rewards_mean))

        # Set y-axis range based on expected convergence range (200-300)
        y_min = min(min(self.ep_rewards_mean), 0)  # Include negative values if any
        y_max = max(max(self.ep_rewards_mean), 300)  # Cap at 300

        # Round to nearest 50
        y_min_plot = math.floor(y_min / 50) * 50
        y_max_plot = math.ceil(y_max / 50) * 50

        plt.ylim(y_min_plot, y_max_plot)

        # Hide grid lines
        plt.grid(False)

        # Set ticks at start and end
        plt.xticks([0, len(self.ep_rewards_mean)])
        plt.yticks([y_min_plot, y_max_plot])

        # Save as vector image
        plt.savefig('training_curve.svg', format='svg', dpi=300, bbox_inches='tight')
        plt.show()

    # New: Save rewards to file
    def save_rewards_to_file(self, filename=None):
        target_path = filename or self.save_path
        np.savetxt(target_path, self.ep_rewards_mean)


# Post-training testing and visualization
def test_and_visualize(env, model, test_episodes=10, algorithm_name="meta_updates", base_seed=42):
    # Initialize statistical variables
    queue_lengths = []
    all_delays = []
    # Initialize cumulative throughput arrays
    ppo_throughput_sum = [0, 0, 0]  # Cumulative throughput of PPO nodes on three channels
    csma_throughput_sum = [0, 0, 0]  # Cumulative throughput of CSMA nodes on three channels
    total_collision_num = 0
    action_count = np.zeros(8)

    # Initialize cumulative statistics
    total_attempts = [0, 0, 0]  # Total attempts per channel
    total_successes = [0, 0, 0]  # Total successes per channel

    for episode in range(test_episodes):
        # Set different seed for each episode
        episode_seed = base_seed + episode
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        env.seed(episode_seed)
        torch.manual_seed(episode_seed)

        obs = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            # obs, reward, done, _ = env.step(action)  # # Controlled variable
            action_count[action] += 1
            obs, reward, done, _ = env.step(7)  # total greedy # Controlled variable
            # Record real-time queue length - using len() instead of qsize()
            queue_lengths.append(len(env.VR_queue))

        all_delays.extend(env.delay_records)
        total_collision_num += env.collision_num

        # Accumulate statistics per episode
        for i in range(3):
            total_attempts[i] += env.attempt_num_agent[i]
            total_successes[i] += env.success_num_agent[i]

        # Accumulate throughput per experiment
        for i in range(3):
            # PPO node throughput
            ppo_throughput_sum[i] += (env.success_num_agent[i] * env.channel_duration[i]) / env.current_slot

            # CSMA node throughput (sum of all CSMA nodes on that channel)
            csma_sum = sum(env.success_num_CSMA[i]) * env.channel_duration[i] / env.current_slot

            csma_throughput_sum[i] += csma_sum

    # Save delay data to file
    np.save(f'data/delay_data_{algorithm_name}_20_20_20_150mbps_90hz_256QAM_ACBE_advance.npy', np.array(all_delays))  # Controlled variable

    # Create two separate charts
    # 1. Queue length evolution over time
    plt.figure(figsize=(10, 5))
    time_slots = np.arange(len(queue_lengths))
    plt.plot(time_slots, queue_lengths, color='#1f77b4', alpha=0.8, linewidth=1.5, label='Queue Length')
    plt.title('Real-time Queue Length Evolution', fontsize=12, pad=10)
    plt.xlabel('Time (slots)', fontsize=10)
    plt.ylabel('Queue Length (packets)', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')

    # Add queue statistics
    mean_queue = np.mean(queue_lengths)
    max_queue = np.max(queue_lengths)
    min_queue = np.min(queue_lengths)

    stats_text = (f'Maximum: {max_queue}\n'
                  f'Average: {mean_queue:.1f}\n'
                  f'Minimum: {min_queue}')

    plt.text(len(time_slots) * 0.02, max_queue * 0.95,
             stats_text,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.5))
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('queue_length_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Delay distribution chart
    if all_delays:
        plt.figure(figsize=(10, 5))

        # Use better colors and transparency
        n, bins, patches = plt.hist(all_delays, bins=50, density=True, alpha=0.7,
                                    color='#2ecc71', edgecolor='white', linewidth=1)

        # Add kernel density estimation curve
        kernel = stats.gaussian_kde(all_delays)
        x_range = np.linspace(min(all_delays), max(all_delays), 200)
        plt.plot(x_range, kernel(x_range), color='#e74c3c', linewidth=2,
                 label='Density Estimation')

        plt.title('Packet Delay Distribution', fontsize=12, pad=10)
        plt.xlabel('Delay (slots)', fontsize=10)
        plt.ylabel('Density', fontsize=10)
        plt.grid(True, alpha=0.3, linestyle='--')

        # Calculate delay statistics
        mean_delay = np.mean(all_delays)
        max_delay = np.max(all_delays)
        min_delay = np.min(all_delays)
        p95_delay = np.percentile(all_delays, 95)
        p99_delay = np.percentile(all_delays, 99)

        # Add vertical lines for key statistics
        plt.axvline(x=mean_delay, color='#3498db', linestyle='--', alpha=0.8,
                    label=f'Mean: {mean_delay:.1f}')
        plt.axvline(x=p95_delay, color='#f1c40f', linestyle='--', alpha=0.8,
                    label=f'P95: {p95_delay:.1f}')
        plt.axvline(x=p99_delay, color='#e74c3c', linestyle='--', alpha=0.8,
                    label=f'P99: {p99_delay:.1f}')

        # Add detailed statistics text box
        stats_text = (f'Maximum: {max_delay:.1f}\n'
                      f'P99: {p99_delay:.1f}\n'
                      f'P95: {p95_delay:.1f}\n'
                      f'Mean: {mean_delay:.1f}\n'
                      f'Minimum: {min_delay:.1f}')

        plt.text(0.95, 0.95, stats_text,
                 transform=plt.gca().transAxes,
                 verticalalignment='top',
                 horizontalalignment='right',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.5))

        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig('delay_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Calculate differences between consecutive packet delays
        delay_differences = np.diff(all_delays)

        # Calculate jitter (standard deviation of delay differences)
        jitter = np.std(delay_differences)

        # Print jitter statistics
        print("\nJitter Statistics:")
        print(f"Standard Deviation of Delay Differences (Jitter): {jitter:.2f} slots")

    # Print average throughput statistics
    print("\nAverage Performance over", test_episodes, "episodes:")
    print(f"Average Collision Number: {total_collision_num / test_episodes:.2f}")
    print("\nPPO Node Average Throughput:")
    for i in range(3):
        avg_ppo = ppo_throughput_sum[i] / test_episodes
        print(f"Channel {i + 1}: {avg_ppo:.4f}")

    print("\nCSMA Nodes Average Throughput (sum of all CSMA nodes):")
    for i in range(3):
        avg_csma = csma_throughput_sum[i] / test_episodes
        print(f"Channel {i + 1}: {avg_csma:.4f}")

    print(action_count)

    # Print transmission success rate statistics
    print("\nAverage Transmission Success Rate over", test_episodes, "episodes:")
    for i in range(3):
        avg_attempts = total_attempts[i] / test_episodes
        avg_successes = total_successes[i] / test_episodes
        success_rate = total_successes[i] / total_attempts[i] if total_attempts[i] > 0 else 0

        print(f"Channel {i + 1}:")
        print(f"  Average Attempts per Episode: {avg_attempts:.2f}")
        print(f"  Average Successful Transmissions per Episode: {avg_successes:.2f}")
        print(f"  Overall Success Rate: {success_rate:.2%}")


# Perform testing and visualization after training
if __name__ == "__main__":
    # Create environment and train model
    vec_env = make_vec_env(CSMAEnv, n_envs=1)
    model = PPO("MlpPolicy",
                vec_env,
                verbose=1,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                learning_rate=3e-4,
                ent_coef=0.01,
                clip_range=0.2,
                gae_lambda=0.95,
                gamma=0.99)

    reward_callback = RewardCallback()

    # # Train model with callback
    model.learn(total_timesteps=2000000, callback=reward_callback)

    # # Plot training curves
    reward_callback.plot_rewards()
    reward_callback.save_rewards_to_file("final_rewards.txt")

    # Save trained model
    model.save("ppo_20_20_20_150mbps_90hz_256QAM_ACBE")

    test_env = CSMAEnv()

    test_and_visualize(test_env, model, test_episodes=10, algorithm_name='local_ppo', base_seed=777)
