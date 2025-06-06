# MetaPMA
Our paper “Meta-Reinforcement Learning for Low-Latency Multi-Link Access in Mixed Wi-Fi 7 Networks” is currently under submission to IEEE Internet of Things Journal.

MetaPMA is a novel meta-reinforcement learning approach for optimizing Proximal Policy Optimization (PPO) in multi-link wireless coexistence scenarios. This repository implements:

​​Meta-training framework​​ for learning adaptable Wi-Fi access policies

​​Fast adaptation module​​ for rapid deployment in new Wi-Fi environments

​​Baseline PPO implementation​​ for performance comparison

The approach enables learning policies that can quickly adapt to diverse wireless configurations, outperforming conventional PPO in dynamic Wi-Fi scenarios.

Repository Structure

├── data/              # Experimental results (dalay data)

│   ├── [delay_data_meta_updates_30_20_10_150mbps_90hz_256QAM_ACBE]     # An example of experimental delay data for a Typical Asymmetric Coexistence Scenario.

├── MetaPMA_train.py            # Meta-training workflow (nested loops)

├── MetaPMA_adaptation.py       # Fast adaptation for new environments

├── PMAtrain_environment.py     # Baseline PPO training/evaluation

├── ppo_csma_meta.zip           # Pre-trained meta-initialization model

│

└── README.md           # This document

Dependencies
Package	Version

Python	3.x

PyTorch	1.12.1

torchvision	0.13.1

stable-baselines3 2.4.0

gym 0.26.2

numpy	1.23.3

scipy 1.10.1

tqdm	4.61.1

matplotlib	3.6.0
