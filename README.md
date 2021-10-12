[![Build Status](https://app.travis-ci.com/cheind/gcsl.svg?branch=main)](https://app.travis-ci.com/cheind/gcsl)
# Vanilla GCSL
This repository contains a vanilla implementation of *"Learning to Reach Goals via Iterated Supervised Learning"* proposed by Dibya Gosh et al. in 2019. 

In short, the paper proposes a learning framework to progressively refine a goal-conditioned imitation policy `pi_k(a_t|s_t,g)` based on relabeling past experiences as new training goals. In particular, the approach iteratively performs the following steps: a) sample a new goal `g` and collect experiences using `pi_k(-|-,g)`, b) relabel trajectories such that reached states become surrogate goals (details below) and c) update the policy `pi_(k+1)` using a behavioral cloning objective. The approach is self-supervised and does not necessarily rely on expert demonstrations or reward functions. The paper shows, that training for these surrogate tuples actually leads to desirable goal-reaching behavior.

**Relabeling details** 
Let `(s_t,a_t,g)` be a state-action-goal tuple from an experienced trajectory and `(s_(t+r),a_(t+r),g)` any future state reached within the same trajectory. While the agent might have failed to reach `g`, we may construct the relabeled training objective `(s_t,a_t,s_(t+r))`, since `s_(t+r)` was actually reached via `s_t,a_t,s_(t+1),a_(t+1)...s_(t+r)`. 

**Discussion** By definition according to the paper, an optimal policy is one that reaches it goals. In this sense, previous experiences where relabeling has been performed constitute optimal self-supervised training data, regardless of the current state of the policy. Hence, old data can be reused at all times to improve the current policy. A potential drawback of this optimality definition is the absence of an *efficient* goal reaching behavior notion. However, the paper (and subsequent experiments) show experimentally that the resulting behavioral strategies are fairly goal-directed.

## About this repository
This repository contains a vanilla, easy-to-understand  PyTorch-based implementation of the proposed method and applies it to an customized Cartpole environment. In particular, the goal of the adapted Cartpole environment is to: a) maintain an upright pole (zero pole angle) and to reach a particular cart position (shown in red). A qualitative performance comparison of two agents at different training times is shown below. Training started with a random policy, no expert demonstrations were used.

|<img src="./etc/cartpolenet_01000.gif"  width="80%">|<img src="./etc/cartpolenet_05000.gif"  width="80%">|<img src="./etc/cartpolenet_20000.gif"  width="80%">|
|:----------:|:----------:|:------------:|
| 1,000 steps | 5,000 steps | 20,000 steps |

### Dynamic environment experiments
Since we condition our policy on goals, nothing stops us from changing the goals over time, i.e `g -> g(t)`. The following animation shows the agent successfully chasing a moving goal.

<div align="center">
<img src="./etc/cartpolenet_20000_dynamic.gif"  width="40%">
</div>

### Cooperative experiments
Dynamically adapting goals, allows us to study something more way interesting: cooperative human-agent behavior. In the following simple game, the cart should be moved to some abstract goal position. Human and agent share their abilities as follows: the human perceives the game (i.e., through a rendered image) and continuously decides on the target velocity of the cart (through key-presses, velocity is visualized in red). The agent's perceives this target velocity as a sub-goal of its own and combines it with its pole balancing target. Note that the combination of both goals is important for the team to succeed, as otherwise, the agent's actions (pole balancing) may counteract the humans intentions (moving the cart).

<div align="center">
<img src="./etc/cartpolenet_coop.gif" width="40%">
</div>

As may notice, the agent is currently not following the target velocities ideally. I suspect this is due to the current way of training, which just samples a specific velocity and then follows this velocity profile until the episode is over. I expect this to improve, when we also train with dynamic goals.

### Parallel environments

The branch `parallel-ray-envs` hosts the same cartpole example but training is speed-up via [ray](https://www.ray.io/) primitives. In particular, environments rollouts are parallelized and trajectory results are incorporated on the fly. The parallel version is roughly **35% faster** than the sequential one. Its currently not merged with main, since it requires a bit more code to digest.

## Run the code
Install
```
pip install git+https://github.com/cheind/gcsl.git
```
and start training via
```
python -m gcsl.examples.cartpole train
```
which will save models to `./tmp/cartpoleagent_xxxxx.pth`. To evaluate, run
```
python -m gcsl.examples.cartpole eval ./tmp/cartpolenet_20000.pth
```
See command line options for tuning. The above animation for the dynamic goal was created via the following command
```
python -m examples.cartpole eval ^
 tmp\cartpolenet_20000.pth ^
 -seed 123 ^
 -num-episodes 1 ^
 -max-steps 500 ^
 -goal-xmin "-1" ^
 -goal-xmax "1" ^
 --dynamic-goal ^
 --save-gif
```
To run the coop game using a pretrained weights, use
```
python -m gcsl.examples.cartpole play etc\cartpolenet_coop.pth
```



## References
```bibtex
@inproceedings{ghosh2021learning,
    title={Learning to Reach Goals via Iterated Supervised Learning},
    author={Dibya Ghosh and Abhishek Gupta and Ashwin Reddy and Justin Fu 
    and Coline Manon Devin and Benjamin Eysenbach and Sergey Levine},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=rALA0Xo6yNJ}
}

@misc{cheind2021gcsl,
  author = {Christoph Heindl},
  title = {Learning to Reach Goals via Iterated Supervised Learning},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/cheind/gcsl}},
}
```