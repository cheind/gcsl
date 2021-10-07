# Vanilla GCSL
This repository contains a vanilla implementation of *"Learning to Reach Goals via Iterated Supervised Learning"* proposed by Dibya Gosh et al. in 2019. In short, their method uses an imitation learning objective to progressively refine a goal-conditioned policy by reframing past experiences as future goals.

This repository contains a minimal PyTorch-based implementation of the proposed method and applies it to an adapted Cartpole environment. In particular, the goal of the adapted Cartpole environment is to: a) maintain an upright pole (zero pole angle) and to reach a particular cart position (shown in red below). A qualitative performance comparison of two agents at different training times is shown below.

|<img src="./etc/cartpolenet_01000.gif"  width="80%">|<img src="./etc/cartpolenet_05000.gif"  width="80%">|<img src="./etc/cartpolenet_20000.gif"  width="80%">|
|:----------:|:----------:|:------------:|
| 1,000 steps | 5,000 steps | 20,000 steps |

## Run the code
Clone this repository, install the requirements and run
```
python -m examples.cartpole train
```
which will save models to `./tmp/cartpoleagent_xxxxx.pth`. 

To evaluate, run
```
python -m examples.cartpole eval ./tmp/cartpolenet_20000.pth
```
See command line options for tuning.

## Dynamic environment experiments
Since we condition our policy on goals, nothing stops us from changing the goals over time. The following animation shows the agent chasing a dynamic goal.

<div align="center">
<img src="./etc/cartpolenet_20000_dynamic.gif"  width="40%">
</div>

The above animation was created via the following command
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


## References
```bibtex
@article{ghosh2019learning,
  title={Learning to reach goals without reinforcement learning},
  author={Ghosh, Dibya and Gupta, Abhishek and Fu, Justin and Reddy, Ashwin and Devin, Coline and Eysenbach, Benjamin and Levine, Sergey},
  year={2019}
}
```