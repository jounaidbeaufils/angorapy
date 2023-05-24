<br />
<br />

<p align="center"><img src="docs/img/angorapy.svg" width=25% align="center" /></p>
<h3> <p align="center"> Variance Modifications of AngoraPy </p> </h3>

<br />
   
This ReadMe serves as documentation for the additional Gatherers and PPOAgent implementented for Jounaid Beaufils' BTR titled: 'Using Curiosity to Escape Local Maxima in Proximal Policy Optimisation'

Every additional Gatherer requires the VarPPOAgent to run, gatherer is then selected with the `VarPPOAgent.assign_gatherer()` method. While the code is written to be compatible with features of AngoraPy at the time of the fork, recurrent neural networks are not implemented.

## Variance Term
The Variance term is a value added to the PPO Loss. This is done by using the additional variance and variance inspired information gathererd by the selected `VarGatherer` and adding this information to the Loss with it's own loss function. Similar to the `loss.policy_loss` or `loss.value_loss`
### Variance Term Parameters
 `var_by_adv`,  parameter is used to divide the variance term by the advatage function. So that the variance term is larger when when the advantage is lower. 
`c_var`, this is the coeffiecient of the variance term
`var_discount`, this is the discount applied to the variance term.
## VarGatherers

### VarGatherer
The Gathere is modified to predict the  variance of reward  till episode end and the number of steps till episode end, at every step. This variance prediction is combined with the actual variance of rewards returned by the environment during the episode. The process is similar to the advantage estimation descriped in the PPO paper and implemented in AngoraPy. The estimation relies on pooling the variance according to the number of steps.

This is the only model that requires the following paramaters in the VarPPOAgent Constructor:

    agent  =  VarPPOAgent(...,
						    model_builder=build_var_ffn_models,
						    var_pred=True)
    
### VarGathererNoPreds
This gatherer only calculates the variance of the reward till end of episode for every step of the episode, without any estimation.

### VarGathererAbs
This gatherer only stores the absolute value of the current step's reward. This serveded as the most basic implementation of cariance curiosity we could think of. `var_discount` does not apply.

### VarGathererNoise 
This gatherer only generates a random value between 0 and 1 at every step, this is used to check if the gatherers are any better than adding random noise. `var_discount` does not apply.

## Typical Training
### Basic Setup
The simplest way to train an agent is by running the `run_experiment.py` script as follows:

	python run_experiment.py <"exp_str"> <"model_name">

Experiment string is a personal reference message. Model names are `var_pred` , `var_no_pred`, `abs` and  `noise`. Both without the greater-than and smaller-than symbols.

The agent ID is saved in a `experiments_log.txt` along with the `exp_str`, `model_name` and time. A new file is created if need, old ID are safe as each log is is apended on a new line. 