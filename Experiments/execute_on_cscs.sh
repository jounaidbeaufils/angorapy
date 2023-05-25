# run jsubmit for each of the 50 indices
for i in {0..49}
do
  bash jsubmit.sh -p "Experiments/fine_tune.py $i" -t 24:00:00
done

# run 10 iterations of each model type
for i in {0..9}
do
  bash jsubmit.sh -p "Experiments/run_experiment.py jounaid var_pred --div true --env PandaPushDense-v2 --save_interval 0 --n 500" -t 05:00:00
  bash jsubmit.sh -p "Experiments/run_experiment.py jounaid var_pred --div true --env PandaReachDense-v2 --save_interval 0 --n 500" -t 05:00:00
  bash jsubmit.sh -p "Experiments/run_experiment.py jounaid var_pred --env PandaPushDense-v2 --save_interval 0 --n 500" -t 05:00:00
  bash jsubmit.sh -p "Experiments/run_experiment.py jounaid var_pred --env PandaReachDense-v2 --save_interval 0 --n 500" -t 05:00:00
  bash jsubmit.sh -p "Experiments/run_experiment.py jounaid var_no_pred --div true --env PandaPushDense-v2 --save_interval 0 --n 500" -t 05:00:00
  bash jsubmit.sh -p "Experiments/run_experiment.py jounaid var_no_pred --div true --env PandaReachDense-v2 --save_interval 0 --n 500" -t 05:00:00
  bash jsubmit.sh -p "Experiments/run_experiment.py jounaid var_no_pred --env PandaPushDense-v2 --save_interval 0 --n 500" -t 05:00:00
  bash jsubmit.sh -p "Experiments/run_experiment.py jounaid var_no_pred --env PandaReachDense-v2 --save_interval 0 --n 500" -t 05:00:00
  bash jsubmit.sh -p "Experiments/run_experiment.py jounaid abs --env PandaPushDense-v2 --save_interval 0 --n 500" -t 05:00:00
  bash jsubmit.sh -p "Experiments/run_experiment.py jounaid abs --env PandaReachDense-v2 --save_interval 0 --n 500" -t 05:00:00
  bash jsubmit.sh -p "Experiments/run_experiment.py jounaid noise --env PandaPushDense-v2 --save_interval 0 --n 500" -t 05:00:00
  bash jsubmit.sh -p "Experiments/run_experiment.py jounaid noise --env PandaReachDense-v2 --save_interval 0 --n 500" -t 05:00:00
  bash jsubmit.sh -p "Experiments/run_experiment.py jounaid ori --env PandaPushDense-v2 --save_interval 0 --n 500" -t 05:00:00
  bash jsubmit.sh -p "Experiments/run_experiment.py jounaid ori --env PandaReachDense-v2 --save_interval 0 --n 500" -t 05:00:00
done


