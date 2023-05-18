# run jsubmit for each of the 50 indices
for i in {0..49}
do
  bash jsubmit.sh -p "Experiments/fine_tune.py $i" -t 24:00:00
done

# run 10 iterations of each model type
for i in {0..9}
do
  bash jsubmit.sh -p "Experiments/run_experiment.py var_ppo_models var_pred --save_interval 0 --n 500" -t 05:00:00
done

for i in {0..9}
do
  bash jsubmit.sh -p "Experiments/run_experiment.py var_ppo_models var_no_pred --save_interval 0 --n 500" -t 05:00:00
done

for i in {0..9}
do
  bash jsubmit.sh -p "Experiments/run_experiment.py var_ppo_models abs --save_interval 0 --n 500" -t 05:00:00
done

for i in {0..9}
do
  bash jsubmit.sh -p "Experiments/run_experiment.py var_ppo_models noise --save_interval 0 --n 500" -t 05:00:00
done

for i in {0..9}
do
  bash jsubmit.sh -p "Experiments/run_experiment.py var_ppo_models ori --save_interval 0 --n 500" -t 05:00:00
done
