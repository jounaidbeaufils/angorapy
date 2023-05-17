# run jsubmit for each of the 50 indices
for i in {0..49}
do
  bash jsubmit.sh -p "Experiments/fine_tune.py $i" -t 24:00:00
done