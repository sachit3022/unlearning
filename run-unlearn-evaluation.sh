# shell script to run benchmark with different seeds for 512 iterations
for k in {1..6};
do
    for d in {1..7};
    do
        for i in {0..0};
        do
            echo "Running iteration with seed $((1000*$k+ 90*$d+ 90*8*$i))"
            /research/hal-gaudisac/anaconda3/bin/python3 benchmark.py -s $((1000*$k+ 80*$d+ 80*8*$i)) -d "cuda:$d" -exp full_train &
            sleep 5
        done
    done
    wait 
done