# shell script to run benchmark with different seeds for 512 iterations
for n in {1..7};
do
    for d in {1..7};
    do
        for i in {0..0};
        do
            eval "$(conda shell.bash hook)"
            echo "Running iteration with seed $(($n+ 90*$d+ 90*8*$i))"
            /research/hal-gaudisac/anaconda3/bin/python3 benchmark_cifar.py -s $(($n+ 90*$d+ 90*8*$i)) -d "cuda:$d" -cf "cifar_config.yaml" &
            sleep 5
        done
    done
    wait 
done