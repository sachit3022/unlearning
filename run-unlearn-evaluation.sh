# shell script to run benchmark with different seeds for 512 iterations
for n in {1..30}; 
do
    eval "$(conda shell.bash hook)"
    conda activate unl
    echo "Running iteration $n with env $CONDA_DEFAULT_ENV"
    python3 benchmark.py -s $n
done

