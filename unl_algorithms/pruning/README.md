# Unlearning by Pruning

To make sure the algorithm can be run:
```bash
export PYTHONPATH=$PWD:$PYTHONPATH
DEVICE=<enter gpu id>
CUDA_VISIBLE_DEVICES=$DEVICE python unl_algorithms/pruning/example_run.py
```