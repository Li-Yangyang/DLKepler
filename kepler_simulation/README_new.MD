1) run KOI_simulation_quick.py to generate the sample. If you need to modify the simulation, go to kepler_simu_utils.py (L181 L182 L201 L202)
2) run data_shuffle.py to create the training set and test set.
3) run tatooine_net.py to train the neural network(type: CUDA_VISIBLE_DEVICES=0 python tatooine_net.py).
