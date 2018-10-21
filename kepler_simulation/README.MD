1) run KOI_simulation_training.py to generate the training set for neural network
2) To modfity the parameter distribution of input planet population, change L117and L118 in kepler_simu_utils.py

In KOI_simulation_training.py
3) Change L33 to the number of light curve you want to train for NN.
4) After L125, save your data and label by yourself. Then you can feed in the data to the neural network you create.
