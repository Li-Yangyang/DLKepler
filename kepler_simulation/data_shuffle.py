import h5py
from sklearn.model_selection import ShuffleSplit
import numpy as np
import matplotlib.pyplot as plt



for k in np.arange(1):

	#import the spec data here:
	with h5py.File('data_set_for_exoplanet.hdf5', 'r') as hf:
		pos_data = hf["pos"][:]
		neg_data = hf["neg"][:]
		info = hf["info"][:]

	label_pos = np.zeros((pos_data.shape[0],1))+1.0
	label_neg = np.zeros((neg_data.shape[0], 1))

	data_sample = np.concatenate((pos_data, neg_data))
	label = np.concatenate((label_pos, label_neg))
	info_sample = np.concatenate((info, info))


	print(label)
	print(data_sample.shape)
	print(label.shape)


       
	n_splits=1
	test_ratio=0.2
	rs = ShuffleSplit(n_splits=n_splits, test_size=test_ratio)
	count=0
	for train_index, test_index in rs.split(range(data_sample.shape[0])):
		count += 1
		X_train, X_test = data_sample[train_index, :], data_sample[test_index, :]
		y_train, y_test = label[train_index], label[test_index]
		info_train, info_test = info_sample[train_index, :], info_sample[test_index, :]

		trainspec_name='training_data_exo.hdf5'
		with h5py.File(trainspec_name,'w') as hf:
			hf.create_dataset("training_lc",  data=X_train)
			hf.create_dataset("training_label",  data=y_train)
			hf.create_dataset("training_info",  data=info_train)
		testspec_name='test_data_exo.hdf5' 
		with h5py.File(testspec_name,'w') as hf:
			hf.create_dataset("test_lc",  data=X_test)
			hf.create_dataset("test_label",  data=y_test)
			hf.create_dataset("test_info",  data=info_test)
