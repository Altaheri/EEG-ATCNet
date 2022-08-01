""" 
Hamdi Altaheri, King Saud University

hamdi.altahery@gmail.com 

"""

#	Loads the dataset 2a of the BCI Competition IV
# available on http://bnci-horizon-2020.eu/database/data-sets

import numpy as np
import scipy.io as sio
from tensorflow.keras.utils import to_categorical


def get_data_LOSO (data_path, sub): 
    X_train, y_train = [], []
    for subject in range (0,9):
        path = data_path+'s' + str(subject+1) + '/'
        
        X1, y1 = get_data(subject+1, True ,path)
        X2, y2 = get_data(subject+1, False ,path)
        X = np.concatenate((X1, X2), axis=0)
        y = np.concatenate((y1, y2), axis=0)
                   
        if (subject == sub):
            X_test = X
            y_test = y
        elif (X_train == []):
            X_train = X
            y_train = y
        else:
            X_train = np.concatenate((X_train, X), axis=0)
            y_train = np.concatenate((y_train, y), axis=0)
                

    return X_train, y_train, X_test, y_test

def get_data(subject, training, path):

	# Keyword arguments:
	# subject -- number of subject in [1, .. ,9]
	# training -- if True, load training data
	# 			if False, load testing data
	
	# Return:	data_return 	numpy matrix 	size = NO_valid_trial x 22 x 1750
	# 		class_return 	numpy matrix 	size = NO_valid_trial
	
	NO_channels = 22
	NO_tests = 6*48 	
	Window_Length = 7*250 

	class_return = np.zeros(NO_tests)
	data_return = np.zeros((NO_tests,NO_channels,Window_Length))

	NO_valid_trial = 0
	if training:
		a = sio.loadmat(path+'A0'+str(subject)+'T.mat')
	else:
		a = sio.loadmat(path+'A0'+str(subject)+'E.mat')
	a_data = a['data']
	for ii in range(0,a_data.size):
		a_data1 = a_data[0,ii]
		a_data2= [a_data1[0,0]]
		a_data3= a_data2[0]
		a_X 		= a_data3[0]
		a_trial 	= a_data3[1]
		a_y 		= a_data3[2]
# 		a_artifacts = a_data3[5]

		for trial in range(0,a_trial.size):
 			# if(a_artifacts[trial]==0):
				 data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+Window_Length),:22])
				 class_return[NO_valid_trial] = int(a_y[trial])
				 NO_valid_trial +=1


	return data_return[0:NO_valid_trial,:,:], class_return[0:NO_valid_trial]

def prepare_features(path, subject, LOSO=False):

    fs = 250 
    t1 = int(1.5*fs)
    t2 = int(6*fs)
    T = t2-t1
    if LOSO:
        X_train, y_train, X_test, y_test = get_data_LOSO(path, subject)
    else:
        path = path + 's{:}/'.format(subject+1)
        X_train, y_train = get_data(subject+1, True, path)
        X_test, y_test = get_data(subject+1, False, path)

    # prepare training data 	
    N_tr, N_ch,_ = X_train.shape 
    X_train = X_train[:,:,t1:t2].reshape(N_tr,1,N_ch,T)
    y_train_onehot = (y_train-1).astype(int)
    y_train_onehot = to_categorical(y_train_onehot)
    # prepare testing data 
    N_test,N_ch,_ =X_test.shape 
    X_test = X_test[:,:,t1:t2].reshape(N_test,1,N_ch,T)
    y_test_onehot = (y_test-1).astype(int)
    y_test_onehot = to_categorical(y_test_onehot)	

    return X_train,y_train,y_train_onehot,X_test,y_test,y_test_onehot
