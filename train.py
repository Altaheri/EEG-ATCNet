""" 
Hamdi Altaheri, King Saud University

hamdi.altahery@gmail.com 

"""

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import time

import os
from models import *
from TCnetPreprocess import prepare_features
# -----------------------------------------------------------------------------        


# -----------------------------------------------------------------------------   
# Get file paths
# data_path = os.path.expanduser('~') + '/BCI Competition IV/BCI Competition IV-2a/BCI Competition IV 2a mat/'
data_path = data_path = 'D:/5. Datasets/1. EEG datasets/BCI Competition IV/BCI Competition IV-2a/BCI Competition IV 2a mat/'
results_path = os.getcwd() + "/results"
if not  os.path.exists(results_path):
  # Create a new directory because it does not exist 
  os.makedirs(results_path)
  
# Hyperparamters : ------------------------------------------------------------
isStandard = True
classes = 4
channels = 22
LOSO = False
batch_size = 64
epochs = 5
patience = 300
lr = 0.0009 #0.001
dropout = 0.3
LearnCurves = True #Plot Learning Curves?
# -----------------------------------------------------------------------------   
num_train = 10
num_sub = 9

acc = np.zeros((num_sub, num_train))
kappa = np.zeros((num_sub, num_train))

# -----------------------------------------------------------------------------   
log_write = open(results_path + "/log.txt", "w")
# -----------------------------------------------------------------------------   

# Training 
in_exp = time.time()

for sub in range(num_sub): # (9) (for all subjects), (i-1,i) for the ith subject.
  in_sub = time.time()
  print('Training on subject ', sub+1)
  log_write.write( '\nTraining on subject '+ str(sub+1) +'\n')
  
  BestSubjAcc = 0 # variable to save best subject Accuracy.
  bestTrainingHistory = [] 
  
  X_train, _, y_train_onehot, X_test, _, y_test_onehot = prepare_features(data_path, sub, LOSO)
  #scipy.io.savemat('data/A0%d.mat' % (sub+1), mdict={'X_train': X_train, 'y_train': y_train_onehot, 'X_test': X_test, 'y_test': y_test_onehot})

     
  if( isStandard == True):
       for j in range(22):
             scaler = StandardScaler()
             scaler.fit(X_train[:,0,j,:])
             X_train[:,0,j,:] = scaler.transform(X_train[:,0,j,:])
             X_test[:,0,j,:] = scaler.transform(X_test[:,0,j,:])
  opt = Adam(learning_rate=lr)
  
  for train in range(num_train): # How many repetitions of training for subject i.
        in_run = time.time()
        # model = TCNet_Fusion(n_classes = 4)      # train using TCNet_Fusion: https://doi.org/10.1016/j.bspc.2021.102826
        # model = EEGTCNet(n_classes = 4)          # train using EEGTCNet: https://arxiv.org/abs/2006.00622
        # model = EEGNet_classifier(n_classes = 4) # train using EEGNet: https://arxiv.org/abs/1611.08024

        # train using our Proposed model (ATCNet): 
        model = ATCNet(
            n_classes = 4, 
            in_chans = 22, 
            in_samples = 1125, 
            n_windows = 5, 
            attention = 'mha', # Options: None, 'mha', 'cbam', 'se'
            # eegn
            eegn_F1 = 16,
            eegn_D = 2, 
            eegn_kernelSize = 64,
            eegn_poolSize = 7,
            eegn_dropout = dropout,
            # tcn
            tcn_depth = 2, 
            tcn_kernelSize = 4,
            tcn_filters = 32,
            tcn_dropout = dropout, 
            tcn_activation='elu'
            )     

        # ---------------------------------------------------------------------       
        model.compile(loss=categorical_crossentropy,optimizer=opt, metrics=['accuracy'])
        # model.summary() 
        
        filepath = results_path + '/saved models/run-{}'.format(train+1)
        if not  os.path.exists(filepath):
            os.makedirs(filepath)        
        filepath = filepath + '/subject-{}.h5'.format(sub+1)
        
        cp = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=True, 
                             save_weights_only=True, mode='max')
        es = EarlyStopping(monitor='val_accuracy', verbose=1, mode='max', patience=patience)
        history = model.fit(X_train, y_train_onehot, validation_data=(X_test, y_test_onehot), 
                            epochs=epochs, batch_size=batch_size, callbacks=[cp,es], verbose=0)
        
        # ---------------------------------------------------------------------       
        model.load_weights(filepath)
        y_pred = model.predict(X_test).argmax(axis=-1)
        labels = y_test_onehot.argmax(axis=-1)
        acc[sub, train]  = accuracy_score(labels, y_pred)
        kappa[sub, train] = cohen_kappa_score(labels, y_pred)

        out_run = time.time()
        info = 'Subject: {}   Train no. {}   Time: {:.1f} m   '.format(sub+1, train+1, ((out_run-in_run)/60))
        info = info + 'Test_acc: {:.4f}   Test_kappa: {:.4f}'.format(acc[sub, train], kappa[sub, train])
        print(info)
        log_write.write(info +'\n')
        
        # If current training run is better than previous runs, save the history.
        if(BestSubjAcc < acc[sub, train]):
             BestSubjAcc = acc[sub, train]
             bestTrainingHistory = history
        # ---------------------------------------------------------------------       
  
  # ---------------------------------------------------------------------------
  history = bestTrainingHistory 
  
  out_sub = time.time()
  best_run = np.argmax(acc[sub,:])
  info = 'Subject: {}   best_run: {}   Time: {:.1f} m   '.format(sub+1, best_run+1, ((out_sub-in_sub)/60))
  info = info + 'acc: {:.4f}   avg_acc: {:.4f} +- {:.4f}   '.format(acc[sub, best_run], np.average(acc[sub, :]), acc[sub,:].std() )
  info = info + 'kappa: {:.4f}   avg_kappa: {:.4f} +- {:.4f}'.format(kappa[sub, best_run], np.average(kappa[sub, :]), kappa[sub,:].std())
  print(info)
  log_write.write(info+'\n')
  
  # ---------------------------------------------------------------------------       
  # Plot Learning curves 
  if (LearnCurves == True):
    print('Plot Learning Curves ....... \n')
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.show()
    plt.close()
    
  # ---------------------------------------------       
  # Find the metrics
  cf_matrix = confusion_matrix(labels, y_pred, normalize='pred')
  # Generate confusion matrix plot
  display_labels = ['Left hand', 'Right hand','Foot','Tongue']
  disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, 
                              display_labels=display_labels)
  disp.plot()
  disp.ax_.set_xticklabels(display_labels, rotation=12)
  plt.title('Confusion Matrix of Subject: {}'.format(sub+1) )
  plt.savefig(results_path + '/subject{}.png'.format(sub+1))
  plt.show()
  # ---------------------------------------------------------------------------       

out_exp = time.time()
info = 'Time: {:.1f} h   '.format( (out_exp-in_exp)/(60*60) )
print(info)
log_write.write(info+'\n')



# Evaluation 
test_acc = np.zeros(num_sub)
test_kappa = np.zeros(num_sub)
cf_matrix = np.zeros([classes,classes])
evalute_from_file = False

if(evalute_from_file):
    # READ:
    best_models = open(results_path + "/best models.txt", "r")
else:
    # WRITE:
    best_models = open(results_path + "/best models.txt", "w")

for sub in range(num_sub): # (9) (for all subjects), (i-1,i) for the ith subject.

  X_train, _, _, X_test, _, y_test_onehot = prepare_features(data_path, sub, LOSO)
  
  if( isStandard == True):
    for j in range(22):
      scaler = StandardScaler()
      scaler.fit(X_train[:,0,j,:])
      X_test[:,0,j,:] = scaler.transform(X_test[:,0,j,:])
  
  if(evalute_from_file):
      # READ:
      filepath = best_models.readline()
  else:
      # WRITE:
       filepath = '/saved models/run-{}/subject-{}.h5'.format(np.argmax(acc[sub,:])+1, sub+1)+"\n"
       best_models.write(filepath)

  model.load_weights(results_path + filepath[:-1])

  y_pred = model.predict(X_test).argmax(axis=-1)
  labels = y_test_onehot.argmax(axis=-1)
  test_acc[sub] = accuracy_score(labels, y_pred)
  test_kappa[sub] = cohen_kappa_score(labels, y_pred)
  
  cf_matrix = cf_matrix + confusion_matrix(labels, y_pred, normalize='pred')
  
  info = 'Subject: {}   best_run: {}   '.format(sub+1, (np.argmax(acc[sub,:])+1) )
  info = info + 'acc: {:.4f}   avg_acc: {:.4f} +- {:.4f}   '.format(test_acc[sub], np.average(acc[sub, :]), acc[sub,:].std() )
  info = info + 'kappa: {:.4f}   avg_kappa: {:.4f} +- {:.4f}'.format(test_kappa[sub], np.average(kappa[sub, :]), kappa[sub,:].std() )
  print(info)
  if(not log_write.closed):
      log_write.write('\n'+info +'\n')
  
# Calculate Average Accuracy of all subjects 
info = 'Average of all subjects: Accuracy = {:.4f}   Kappa = {:.4f}\n'.format(np.average(test_acc), np.average(test_kappa) ) 
info = info + 'Average of all subjects for all runs: Accuracy = {:.4f}'.format(np.average(acc)) 
info = info + '   Kappa = {:.4f} '.format(np.average(kappa)) 

print(info)
if(not log_write.closed):
    log_write.write(info)

# print acc fogure for all subjects  
fig2, ax = plt.subplots()
x = list(range(1, num_sub+1))
rects1 = ax.bar(x, test_acc, 0.5, label='Acc')
ax.set_ylabel("Test Accuracy")
ax.set_xlabel("Subject")
ax.set_xticks(x)
ax.set_title('Model Accuracies per subject')
ax.set_ylim([0,1])

cf_matrix = cf_matrix / num_sub
# Generate confusion matrix plot
display_labels = ['Left hand', 'Right hand','Foot','Tongue']
disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, 
                            display_labels=display_labels)
disp.plot()
disp.ax_.set_xticklabels(display_labels, rotation=12)
plt.title('Confusion Matrix of All Subjects')
plt.savefig(results_path + '/all subjects.png')
plt.show()


log_write.close() 
best_models.close()   
