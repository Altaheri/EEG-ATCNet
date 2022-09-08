""" 
Copyright (C) 2022 King Saud University, Saudi Arabia 
SPDX-License-Identifier: Apache-2.0 

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the 
License at

http://www.apache.org/licenses/LICENSE-2.0  

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. 

Author:  Hamdi Altaheri 
"""

#%%
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, AveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Conv1D, Conv2D, SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization, LayerNormalization, Flatten 
from tensorflow.keras.layers import Add, Concatenate, Lambda, Input, Permute
from tensorflow.keras.constraints import max_norm

from tensorflow.keras import backend as K
from attention_models import attention_block
        
#%% The proposed ATCNet model, https://doi.org/10.1109/TII.2022.3197419
def ATCNet(n_classes, in_chans = 22, in_samples = 1125, n_windows = 3, attention = None, 
           eegn_F1 = 16, eegn_D = 2, eegn_kernelSize = 64, eegn_poolSize = 8, eegn_dropout=0.3, 
           tcn_depth = 2, tcn_kernelSize = 4, tcn_filters = 32, tcn_dropout = 0.3, 
           tcn_activation = 'elu', fuse = 'average'):
    """ ATCNet model from Altaheri et al 2022.
        See details at https://ieeexplore.ieee.org/abstract/document/9852687
    
        Notes
        -----
        The initial values in this model are based on the values identified by
        the authors
        
        References
        ----------
        .. H. Altaheri, G. Muhammad and M. Alsulaiman, "Physics-informed 
           attention temporal convolutional network for EEG-based motor imagery 
           classification," in IEEE Transactions on Industrial Informatics, 2022, 
           doi: 10.1109/TII.2022.3197419.
    """
    input_1 = Input(shape = (1,in_chans, in_samples))   #     TensorShape([None, 1, 22, 1125])
    input_2 = Permute((3,2,1))(input_1) 
    regRate=.25
    numFilters = eegn_F1
    F2 = numFilters*eegn_D

    block1 = Conv_block(input_layer = input_2, F1 = eegn_F1, D = eegn_D, 
                        kernLength = eegn_kernelSize, poolSize = eegn_poolSize,
                        in_chans = in_chans, dropout = eegn_dropout)
    block1 = Lambda(lambda x: x[:,:,-1,:])(block1)
     
    # Sliding window 
    sw_concat = []   # to store concatenated or averaged sliding window outputs
    for i in range(n_windows):
        st = i
        end = block1.shape[1]-n_windows+i+1
        block2 = block1[:, st:end, :]
        
        # Attention_model
        if attention is not None:
            block2 = attention_block(block2, attention)

        # Temporal convolutional network (TCN)
        block3 = TCN_block(input_layer = block2, input_dimension = F2, depth = tcn_depth,
                            kernel_size = tcn_kernelSize, filters = tcn_filters, 
                            dropout = tcn_dropout, activation = tcn_activation)
        # Get feature maps of the last sequence
        block3 = Lambda(lambda x: x[:,-1,:])(block3)
        
        # Outputs of sliding window: Average_after_dense or concatenate_then_dense
        if(fuse == 'average'):
            sw_concat.append(Dense(n_classes, kernel_constraint = max_norm(regRate))(block3))
        elif(fuse == 'concat'):
            if i == 0:
                sw_concat = block3
            else:
                sw_concat = Concatenate()([sw_concat, block3])
                
    if(fuse == 'average'):
        if len(sw_concat) > 1: # more than one window
            sw_concat = tf.keras.layers.Average()(sw_concat[:])
        else: # one window (# windows = 1)
            sw_concat = sw_concat[0]
    elif(fuse == 'concat'):
        sw_concat = Dense(n_classes, kernel_constraint = max_norm(regRate))(sw_concat)
            
    
    softmax = Activation('softmax', name = 'softmax')(sw_concat)
    
    return Model(inputs = input_1, outputs = softmax)

#%% Convolutional (CV) block used in the ATCNet model
def Conv_block(input_layer, F1=4, kernLength=64, poolSize=8, D=2, in_chans=22, dropout=0.1):
    """ Conv_block
    
        Notes
        -----
        This block is the same as EEGNet with SeparableConv2D replaced by Conv2D 
        The original code for this model is available at: https://github.com/vlawhern/arl-eegmodels
        See details at https://arxiv.org/abs/1611.08024
    """
    F2= F1*D
    block1 = Conv2D(F1, (kernLength, 1), padding = 'same',data_format='channels_last',use_bias = False)(input_layer)
    block1 = BatchNormalization(axis = -1)(block1)
    block2 = DepthwiseConv2D((1, in_chans), use_bias = False, 
                                    depth_multiplier = D,
                                    data_format='channels_last',
                                    depthwise_constraint = max_norm(1.))(block1)
    block2 = BatchNormalization(axis = -1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((8,1),data_format='channels_last')(block2)
    block2 = Dropout(dropout)(block2)
    block3 = Conv2D(F2, (16, 1),
                            data_format='channels_last',
                            use_bias = False, padding = 'same')(block2)
    block3 = BatchNormalization(axis = -1)(block3)
    block3 = Activation('elu')(block3)
    
    block3 = AveragePooling2D((poolSize,1),data_format='channels_last')(block3)
    block3 = Dropout(dropout)(block3)
    return block3

#%% Temporal convolutional (TC) block used in the ATCNet model
def TCN_block(input_layer,input_dimension,depth,kernel_size,filters,dropout,activation='relu'):
    """ TCN_block from Bai et al 2018
        Temporal Convolutional Network (TCN)
        
        Notes
        -----
        THe original code available at https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
        This implementation has a slight modification from the original code
        and it is taken from the code by Ingolfsson et al at https://github.com/iis-eth-zurich/eeg-tcnet
        See details at https://arxiv.org/abs/2006.00622

        References
        ----------
        .. Bai, S., Kolter, J. Z., & Koltun, V. (2018).
           An empirical evaluation of generic convolutional and recurrent networks
           for sequence modeling.
           arXiv preprint arXiv:1803.01271.
    """    
    
    block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=1,activation='linear',
                   padding = 'causal',kernel_initializer='he_uniform')(input_layer)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=1,activation='linear',
                   padding = 'causal',kernel_initializer='he_uniform')(block)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    if(input_dimension != filters):
        conv = Conv1D(filters,kernel_size=1,padding='same')(input_layer)
        added = Add()([block,conv])
    else:
        added = Add()([block,input_layer])
    out = Activation(activation)(added)
    
    for i in range(depth-1):
        block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=2**(i+1),activation='linear',
                   padding = 'causal',kernel_initializer='he_uniform')(out)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=2**(i+1),activation='linear',
                   padding = 'causal',kernel_initializer='he_uniform')(block)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        added = Add()([block, out])
        out = Activation(activation)(added)
        
    return out


#%% Reproduced TCNet_Fusion model: https://doi.org/10.1016/j.bspc.2021.102826
def TCNet_Fusion(n_classes,Chans=22, Samples=1125, layers=2, kernel_s=4, filt=12,
                 dropout=0.3, activation='elu', F1=24, D=2, kernLength=32, dropout_eeg=0.3):
    """ TCNet_Fusion model from Musallam et al 2021.
    See details at https://doi.org/10.1016/j.bspc.2021.102826
    
        Notes
        -----
        The initial values in this model are based on the values identified by
        the authors
        
        References
        ----------
        .. Musallam, Y.K., AlFassam, N.I., Muhammad, G., Amin, S.U., Alsulaiman,
           M., Abdul, W., Altaheri, H., Bencherif, M.A. and Algabri, M., 2021. 
           Electroencephalography-based motor imagery classification
           using temporal convolutional network fusion. 
           Biomedical Signal Processing and Control, 69, p.102826.
    """
    input1 = Input(shape = (1,Chans, Samples))
    input2 = Permute((3,2,1))(input1)
    regRate=.25

    numFilters = F1
    F2= numFilters*D
    
    EEGNet_sep = EEGNet(input_layer=input2,F1=F1,kernLength=kernLength,D=D,Chans=Chans,dropout=dropout_eeg)
    block2 = Lambda(lambda x: x[:,:,-1,:])(EEGNet_sep)
    FC = Flatten()(block2) 

    outs = TCN_block(input_layer=block2,input_dimension=F2,depth=layers,kernel_size=kernel_s,filters=filt,dropout=dropout,activation=activation)

    Con1 = Concatenate()([block2,outs]) 
    out = Flatten()(Con1) 
    Con2 = Concatenate()([out,FC]) 
    dense        = Dense(n_classes, name = 'dense',kernel_constraint = max_norm(regRate))(Con2)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1,outputs=softmax)

#%% Reproduced EEGTCNet model: https://arxiv.org/abs/2006.00622
def EEGTCNet(n_classes, Chans=22, Samples=1125, layers=2, kernel_s=4, filt=12, dropout=0.3, activation='elu', F1=8, D=2, kernLength=32, dropout_eeg=0.2):
    """ EEGTCNet model from Ingolfsson et al 2020.
    See details at https://arxiv.org/abs/2006.00622
    
    The original code for this model is available at https://github.com/iis-eth-zurich/eeg-tcnet
    
        Notes
        -----
        The initial values in this model are based on the values identified by the authors
        
        References
        ----------
        .. Ingolfsson, T. M., Hersche, M., Wang, X., Kobayashi, N.,
           Cavigelli, L., & Benini, L. (2020, October). 
           Eeg-tcnet: An accurate temporal convolutional network
           for embedded motor-imagery brain–machine interfaces. 
           In 2020 IEEE International Conference on Systems, 
           Man, and Cybernetics (SMC) (pp. 2958-2965). IEEE.
    """
    input1 = Input(shape = (1,Chans, Samples))
    input2 = Permute((3,2,1))(input1)
    regRate=.25
    numFilters = F1
    F2= numFilters*D

    EEGNet_sep = EEGNet(input_layer=input2,F1=F1,kernLength=kernLength,D=D,Chans=Chans,dropout=dropout_eeg)
    block2 = Lambda(lambda x: x[:,:,-1,:])(EEGNet_sep)
    outs = TCN_block(input_layer=block2,input_dimension=F2,depth=layers,kernel_size=kernel_s,filters=filt,dropout=dropout,activation=activation)
    out = Lambda(lambda x: x[:,-1,:])(outs)
    dense        = Dense(n_classes, name = 'dense',kernel_constraint = max_norm(regRate))(out)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1,outputs=softmax)

#%% Reproduced EEGNeX model: https://arxiv.org/abs/2207.12369
def EEGNeX_8_32(n_timesteps, n_features, n_outputs):
    """ EEGNeX model from Chen et al 2022.
    See details at https://arxiv.org/abs/2207.12369
    
    The original code for this model is available at https://github.com/chenxiachan/EEGNeX
           
        References
        ----------
        .. Chen, X., Teng, X., Chen, H., Pan, Y., & Geyer, P. (2022).
           Toward reliable signals decoding for electroencephalogram: 
           A benchmark study to EEGNeX. arXiv preprint arXiv:2207.12369.
    """

    model = Sequential()
    model.add(Input(shape=(1, n_features, n_timesteps)))

    model.add(Conv2D(filters=8, kernel_size=(1, 32), use_bias = False, padding='same', data_format="channels_first"))
    model.add(LayerNormalization())
    model.add(Activation(activation='elu'))
    model.add(Conv2D(filters=32, kernel_size=(1, 32), use_bias = False, padding='same', data_format="channels_first"))
    model.add(LayerNormalization())
    model.add(Activation(activation='elu'))

    model.add(DepthwiseConv2D(kernel_size=(n_features, 1), depth_multiplier=2, use_bias = False, depthwise_constraint=max_norm(1.), data_format="channels_first"))
    model.add(LayerNormalization())
    model.add(Activation(activation='elu'))
    model.add(AveragePooling2D(pool_size=(1, 4), padding='same', data_format="channels_first"))
    model.add(Dropout(0.5))

    
    model.add(Conv2D(filters=32, kernel_size=(1, 16), use_bias = False, padding='same', dilation_rate=(1, 2), data_format='channels_first'))
    model.add(LayerNormalization())
    model.add(Activation(activation='elu'))
    
    model.add(Conv2D(filters=8, kernel_size=(1, 16), use_bias = False, padding='same', dilation_rate=(1, 4),  data_format='channels_first'))
    model.add(LayerNormalization())
    model.add(Activation(activation='elu'))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(n_outputs, kernel_constraint=max_norm(0.25)))
    model.add(Activation(activation='softmax'))
    
    # save a plot of the model
    # plot_model(model, show_shapes=True, to_file='EEGNeX_8_32.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model 

#%% Reproduced EEGNet model: https://arxiv.org/abs/1611.08024
def EEGNet_classifier(n_classes, Chans=22, Samples=1125, F1=8, D=2, kernLength=64, dropout_eeg=0.25):
    input1 = Input(shape = (1,Chans, Samples))   
    input2 = Permute((3,2,1))(input1) 
    regRate=.25

    eegnet = EEGNet(input_layer=input2, F1=F1, kernLength=kernLength, D=D, Chans=Chans, dropout=dropout_eeg)
    eegnet = Flatten()(eegnet)
    dense = Dense(n_classes, name = 'dense',kernel_constraint = max_norm(regRate))(eegnet)
    softmax = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)

def EEGNet(input_layer, F1=8, kernLength=64, D=2, Chans=22, dropout=0.25):
    """ EEGNet model from Lawhern et al 2018
    See details at https://arxiv.org/abs/1611.08024
    
    The original code for this model is available at: https://github.com/vlawhern/arl-eegmodels
    
        Notes
        -----
        The initial values in this model are based on the values identified by the authors
        
        References
        ----------
        .. Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon,
           S. M., Hung, C. P., & Lance, B. J. (2018).
           EEGNet: A Compact Convolutional Network for EEG-based
           Brain-Computer Interfaces.
           arXiv preprint arXiv:1611.08024.
    """
    F2= F1*D
    block1 = Conv2D(F1, (kernLength, 1), padding = 'same',data_format='channels_last',use_bias = False)(input_layer)
    block1 = BatchNormalization(axis = -1)(block1)
    block2 = DepthwiseConv2D((1, Chans), use_bias = False, 
                                    depth_multiplier = D,
                                    data_format='channels_last',
                                    depthwise_constraint = max_norm(1.))(block1)
    block2 = BatchNormalization(axis = -1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((8,1),data_format='channels_last')(block2)
    block2 = Dropout(dropout)(block2)
    block3 = SeparableConv2D(F2, (16, 1),
                            data_format='channels_last',
                            use_bias = False, padding = 'same')(block2)
    block3 = BatchNormalization(axis = -1)(block3)
    block3 = Activation('elu')(block3)
    block3 = AveragePooling2D((8,1),data_format='channels_last')(block3)
    block3 = Dropout(dropout)(block3)
    return block3

#%% Reproduced DeepConvNet model: https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730
def DeepConvNet(nb_classes, Chans = 64, Samples = 256,
                dropoutRate = 0.5):
    """ Keras implementation of the Deep Convolutional Network as described in
    Schirrmeister et. al. (2017), Human Brain Mapping.
    See details at https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730
    
    The original code for this model is available at:
        https://github.com/braindecode/braindecode
    
    This implementation is taken from code by the Army Research Laboratory (ARL) 
    at https://github.com/vlawhern/arl-eegmodels
    
    This implementation assumes the input is a 2-second EEG signal sampled at 
    128Hz, as opposed to signals sampled at 250Hz as described in the original
    paper. We also perform temporal convolutions of length (1, 5) as opposed
    to (1, 10) due to this sampling rate difference. 
    
    Note that we use the max_norm constraint on all convolutional layers, as 
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication 
    with the original authors.
    
                      ours        original paper
    pool_size        1, 2        1, 3
    strides          1, 2        1, 3
    conv filters     1, 5        1, 10
    
    Note that this implementation has not been verified by the original 
    authors. 
    
    """

    # start the model
    # input_main   = Input((Chans, Samples, 1))
    input_main   = Input((1, Chans, Samples))
    input_2 = Permute((2,3,1))(input_main) 
    
    block1       = Conv2D(25, (1, 5), 
                                 input_shape=(Chans, Samples, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(input_2)
    block1       = Conv2D(25, (Chans, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
    block1       = Activation('elu')(block1)
    block1       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
    block1       = Dropout(dropoutRate)(block1)
  
    block2       = Conv2D(50, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block2       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block2)
    block2       = Activation('elu')(block2)
    block2       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
    block2       = Dropout(dropoutRate)(block2)
    
    block3       = Conv2D(100, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block2)
    block3       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block3)
    block3       = Activation('elu')(block3)
    block3       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
    block3       = Dropout(dropoutRate)(block3)
    
    block4       = Conv2D(200, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block3)
    block4       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block4)
    block4       = Activation('elu')(block4)
    block4       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
    block4       = Dropout(dropoutRate)(block4)
    
    flatten      = Flatten()(block4)
    
    dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax      = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)

#%% need these for ShallowConvNet
def square(x):
    return K.square(x)

def log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000))   

#%% Reproduced ShallowConvNet model: https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730
def ShallowConvNet(nb_classes, Chans = 64, Samples = 128, dropoutRate = 0.5):
    """ Keras implementation of the Shallow Convolutional Network as described
    in Schirrmeister et. al. (2017), Human Brain Mapping.
    See details at https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730
    
    The original code for this model is available at:
        https://github.com/braindecode/braindecode

    This implementation is taken from code by the Army Research Laboratory (ARL) 
    at https://github.com/vlawhern/arl-eegmodels

    Assumes the input is a 2-second EEG signal sampled at 128Hz. Note that in 
    the original paper, they do temporal convolutions of length 25 for EEG
    data sampled at 250Hz. We instead use length 13 since the sampling rate is 
    roughly half of the 250Hz which the paper used. The pool_size and stride
    in later layers is also approximately half of what is used in the paper.
    
    Note that we use the max_norm constraint on all convolutional layers, as 
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication 
    with the original authors.
    
                     ours        original paper
    pool_size        1, 35       1, 75
    strides          1, 7        1, 15
    conv filters     1, 13       1, 25    
    
    Note that this implementation has not been verified by the original 
    authors. We do note that this implementation reproduces the results in the
    original paper with minor deviations. 
    """

    # start the model
    # input_main   = Input((Chans, Samples, 1))
    input_main   = Input((1, Chans, Samples))
    input_2 = Permute((2,3,1))(input_main) 

    block1       = Conv2D(40, (1, 13), 
                                 input_shape=(Chans, Samples, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(input_2)
    block1       = Conv2D(40, (Chans, 1), use_bias=False, 
                          kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
    block1       = Activation(square)(block1)
    block1       = AveragePooling2D(pool_size=(1, 35), strides=(1, 7))(block1)
    block1       = Activation(log)(block1)
    block1       = Dropout(dropoutRate)(block1)
    flatten      = Flatten()(block1)
    dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax      = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)