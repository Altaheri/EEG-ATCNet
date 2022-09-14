# EEG-ATCNet

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/physics-inform-attention-temporal/eeg-4-classes-on-bci-competition-iv-2a)](https://paperswithcode.com/sota/eeg-4-classes-on-bci-competition-iv-2a?p=physics-inform-attention-temporal)

This repository provides code for the Attention Temporal Convolutional Network [(ATCNet)](https://doi.org/10.1109/TII.2022.3197419) proposed in the paper: [Physics-informed attention temporal convolutional network for EEG-based motor imagery classification](https://doi.org/10.1109/TII.2022.3197419)

Authors: Hamdi Altaheri, Ghulam Muhammad, Mansour Alsulaiman

Center of Smart Robotics Research, King Saud University, Saudi Arabia

##
In addition to the proposed [ATCNet](https://doi.org/10.1109/TII.2022.3197419) model, the [*models.py*](https://github.com/Altaheri/EEG-ATCNet/blob/main/models.py) file includes the implementation of other related methods, which can be compared with [ATCNet](https://doi.org/10.1109/TII.2022.3197419), including:
* **EEGNet**, [[paper](https://arxiv.org/abs/1611.08024), [original code](https://github.com/vlawhern/arl-eegmodels)]
* **EEG-TCNet**, [[paper](https://arxiv.org/abs/2006.00622), [original code](https://github.com/iis-eth-zurich/eeg-tcnet)]
* **TCNet_Fusion**, [[paper](https://doi.org/10.1016/j.bspc.2021.102826)]
* **EEGNeX**, [[paper](https://arxiv.org/abs/2207.12369), [original code](https://github.com/chenxiachan/EEGNeX)]
* **DeepConvNet**, [[paper](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730), [original code](https://github.com/braindecode/braindecode)]
* **ShallowConvNet**, [[paper](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730), [original code](https://github.com/braindecode/braindecode)]

##
This repository includes the implementation of the following attention schemes in the [*attention_models.py*](https://github.com/Altaheri/EEG-ATCNet/blob/main/attention_models.py) file: 
* [Multi-head self-attention (mha)](https://arxiv.org/abs/1706.03762)
* [Multi-head attention with locality self-attention (mhla)](https://arxiv.org/abs/2112.13492v1)
* [Squeeze-and-excitation attention (se)](https://arxiv.org/abs/1709.01507)
* [Convolutional block attention module (cbam)](https://arxiv.org/abs/1807.06521)

These attention blocks can be called using the *attention_block(net,  attention_model)* method in the [*attention_models.py*](https://github.com/Altaheri/EEG-ATCNet/blob/main/attention_models.py) file, where *'net'* is the input layer and *'attention_model'* indicates the type of the attention mechanism, which has five options: *None*, [*'mha'*](https://arxiv.org/abs/1706.03762), [*'mhla'*](https://arxiv.org/abs/2112.13492v1), [*'cbam'*](https://arxiv.org/abs/1807.06521), and [*'se'*](https://arxiv.org/abs/1709.01507).
```
Example: 
    input = Input(shape = (10, 100, 1))   
    block1 = Conv2D(1, (1, 10))(input)
    block2 = attention_block(block1,  'mha') # mha: multi-head self-attention
    output = Dense(4, activation="softmax")(Flatten()(block2))
```
##
The [*preprocess.py*](https://github.com/Altaheri/EEG-ATCNet/blob/main/preprocess.py) file loads and divides the dataset based on two approaches: 
1. [Subject-specific (subject-dependent)](https://link.springer.com/article/10.1007/s00521-021-06352-5#Sec9:~:text=Full%20size%20table-,Performance%20evaluation,-For%20the%20MI) approach. In this approach, we used the same training and testing data as the original [BCI-IV-2a](https://www.bbci.de/competition/iv/) competition division, i.e., trials in session 1 for training, and trials in session 2 for testing. 
2. [Leave One Subject Out (LOSO)](https://link.springer.com/article/10.1007/s00521-021-06352-5#Sec9:~:text=Full%20size%20table-,Performance%20evaluation,-For%20the%20MI) approach. LOSO is used for  **Subject-independent** evaluation. In LOSO, the model is trained and evaluated by several folds, equal to the number of subjects, and for each fold, one subject is used for evaluation and the others for training. The LOSO evaluation technique ensures that separate subjects (not visible in the training data) are usedto evaluate the model.

The *get_data()* method in the [*preprocess.py*](https://github.com/Altaheri/EEG-ATCNet/blob/main/preprocess.py) file is used to load the dataset and split it into training and testing. This method uses the [subject-specific](https://link.springer.com/article/10.1007/s00521-021-06352-5#Sec9:~:text=Full%20size%20table-,Performance%20evaluation,-For%20the%20MI) approach by default. If you want to use the [subject-independent (LOSO)](https://link.springer.com/article/10.1007/s00521-021-06352-5#Sec9:~:text=Full%20size%20table-,Performance%20evaluation,-For%20the%20MI) approach, set the parameter *LOSO = True*.


## About ATCNet
ATCNet model consists of three main blocks: 
1. **Convolutional (CV) block**: encodes low-level spatio-temporal information within the MI-EEG signal into a sequence of high-level temporal representations through three convolutional layers. 
2. **Attention (AT) block**: highlights the most important information in the temporal sequence using a multi-head self-attention (MSA). 
3. **Temporal convolutional (TC) block**: extracts high-level temporal features from the highlighted information using a temporal convolutional layer
* [ATCNet](https://doi.org/10.1109/TII.2022.3197419) model also utilizes the convolutional-based sliding window to augment MI data and boost the performance of MI classification efficiently. 

<p align="center">
The components of the ATCNet model
</p>
<p align="center">
<img src="https://user-images.githubusercontent.com/25565236/185448044-17020feb-fd0d-402b-93aa-2942cba9b8af.png" alt="The components of the proposed ATCNet model" width="300"/>
</p>
<p align="center">
Visualize the transition of data in the ATCNet model.
</p>
<p align="center">
<img src="https://user-images.githubusercontent.com/25565236/185449791-e8539453-d4fa-41e1-865a-2cf7e91f60ef.png" alt="The components of the proposed ATCNet model" width="500"/>
</p>

## Development environment
Models were trained and tested by a single GPU, Nvidia [GTX 2070 8GB](https://www.nvidia.com/en-me/geforce/graphics-cards/rtx-2070/) (Driver Version: [512.78](https://www.nvidia.com/download/driverResults.aspx/188599/en-us/), [CUDA 11.3](https://developer.nvidia.com/cuda-11.3.0-download-archive)), using Python 3.7 with [TensorFlow](https://www.tensorflow.org/) framework. [Anaconda 3](https://www.anaconda.com/products/distribution) was used on [Ubuntu 20.04.4 LTS](https://releases.ubuntu.com/20.04/) and [Windows 11](https://www.microsoft.com/en-hk/software-download/windows11).
The following packages are required:
* TensorFlow 2.7
* matplotlib 3.5
* NumPy 1.20
* scikit-learn 1.0
* SciPy 1.7

## Dataset 
The [BCI Competition IV-2a](http://www.bbci.de/competition/iv/#dataset2a) dataset needs to be downloaded and the data path placed at 'data_path' variable in [*main.py*](https://github.com/Altaheri/EEG-ATCNet/blob/main/main.py) file. The dataset can be downloaded from [here](http://bnci-horizon-2020.eu/database/data-sets).

## References
If you find this work useful in your research, please use the following BibTeX entry for citation

```
@article{9852687,
  title={Physics-informed attention temporal convolutional network for EEG-based motor imagery classification}, 
  author={Altaheri, Hamdi and Muhammad, Ghulam and Alsulaiman, Mansour},
  journal={IEEE Transactions on Industrial Informatics}, 
  year={2022},
  doi={10.1109/TII.2022.3197419}
  }
  
@article{altaheri2021deep,
  title={Deep learning techniques for classification of electroencephalogram (EEG) motor imagery (MI) signals: a review},
  author={Altaheri, Hamdi and Muhammad, Ghulam and Alsulaiman, Mansour and Amin, Syed Umar and Altuwaijri, Ghadir Ali and Abdul, Wadood and Bencherif, Mohamed A and Faisal, Mohammed},
  journal={Neural Computing and Applications},
  pages={1--42},
  year={2021},
  publisher={Springer}
}
```

