# EEG-ATCNet

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/physics-inform-attention-temporal/eeg-4-classes-on-bci-competition-iv-2a)](https://paperswithcode.com/sota/eeg-4-classes-on-bci-competition-iv-2a?p=physics-inform-attention-temporal)

This repository provides code for the Attention Temporal Convolutional Network [(ATCNet)](https://doi.org/10.1109/TII.2022.3197419) proposed in the paper: [Physics-informed attention temporal convolutional network for EEG-based motor imagery classification](https://doi.org/10.1109/TII.2022.3197419)

Authors: Hamdi Altaheri, Ghulam Muhammad, Mansour Alsulaiman

Center of Smart Robotics Research, King Saud University, Saudi Arabia
##
**Updates**: 
* The regularization parameters of [ATCNet](https://doi.org/10.1109/TII.2022.3197419) have been modified, resulting in an enhancement in the model's performance and fortifying it against overfitting.
* The current [*main_TrainTest.py*](https://github.com/Altaheri/EEG-ATCNet/blob/main/main_TrainTest.py) file, following the training and evaluation method outlined in [Paper 1](https://doi.org/10.1109/TII.2022.3197419) and [paper 2](https://ieeexplore.ieee.org/document/10142002), has been identified as not aligning with industry best practices. In response, we strongly recommend adopting the methodology implemented in the refined [*main_TrainValTest.py*](https://github.com/Altaheri/EEG-ATCNet/blob/main/main_TrainValTest.py) file. This updated version splits the data into train/valid/test sets, following the guidelines detailed in this [post](https://braindecode.org/stable/auto_examples/model_building/plot_how_train_test_and_tune.html#) ([Option 2](https://braindecode.org/stable/auto_examples/model_building/plot_how_train_test_and_tune.html#option-2-train-val-test-split:~:text=Number%20of%20samples.%27%3E-,Option%202%3A%20Train%2DVal%2DTest%20Split,-When%20evaluating%20different)). 
##
In addition to the proposed [ATCNet](https://doi.org/10.1109/TII.2022.3197419) model, the [*models.py*](https://github.com/Altaheri/EEG-ATCNet/blob/main/models.py) file includes the implementation of other related methods, which can be compared with [ATCNet](https://doi.org/10.1109/TII.2022.3197419), including:
* **EEGNet**, [[paper](https://arxiv.org/abs/1611.08024), [original code](https://github.com/vlawhern/arl-eegmodels)]
* **EEG-TCNet**, [[paper](https://arxiv.org/abs/2006.00622), [original code](https://github.com/iis-eth-zurich/eeg-tcnet)]
* **TCNet_Fusion**, [[paper](https://doi.org/10.1016/j.bspc.2021.102826)]
* **MBEEG_SENet**, [[paper](https://doi.org/10.3390/diagnostics12040995)]
* **EEGNeX**, [[paper](https://arxiv.org/abs/2207.12369), [original code](https://github.com/chenxiachan/EEGNeX)]
* **DeepConvNet**, [[paper](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730), [original code](https://github.com/braindecode/braindecode)]
* **ShallowConvNet**, [[paper](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730), [original code](https://github.com/braindecode/braindecode)]

The following table shows the performance of [ATCNet](https://doi.org/10.1109/TII.2022.3197419) and other reproduced models based on the methodology defined in the [*main_TrainValTest.py*](https://github.com/Altaheri/EEG-ATCNet/blob/main/main_TrainValTest.py) file:

<table>
    <tr>
        <td rowspan="2">Model</td>
        <td rowspan="2">#params</td>
        <td colspan="2">BCI Competition IV-2a dataset (<a href="https://www.bbci.de/competition/iv/#dataset2a">BCI 4-2a</a>) </td>
        <td colspan="2">High Gamma Dataset (<a href="https://github.com/robintibor/high-gamma-dataset">HGD</a>)<sup>*</sup></td>
    </tr>
    <tr>
        <td>training time (m) <sup>1,2</sup></td>
        <td>accuracy (%)</td>
        <td>training time (m) <sup>1,2</sup></td>
        <td>accuracy (%)</td>
    </tr>
    <tr>
        <td>ATCNet</td>
        <td>113,732</td>
        <td>13.5</td>
        <td>81.10</td>
        <td>62.6</td>
        <td>92.05</td>
    </tr>
    <tr>
        <td>TCNet_Fusion</td>
        <td>17,248</td>
        <td>8.8</td>
        <td>69.83</td>
        <td>65.2</td>
        <td>89.73</td>
    </tr>
    <tr>
        <td>EEGTCNet</td>
        <td>4,096</td>
        <td>7.0</td>
        <td>65.36</td>
        <td>36.8</td>
        <td>87.80</td>
    </tr>
    <tr>
        <td>MBEEG_SENet</td>
        <td>10,170</td>
        <td>15.2</td>
        <td>69.21</td>
        <td>104.3</td>
        <td>90.13</td>
    </tr>
    <tr>
        <td>EEGNet</td>
        <td>2,548</td>
        <td>6.3</td>
        <td>68.67</td>
        <td>36.5</td>
        <td>88.25</td>
    </tr>
    <tr>
        <td>DeepConvNet</td>
        <td>553,654</td>
        <td>7.5</td>
        <td>42.78</td>
        <td>43.9</td>
        <td>87.53</td>
    </tr>
    <tr>
        <td>ShallowConvNet</td>
        <td>47,364</td>
        <td>8.2</td>
        <td>67.48</td>
        <td>61.8</td>
        <td>87.00</td>
    </tr>
</table>
<sup>1 using Nvidia GTX 1080 Ti 12GB </sup><br>
<sup>2 (500 epochs, without early stopping)</sup><br>
<sup>* please note that <a href="https://github.com/robintibor/high-gamma-dataset">HGD</a> is for "executed movements" NOT "motor imagery"</sup>

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
2. [Leave One Subject Out (LOSO)](https://link.springer.com/article/10.1007/s00521-021-06352-5#Sec9:~:text=Full%20size%20table-,Performance%20evaluation,-For%20the%20MI) approach. LOSO is used for  **Subject-independent** evaluation. In LOSO, the model is trained and evaluated by several folds, equal to the number of subjects, and for each fold, one subject is used for evaluation and the others for training. The LOSO evaluation technique ensures that separate subjects (not visible in the training data) are used to evaluate the model.

The *get_data()* method in the [*preprocess.py*](https://github.com/Altaheri/EEG-ATCNet/blob/main/preprocess.py) file is used to load the dataset and split it into training and testing. This method uses the [subject-specific](https://link.springer.com/article/10.1007/s00521-021-06352-5#Sec9:~:text=Full%20size%20table-,Performance%20evaluation,-For%20the%20MI) approach by default. If you want to use the [subject-independent (LOSO)](https://link.springer.com/article/10.1007/s00521-021-06352-5#Sec9:~:text=Full%20size%20table-,Performance%20evaluation,-For%20the%20MI) approach, set the parameter *LOSO = True*.


## About ATCNet
ATCNet is inspired in part by the Vision Transformer ([ViT](https://arxiv.org/abs/2010.11929)). [ATCNet](https://doi.org/10.1109/TII.2022.3197419) differs from [ViT](https://arxiv.org/abs/2010.11929) by the following:
* [ViT](https://arxiv.org/abs/2010.11929) uses single-layer linear projection while [ATCNet](https://doi.org/10.1109/TII.2022.3197419) uses multilayer nonlinear projection, i.e., convolutional projection specifically designed for EEG-based brain signals.
* [ViT](https://arxiv.org/abs/2010.11929) consists of a stack of encoders where the output of the previous encoder is the input of the subsequent. [ATCNet](https://doi.org/10.1109/TII.2022.3197419) consists of parallel encoders and the outputs of all encoders are concatenated.
* The encoder block in [ViT](https://arxiv.org/abs/2010.11929) consists of a multi-head self-attention (MHA) followed by a multilayer perceptron (MLP), while in [ATCNet](https://doi.org/10.1109/TII.2022.3197419) the MHA is followed by a temporal convolutional network (TCN).
* The first encoder in [ViT](https://arxiv.org/abs/2010.11929) receives the entire input sequence, while each encoder in [ATCNet](https://doi.org/10.1109/TII.2022.3197419) receives a shifted window from the input sequence.

![ATCNet vs Vit](https://github.com/Altaheri/EEG-ATCNet/assets/25565236/210f6a4e-c212-4a9e-9336-415f0df4e293)

[ATCNet](https://doi.org/10.1109/TII.2022.3197419) model consists of three main blocks: 
1. **Convolutional (CV) block**: encodes low-level spatio-temporal information within the MI-EEG signal into a sequence of high-level temporal representations through three convolutional layers. 
2. **Attention (AT) block**: highlights the most important information in the temporal sequence using a multi-head self-attention (MHA). 
3. **Temporal convolutional (TC) block**: extracts high-level temporal features from the highlighted information using a temporal convolutional layer
* [ATCNet](https://doi.org/10.1109/TII.2022.3197419) model also utilizes the convolutional-based sliding window to augment MI data and boost the performance of MI classification efficiently. 

<p align="center">
Visualize the transition of data in the ATCNet model.
</p>
<p align="center">
<img src="https://user-images.githubusercontent.com/25565236/185449791-e8539453-d4fa-41e1-865a-2cf7e91f60ef.png" alt="The components of the proposed ATCNet model" width="700"/>
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
The [BCI Competition IV-2a](https://www.bbci.de/competition/iv/#dataset2a) dataset needs to be downloaded, and the data path should be set in the 'data_path' variable in the [*main_TrainValTest.py*](https://github.com/Altaheri/EEG-ATCNet/blob/main/main_TrainValTest.py) file. The dataset can be downloaded from [here](http://bnci-horizon-2020.eu/database/data-sets).


## References
If you find this work useful in your research, please use the following BibTeX entry for citation

```
@article{9852687,
  title={Physics-Informed Attention Temporal Convolutional Network for EEG-Based Motor Imagery Classification},
  author={Altaheri, Hamdi and Muhammad, Ghulam and Alsulaiman, Mansour},
  journal={IEEE Transactions on Industrial Informatics},
  year={2023},
  volume={19},
  number={2},
  pages={2249--2258},
  publisher={IEEE}
  doi={10.1109/TII.2022.3197419}
}

@article{10142002,
  title={Dynamic convolution with multilevel attention for EEG-based motor imagery decoding}, 
  author={Altaheri, Hamdi and Muhammad, Ghulam and Alsulaiman, Mansour},
  journal={IEEE Internet of Things Journal}, 
  year={2023},
  volume={10},
  number={21},
  pages={18579-18588},
  publisher={IEEE}
  doi={10.1109/JIOT.2023.3281911}
}

@article{altaheri2023deep,
  title={Deep learning techniques for classification of electroencephalogram (EEG) motor imagery (MI) signals: A review},
  author={Altaheri, Hamdi and Muhammad, Ghulam and Alsulaiman, Mansour and Amin, Syed Umar and Altuwaijri, Ghadir Ali and Abdul, Wadood and Bencherif, Mohamed A and Faisal, Mohammed},
  journal={Neural Computing and Applications},
  year={2023},
  volume={35},
  number={20},
  pages={14681--14722},
  publisher={Springer}
  doi={10.1007/s00521-021-06352-5}
}

```
