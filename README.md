# DeepCOLA

This code is a Python implementation of DeepCOLA (Deep COmpetitive Learning Algorithm). The DeepCOLA takes appliance-level electricity consumption data as input and produces the clusters.

# How to run the code

The DeepCOLA algorithm requires following libraries in order to run and produce clusters.

* Python 3
* Numpy
* SciPy
* Matplotlib
* Tensorflow
* Keras

The DeepCOLA algorithm utlizes TensorFlow as backend. The easiast way to install above libraries is by using Anaconda from the [Link](https://www.anaconda.com/). The Anaconda tool provides built-in support for Python 3, Numpy, SciPy and Matplotlib. However, Keras can be installed using the following Conda comannds. 

```
conda install -c conda-forge keras 
```

Another option to run the code without Anaconda is by installing all the libraries individually using Pip3 as the code requires Python 3.

```
Numpy
sudo pip3 install numpy

Scipy
sudo pip3 install scipy

Matplotlib
sudo pip3 install matplotlib
```

Please refer to these links [TensorFlow](https://www.tensorflow.org/install/) and [Keras](https://keras.io/) to install them.

# Dataset

The [UK-Dale](http://jack-kelly.com/data/) electricity consumption dataset is used for the experiments. This dataset contains consumption at appliance level. A dataset file is provided with the code which contains consumption for one day. The origianl UK-Dale dataset provides individual files for each appliance for various time duration. However, we have applied data preprocessing and combined the consumption from appliances into one file.
