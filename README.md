# posenet-python
 Lightweight pose estimation using Tensorflow v2.0 and up

## Installation

The code has been tested to work on a python 3.7 distribution, the minimal installation of [miniconda](https://docs.conda.io/en/latest/miniconda.html) should be sufficient.

Once installed open a command window and create a python 2.7 environent, for example called _posenet_.
```
conda create --name posenet python=3.7
```

If that has succesfully worked enter the posenet environment.
```
conda activate posenet
```

We now need to install some dependencies from conda and pip
``` 
conda install numpy
conda install tensorflow
pip install opencv-python==3.4.5.20
```
**Note:** installing opencv through conda _might_ work for the pose estimation, but Anaconda's default opencv does not include ffpmeg/VideoCapture support, which is needed for capturing poses using a webcam. 

To run the webcam test code, run the following

```
python webcam_test_py
```
