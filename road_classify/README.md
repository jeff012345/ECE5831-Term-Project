1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for Python 3.7 and 64-bit

2. Create a new virtual environment
   Open command prompt
   Change directory to Miniconda install directory
   cd to `condabin`   
   ```python
   conda -n tf_gpu python=3.5
   conda activate tf_gpu
   conda install tensorflow-gpu
   pip install -r requirements.txt
   ```

### train.py
`python train.py`

Simple CNN created from a tutorial.

### train_kfolds.py
`python train_kfolds.py`

Same CNN as train.py but subsets the training set using k-folds. Still leaves a few images out for final validation.

### predict.py
`python predict.py`

Used a pretrained model in the h5 format to read images and make a prediction