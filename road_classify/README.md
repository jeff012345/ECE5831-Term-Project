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
   
   Technically this can work with tensorflow CPU version but it's much slower.

### train.py
`python train.py`

Update paths in file

### train_kfolds.py
`python train_kfolds.py`

Broken

### predict.py
`python predict.py`

Used a pretrained model in the h5 format to read images and make a prediction
Update paths in file

### evaluate.py

Uses the pretrained model to validate the accuracy and create some graphs
Check the file paths in the file

### separate_with_mask.py

Used to remove all non-road pixels from the base image using the road mask from the DeepGlobe data set


