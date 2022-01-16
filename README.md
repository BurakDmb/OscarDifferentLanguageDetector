# Different Language Detector for OSCAR corpus datasets.

## Installation:

* Create a virtual environment:
`conda create -n pyspark python=3.8`

* Install pyspark and other modules:

```
PYSPARK_HADOOP_VERSION=3.2 pip install pyspark[sql] -v
pip install langdetect pyarrow scikit-learn datasets ipykernel
```

* Then, download hadoop from https://hadoop.apache.org/releases.html , From this page, select the version 3.2.*. After that, unzip the tar.gz to ~ directory and add these lines at the end of your your .bashrc file
```
export HADOOP_HOME=~/hadoop-3.2.2
export LD_LIBRARY_PATH="$HADOOP_HOME/lib/native/:$LD_LIBRARY_PATH"

```

## Source codes and explanation:

1- `python generate_and_write_tfidf_dataset_vector.py`
This script generates tfidf vectors of the dataset by using spark mllib tfidf feature extractors.

2- `python train_logistic_regression_and_save.py`
This script trains a logistic regression model and trained model is saved into the directory model_logistic_regression

3- `test_logistic_regression.py`
This script reads trained model and vectorized dataset from files. It splits training and test by using the same seed used in the training and it evaluates the test dataset.

4- `python train_random_forest_and_save.py`
This script trains a logistic regression model and trained model is saved into the directory model_logistic_regression

5- `test_random_forest.py`
This script reads trained model and vectorized dataset from files. It splits training and test by using the same seed used in the training and it evaluates the test dataset.

6- `test_lr_and_rf.py`
This script reads lr and rf trained models and vectorized dataset from files. It splits training and test by using the same seed used in the training and it evaluates the test dataset. It will combine the output of the lr and rf to generate a ensemble predictor where two predictions are OR gated.
