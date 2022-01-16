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
2- `python train_logistic_regression_and_save.py`
3- `test_logistic_regression.py`
