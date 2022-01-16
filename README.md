# Different Language Detector for OSCAR corpus datasets.

```
conda create -n pyspark python=3.8

conda install pyspark
conda install -c conda-forge langdetect
conda install -c conda-forge pyarrow
conda install -c conda-forge scikit-learn
conda install -c huggingface -c conda-forge datasets
conda install -c conda-forge ipykernel
```

1- `python generate_and_write_tfidf_dataset_vector.py`
2- `python train_logistic_regression_and_save.py`
3- `test_logistic_regression.py`
