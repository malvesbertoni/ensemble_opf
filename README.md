# Optimum-Path Forest Stacking-based Ensemble for Intrusion Detection

*This repository holds all the necessary code to run the very-same experiments described in the paper "Optimum-Path Forest Stacking-based Ensemble for Intrusion Detection".*

## References

If you use our work to fulfill any of your needs, please cite us:

```BibTex
@article{Bertoni:21,
  author = {Mateus A. Bertoni and Gustavo H. de Rosa and Jose R. F. Brega},
  title = {Optimum-path forest stacking-based ensemble for intrusion detection},
  journal = {Evolutionary Intelligence},
  doi = {10.1007/s12065-021-00609-7},
  year = {2021},
  month = may,
  publisher = {Springer Science and Business Media {LLC}}
}
```

## Structure

  * `data/`
    * `nsl-kdd.tar.gz`: A compressed file containing the NSL-KDD dataset;
    * `unespy.tar.gz`: A compressed file containing the uneSPY dataset;
  * `models/`
    * `stacking.py`: A class that implements a stacking-based ensemble;
  * `utils/`
    * `classifiers.py`: Wraps the classifiers classes;
    * `load.py`: Loads the dataset according to desired format;
    * `wrapper.py`: Wraps the classification tasks.

## How-to-Use

There are 4 simple steps in order to accomplish the same experiments described in the paper:

 * Install the requirements;
 * Perform the single classifier classification (using k-folds or split);
 * Perform the stacking-based ensemble classification;
 * Process post-classification information for further comparison;
 
### Installation

Please install all the pre-needed requirements using:

```pip install -r requirements.txt```

### Classification with `k-folds` or `split`

Our first classification script is to use the k-folds cross validation to mitigate the randomness of the dataset. With that in mind, just run the following script with the input arguments:

```python single_classifier_kfolds.py -h```

Our seond classification script is to use the traditional training/testing set split validation to mitigate the randomness of the dataset. With that in mind, just run the following script with the input arguments:

```python single_classifier_split.py -h```

*Note that it will output a helper file in order to assist in choosing the correct arguments for the script.*

### Classification with stacking-based ensemble 

Additionally, we offer a stacking-based ensmeble to construct more robust classifiers. With that in mind, just run the following script with the input arguments:

```python stacking_classifier.py -h```

*Note that it will output a helper file in order to assist in choosing the correct arguments for the script.*

### Post-classification processing

Finally, after concluding the classification step, it is now possible to load back the metrics found during the classification procedure and output them to more-readable files. Run the following script in order to fulfill that purpose:

```python process_single_metrics_file.py -h```

Also, if one have performed an ensemble-based classification, please use the following script:

```python process_ensemble_metrics_file.py -h```

*Note that the classification process will always output a `.pkl` file, while the other scripts will output a `*.txt` file.*
