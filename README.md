ds-utils
-----

A set of classes to ease the pre-processing of data to feed machine learning algorithms.

**python 2.7 and python 3.12 compatible**

Main tool:

```python
class Preprocessor(Model):
    """ Preprocessor
    This class allows for the complete pre-processing of a DataFrame into a sparse matrix for model input.
    """

    def __init__(self):
        super(Preprocessor, self).__init__(name="Preprocessor")
        self.prunificator = None
        self.counterizor = None
        self.vectorizor = None
        self.imputor = None
        self.sparsifior = None
        self.variance_selector = None

    def fit_transform(self, df, pruning_frequency=None, do_not_use=None, sharp_categorical_dict=None, na_strategy=MeanStrategy(), variance_threshold=None, low_memory=True):
        """ Pre-process input_files for the training phase. Once completed, you should save the resulting Preprocessor object for the predict phase.

        Args:
            df (pandas DataFrame): dataframe to be pre-processed.

            pruning_frequency (float or None): Frequency below which value in categorical features are pruned (set to *misc*). (deactivated by default)

            do_not_use (list or None): Leave these columns alone!

            sharp_categorical_dict (dict): {'column': {'sep': "#", 'norm': True/False} }.

                                           If not provided, program looks for columns ending in *_cat* and automatically
                                           creates an entry in the dict with value {'sep': "#", 'norm': True}.

            na_strategy (Strategy): Strategy used to impute missing values.

            variance_threshold (float or None): Threshold for variance selector. (deactivated by default)

            low_memory (bool): If True, counterizor will not use parallel computation. default: False

        Returns:
            namedtuple('data', ['X', 'other', 'names'])

                data.X (scipy sparse matrix): model input

                data.other (pandas DataFrame): columns unused
        """
```

First release in 2016. 

Documentation compiled using *sphynx*. 
