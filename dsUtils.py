#!/usr/bin/env python
# -*- coding: utf-8 -*-

# numpy==1.16.0
# pandas==0.23.4
# scikit-learn==0.20.0
# scipy==0.16.1

import re
import logging
import unittest
import numpy as np
import pandas as pd
import multiprocessing as mp
from abc import ABCMeta
from scipy import sparse
from itertools import repeat, compress
from collections import Counter, OrderedDict, namedtuple
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import VarianceThreshold

logger = logging.getLogger(__name__)


class Model(object):

    def __init__(self, name):
        self.name = name

    def fit(self, *args):
        pass

    def transform(self, *args):
        pass

    def fit_transform(self, *args):
        pass

    def predict(self, *args):
        pass

    def predict_proba(self, *args):
        pass

    def fit_predict(self, *args):
        pass


class Strategy(object):
    """ Skeleton for missing values imputation strategy.

    Note:
       Metaclass! You should define your own derived class.
   """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    def na_dictionary(self, *args):
        pass


class CustomStrategy(Strategy):
    def __init__(self, strategy, default=None):
        super(CustomStrategy, self).__init__()
        self.strategy = strategy
        self.default = default

    def na_dictionary(self, df, imputed_cols):
        if len(imputed_cols) == 0:
            logger.warning("No columns provided!")
        not_imputed = list(set(df.columns.values).difference(imputed_cols))
        if len(not_imputed) > 0:
            logger.warning("Columns {columns} will not be imputed.".format(columns=not_imputed))
        if self.default == 0:
            na_dict = ZeroStrategy().na_dictionary(df, imputed_cols)
        else:
            na_dict = MeanStrategy().na_dictionary(df, imputed_cols)
        for col in self.strategy:
            if col in imputed_cols:
                if isinstance(self.strategy[col], (int, float, bool)):
                    na_dict[col] = self.strategy[col]
                elif self.strategy[col] == 'median':
                    value = df[col].median()
                    if pd.isnull(value):
                        logger.warning("Entire column {col} is NA => default imputation!".format(col=col))
                        # na_dict.pop(col)
                    else:
                        na_dict[col] = value
                elif self.strategy[col] == 'mean' and self.default != "mean":
                    value = df[col].mean()
                    if pd.isnull(value):
                        logger.warning("Entire column {col} is NA => default imputation!".format(col=col))
                        # na_dict.pop(col)
                    else:
                        na_dict[col] = value
                else:
                    logger.warning("Strategy unknown for column {col} => default imputation!".format(col=col))
                    # na_dict.pop(col)
            else:
                logger.warning("Column {col} not in imputed_cols!".format(col=col))
        return na_dict


class NoneStrategy(Strategy):
    def __init__(self):
        super(NoneStrategy, self).__init__()

    def na_dictionary(self, df, imputed_cols):
        return {}


class ZeroStrategy(Strategy):
    def __init__(self):
        super(ZeroStrategy, self).__init__()

    def na_dictionary(self, df, imputed_cols):
        na_dict = {col: 0 for col in imputed_cols}
        return na_dict


class MeanStrategy(Strategy):
    def __init__(self):
        super(MeanStrategy, self).__init__()

    def na_dictionary(self, df, imputed_cols):
        na_dict = {col: df[col].mean() if not pd.isnull(df[col].mean()) else 0 for col in imputed_cols}
        return na_dict


class MedianStrategy(Strategy):
    def __init__(self):
        super(MedianStrategy, self).__init__()

    def na_dictionary(self, df, imputed_cols):
        na_dict = {col: df[col].median() if not pd.isnull(df[col].median()) else 0 for col in imputed_cols}
        return na_dict


CATEGORICAL_FEATURE = "cat_{name}"
REGEXP_CF = "^cat_.*$"
SHARP_CATEGORICAL_FEATURE = "#_cat_{name}"
REGEXP_SCF = "^#_cat_.*$"


def my_counter(x):
    return Counter(filter(None, x))


def my_normalized_counter(x):
    counter_ = my_counter(x)
    total_count = sum(counter_.values())
    return {label: count * 1.0 / total_count for label, count in counter_.items()} if total_count > 0 else counter_


def my_instance(v):
    if isinstance(v, (Counter, dict)):
        return v
    elif isinstance(v, list):
        return my_counter(v)
    else :
        return{v: 1}


class Prunificator(Model):
    """ Prunificator
    This class allows to remove values in categorical features whose frequency is less than *frequency*.

    """

    def __init__(self, ignored_columns=None, sharp_categorical_dict=None, frequency=0.0001):
        super(Prunificator, self).__init__(name="Prunificator")
        self.ignored_columns = ignored_columns or []
        self.sharp_categorical_dict = sharp_categorical_dict or {}
        self.categorical_columns = None
        self.frequency = frequency
        self.keeped_values_dict = {}

    def fit(self, df):
        # GET (NAME: KIND) OF COLUMNS
        columns_kind = {col: df[col].dtype.kind for col in df.columns if col not in self.ignored_columns}
        # CATEGORICAL FEATURES
        tmp = [col for col, kind in columns_kind.items() if kind in 'if']
        self.categorical_columns = list(set(columns_kind).difference(tmp + self.sharp_categorical_dict.keys()))

        for column in self.categorical_columns:
            counts = df[column].value_counts(normalize=True)
            self.keeped_values_dict[column] = counts[counts > self.frequency].index.tolist()
        return self

    def transform(self, df):
        for column in self.categorical_columns:
            df[column].where(df[column][df[column].notnull()].isin(self.keeped_values_dict[column]), other="misc",
                             inplace=True)
        return df

    def fit_transform(self, df):
        return self.fit(df).transform(df)


class Aggregator(Model):
    """ Aggregator
    This class allows for the transformation of an unaggregated data frame into an aggregated one
    (*Vectorizor* compatible).

    Args:
        aggregation_keys (list): List of columns used for the aggregation (group by).
        aggregation_strategy (dict): Dictionary {'column': aggregation_function}.
                Built-in aggregation functions exist as *counter* and *normalized_counter*.
        vectorizor_compatibility (bool): Custom arg. with default value = False.
                Use True for exploratory data analysis (categorical features will not be mapped to dictionary).
    """
    aggregation_functions = {"counter": my_counter, "normalized_counter": my_normalized_counter}

    def __init__(self, aggregation_keys, aggregation_strategy, vectorizor_compatibility=True):
        super(Aggregator, self).__init__(name="Aggregator")
        self.aggregation_keys = aggregation_keys if isinstance(aggregation_keys, list) else [aggregation_keys]
        self.aggregation_strategy = aggregation_strategy
        self.vectorizor_compatibility = vectorizor_compatibility
        self.categorical_columns = None

    def fit(self, df):
        """ Fit the class attributes according to the provided data frame.

        Args:
            df (pandas DataFrame): The data frame to be aggregated.

        Returns:
            self (object)
        """
        # GET (NAME: KIND) OF COLUMNS
        columns_kind = {col: df[col].dtype.kind for col in df.columns if col not in self.aggregation_keys}
        # CATEGORICAL FEATURES
        self.categorical_columns = list(
            set(columns_kind).difference([col for col, kind in columns_kind.items() if kind in 'if']))
        # WARNING
        missing = list(set(columns_kind).difference(self.aggregation_strategy))
        if len(missing) > 0:
            logger.warning(
                'Aggregation functions not provided for all columns, columns {columns} will be dropped!'.format(
                    columns=missing))
            [self.categorical_columns.remove(m) for m in missing if m in self.categorical_columns]
        unknown = list(set(self.aggregation_strategy).difference(columns_kind))
        if len(unknown) > 0:
            logger.warning('Aggregation functions provided for non existing columns {columns}!'.format(columns=unknown))
            [self.aggregation_strategy.pop(u) for u in unknown]
        # AGGREGATION STRATEGY
        for col, strategy in self.aggregation_strategy.items():
            self.aggregation_strategy[col] = self.aggregation_functions[
                strategy] if strategy in self.aggregation_functions else strategy
        return self

    def transform(self, df):
        """ Transform **df** into **agg_df** which is *Vectorizor* compatible.

        Args:
            df (pandas DataFrame): The data frame to be aggregated.

        Returns:
            agg_df (pandas Dataframe): The resulting aggregated data frame.
        """
        # CATEGORICAL FEATURES
        if self.categorical_columns:
            df.fillna({col: 'other' for col in self.categorical_columns}, inplace=True)
            df.replace('', {col: 'other' for col in self.categorical_columns}, inplace=True)
        print(self.aggregation_strategy)
        agg_df = df.groupby(self.aggregation_keys).aggregate(self.aggregation_strategy).reset_index()
        if self.vectorizor_compatibility:
            for col in self.categorical_columns:
                agg_df[col] = agg_df[col].map(lambda v: my_instance(v))
        agg_df.rename(columns={col: CATEGORICAL_FEATURE.format(name=col) for col in self.categorical_columns},
                      inplace=True)
        return agg_df

    def fit_transform(self, df):
        return self.fit(df=df).transform(df=df)


class Counterizor(Model):
    """ Counterizor
    This class allows for the transformation of categorical features into Counters for *Vectorizor* compatibility.

    Args:
        ignored_colums (list): Leave these columns alone!
        sharp_categorical_dict (dict): Dictionary {'column': {'sep': "#", 'norm': True/False} }.
    """

    def __init__(self, ignored_columns=None, sharp_categorical_dict=None):
        super(Counterizor, self).__init__(name="Counterizor")
        self.ignored_columns = ignored_columns or []
        self.sharp_categorical_dict = sharp_categorical_dict or {}
        self.categorical_columns = None

    @staticmethod
    def _counter(x, norm):
        counter_ = Counter(filter(None, x))
        if norm:
            total_count = sum(counter_.values())
            counter_ = {label: count * 1.0 / total_count for label, count in
                        counter_.items()} if total_count > 0 else counter_
        return counter_

    def fit(self, df):
        """ Fit the class attributes according to the provided data frame.

        Args:
            df (pandas DataFrame): The data frame to be counterized.

        Returns:
            self (Counterizor)
        """
        # GET (NAME: KIND) OF COLUMNS
        columns_kind = {col: df[col].dtype.kind for col in df.columns if col not in self.ignored_columns}
        # CATEGORICAL FEATURES
        tmp = [col for col, kind in columns_kind.items() if kind in 'if']
        self.categorical_columns = list(set(columns_kind).difference(tmp + list(self.sharp_categorical_dict.keys())))
        return self

    def transform(self, df):
        """ Transform **df** in place.

        Args:
            df (pandas DataFrame): The data frame to be counterized.

        Returns:
            df (pandas Dataframe): The resulting counterized data frame.
        """
        # CATEGORICAL FEATURES
        if self.categorical_columns:
            df.fillna({column: '' for column in self.categorical_columns}, inplace=True)
            df[self.categorical_columns] = df[self.categorical_columns].applymap(
                lambda v: {v: 1} if v else {"other": 1})
            df.rename(columns={column: CATEGORICAL_FEATURE.format(name=column) for column in self.categorical_columns},
                      inplace=True)
        # SHARP CATEGORICAL FEATURES
        if self.sharp_categorical_dict:
            df.fillna({column: '' for column in self.sharp_categorical_dict}, inplace=True)
            for column, param in self.sharp_categorical_dict.items():
                df[column] = df[column].map(lambda v: Counterizor._counter(v.split(param['sep']), param['norm']))
            df.rename(columns={column: SHARP_CATEGORICAL_FEATURE.format(name=column) for column in
                               self.sharp_categorical_dict}, inplace=True)
        return df

    def fit_transform(self, df):
        return self.fit(df).transform(df)


def process_counterizor(args):
    counterizor_, partial_df = args
    return counterizor_.transform(partial_df)


def parallel_counterizor(df, counterizor, n_jobs=-1):
    n_jobs = mp.cpu_count() * 2 - 1 if n_jobs == -1 else n_jobs
    p = mp.Pool(n_jobs)
    pool_results = p.map(process_counterizor, zip(repeat(counterizor), np.array_split(df, min(len(df), n_jobs))))
    p.close()
    p.join()
    return pd.concat([r for r in pool_results])


class Vectorizor(Model):
    """
    VECTORIZOR
    input = counterized DataFrame
    output = namedtuple('data', ['df', 'sm', 'sm_columns'])
    """

    def __init__(self, ignored_columns=None):
        super(Vectorizor, self).__init__(name="Vectorizor")
        self.ignored_columns = ignored_columns or []
        self.sharp_categorical_columns = None
        self.sharp_categorical_vectorizers = {}
        self.categorical_columns = None
        self.categorical_vectorizers = {}
        self.too_many_categories = None
        self.numeric_columns = None

    def fit(self, df):
        # GET (NAME: KIND) OF COLUMNS
        columns_kind = {col: df[col].dtype.kind for col in df.columns if col not in self.ignored_columns}
        # NUMERICAL FEATURES
        self.numeric_columns = [col for col, kind in columns_kind.items() if kind in 'if']
        # CATEGORICAL FEATURES
        self.sharp_categorical_columns = [col for col in columns_kind if re.match(REGEXP_SCF, col)]
        self.categorical_columns = [col for col in columns_kind if re.match(REGEXP_CF, col)]
        if self.sharp_categorical_columns:
            for column in self.sharp_categorical_columns:
                self.sharp_categorical_vectorizers[column] = DictVectorizer(sparse=True).fit(df[column])
        if self.categorical_columns:
            for column in self.categorical_columns:
                self.categorical_vectorizers[column] = DictVectorizer(sparse=True).fit(df[column])
        self.too_many_categories = [len(self.categorical_vectorizers[col].vocabulary_) > 7 for col in
                                    self.categorical_columns]
        missing = list(set(columns_kind).difference(
            self.numeric_columns + self.sharp_categorical_columns + self.categorical_columns + self.ignored_columns))
        if len(missing) > 0:
            logger.warning("Found non-counterized categorical columns {columns} \
                \n option 1: add them to ignored_columns_list \n option 2: consider Counterizor first".format(
                columns=missing))
            return
        return self

    def transform(self, df):
        sharp_sparse = OrderedDict()
        sm_columns = []
        sm = None
        if self.sharp_categorical_columns:
            for column in self.sharp_categorical_columns:
                sharp_sparse[column] = self.sharp_categorical_vectorizers[column].transform(df[column])
                names_tmp = sorted(self.sharp_categorical_vectorizers[column].vocabulary_,
                                   key=self.sharp_categorical_vectorizers[column].vocabulary_.get)
                sm_columns += ['{column}_{name}'.format(column=column, name=name) for name in names_tmp]
                df.drop(column, axis=1, inplace=True)
        nb_of_samples = len(df)
        concat_ = []
        if self.categorical_columns:
            # ################### #
            # BECAUSE OF AUGUSTIN #
            # ################### #
            for i, column in enumerate(self.categorical_columns):
                if self.too_many_categories[i]:
                    sharp_sparse[column] = self.categorical_vectorizers[column].transform(df[column])
                    names_tmp = sorted(self.categorical_vectorizers[column].vocabulary_,
                                       key=self.categorical_vectorizers[column].vocabulary_.get)
                    sm_columns += ['{column}_{name}'.format(column=column, name=name) for name in names_tmp]
                else:
                    names_tmp = sorted(self.categorical_vectorizers[column].vocabulary_,
                                       key=self.categorical_vectorizers[column].vocabulary_.get)
                    names_categorical_columns = ['{column}_{name}'.format(column=column, name=name) for name in
                                                 names_tmp]
                    df_cat = pd.DataFrame(self.categorical_vectorizers[column].transform(df[column]).todense(),
                                          columns=names_categorical_columns)
                    suffixes = [m.group(1) for m in
                                [re.match('^{p}_(.*)'.format(p=column), col) for col in names_categorical_columns] if m]
                    if "other" in suffixes:
                        suffixes.remove('other')
                        df_cat.loc[
                            df_cat["{p}_other".format(p=column)] == 1, ["{p}_{s}".format(p=column, s=suffix) for suffix
                                                                        in suffixes]] = np.nan
                        # df_cat.ix[
                        #     df_cat["{p}_other".format(p=column)] == 1, ["{p}_{s}".format(p=column, s=suffix) for suffix
                        #                                                 in suffixes]] = np.nan
                        df_cat.drop("{p}_other".format(p=column), axis=1, inplace=True)
                    concat_.append(df_cat)
                df.drop(column, axis=1, inplace=True)
                # ################### #
                # BECAUSE OF AUGUSTIN #
                # ################### #
        df.reset_index(drop=True, inplace=True)
        concat_.insert(0, df)
        blocks = [sharp_sparse[key] for key in sharp_sparse if sharp_sparse[key].shape[1] > 0]
        if len(blocks) > 0:
            sm = sparse.hstack(blocks=blocks, format='csr')
            assert len(sm_columns) == sm.shape[1] and sm.shape[0] == nb_of_samples
        concatenated_df = pd.concat(concat_, axis=1)
        assert len(concatenated_df) == nb_of_samples
        data = namedtuple('data', ['df', 'sm', 'sm_columns'])
        return data(df=concatenated_df, sm=sm, sm_columns=sm_columns)

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)


class Imputor(Model):
    """
    IMPUTOR
    input = vectorized DataFrame
    output = same DataFrame with missing values imputed
    """

    def __init__(self, ignored_columns=None, na_strategy=None):
        super(Imputor, self).__init__(name="Imputor")
        self.ignored_columns = ignored_columns or []
        self.na_strategy = na_strategy
        self.na_dict = {}

    def fit(self, df):
        imputed_columns = [col for col in df.columns if col not in self.ignored_columns]
        if imputed_columns and self.na_strategy:
            self.na_dict = self.na_strategy.na_dictionary(df, imputed_columns)
        return self

    def transform(self, df):
        df.fillna(self.na_dict, inplace=True)
        return df

    def fit_transform(self, df):
        return self.fit(df).transform(df)


class Sparsifior(Model):
    """
    SPARSIFIOR
    input = DataFrame + sparse matrix
    output = namedtuple('data', ['keys', 'features', 'X', 'y'])
    """

    def __init__(self, keys=None, target=None):
        super(Sparsifior, self).__init__(name="Sparsifior")
        self.keys = keys if isinstance(keys, list) else ([keys] if keys else [])
        self.target = target

    def fit(self):
        pass

    def transform(self, df=None, sm=None, sm_columns=None):
        keys = None
        features = []
        X = None
        y = None
        X_sparse_dict = OrderedDict()

        if df is not None and df.shape[1] > 0:
            non_numeric = [col for col in df.columns if
                           df[col].dtype.kind not in 'if' and col not in self.keys and col != self.target]
            if len(non_numeric) > 0:
                logger.error("Non-numeric feature found! Did you forget to specify keys and/or target?")
                return
            if not set(self.keys).issubset(set(df.columns)):
                logger.warning("Keys {keys} do not exist in the DataFrame you provided!".format(keys=self.keys))
            else:
                keys = df[self.keys]
                df.drop(keys, axis=1, inplace=True)
            if self.target and self.target not in df.columns:
                logger.warning(
                    "Target {target} does not exist in the DataFrame you provided!".format(target=self.target))
            elif not self.target:
                pass
            else:
                y = df[self.target]
                df.drop(self.target, axis=1, inplace=True)
            X_sparse_dict['df'] = sparse.csr_matrix(df.values)
            features += df.columns.tolist()

        if sm is not None and sm.shape[1] > 0:
            X_sparse_dict['sm'] = sm
            features = features + sm_columns if sm_columns else logger.warning(
                "You did not provide the sparse matrix features names.")

        if len(X_sparse_dict) > 0:
            X = sparse.hstack(blocks=[X_sparse_dict[key] for key in X_sparse_dict], format='csr')

        data = namedtuple('data', ['keys', 'features', 'X', 'y'])
        return data(keys=keys, features=features, X=X, y=y)


class VarianceSelector(Model):
    """
    Feature selector that removes all low variance features.
    """

    def __init__(self, threshold):
        super(VarianceSelector, self).__init__(name="VarianceSelector")
        self.threshold = threshold
        self.variance_selector = VarianceThreshold(threshold)
        if type(threshold) != int and type(threshold) != float:
            logger.warning('Input type cannot be {x}. '
                           ' It can only be int or float.'.format(x=type(threshold)))

    def fit(self, X):
        """Learn empirical variances from X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            Sample vectors from which to compute variances.
        Returns
        -------
        self : object
        """
        self.variance_selector.fit(X)
        return self

    def transform(self, X):
        """Reduce X to the selected features.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.
        Returns
        -------
        X_r : array of shape [n_samples, n_selected_features]
            The input samples with only the selected features.
        """
        return self.variance_selector.transform(X)

    def fit_transform(self, X):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            Training set.
        Returns
        -------
        X_new : array of shape [n_samples, n_features_new]
                Transformed array.
        """
        return self.fit(X).transform(X)

    def select_features(self, features):
        """Select features with enough variance.

        Parameters
        ----------
        features : array of shape [n_features]
                   Names of the features.
        Returns
        -------
        selected_features: array of shape [n_selected_features].
                           An array with only the names of the selected_features.
        """
        return list(compress(features, self.variance_selector.get_support()))


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

    def fit(self):
        pass

    def fit_transform(self, df, pruning_frequency=None, do_not_use=None, sharp_categorical_dict=None,
                      na_strategy=MeanStrategy(), variance_threshold=None, low_memory=True):
        """ Pre-process input_files for the training phase. Once completed, you should save the resulting Preprocessor
        object for the predict phase.

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
        # INFER SHARP CATEGORICAL FEATURES IF NOT PROVIDED
        sharp_categorical_dict = sharp_categorical_dict or {m.group(1): {"sep": "#", "norm": True} for m in
                                                            [re.match("(.*_cat)$", col) for col in df.columns] if m}

        # PRUNIFICATOR: FIT & TRANSFORM (IF ACTIVATED)
        if pruning_frequency:
            self.prunificator = Prunificator(ignored_columns=do_not_use,
                                                           sharp_categorical_dict=sharp_categorical_dict,
                                                           frequency=pruning_frequency)
            df = self.prunificator.fit_transform(df)

        # COUNTERIZOR: FIT & TRANSFORM
        self.counterizor = Counterizor(ignored_columns=do_not_use,
                                                     sharp_categorical_dict=sharp_categorical_dict).fit(df)
        df = parallel_counterizor(df, self.counterizor,
                                                n_jobs=-1) if not low_memory else self.counterizor.transform(df)

        # VECTORIZOR: FIT & TRANSFORM
        self.vectorizor = Vectorizor(ignored_columns=do_not_use)
        vectorized_data = self.vectorizor.fit_transform(df)

        # IMPUTOR: FIT & TRANSFORM
        self.imputor = Imputor(ignored_columns=do_not_use, na_strategy=na_strategy)
        imputed_df = self.imputor.fit_transform(vectorized_data.df)

        # PRUNIFICATOR: TRANSFORM
        self.sparsifior = Sparsifior(keys=do_not_use)
        sparsified_data = self.sparsifior.transform(imputed_df, vectorized_data.sm, vectorized_data.sm_columns)

        # VARIANCE SELECTOR: FIT & TRANSFORM (IF ACTIVATED)
        if variance_threshold:
            self.variance_selector = VarianceSelector(threshold=variance_threshold)
            X = self.variance_selector.fit_transform(sparsified_data.X)
            features = self.variance_selector.select_features(sparsified_data.features)
        else:
            X = sparsified_data.X
            features = sparsified_data.features

        # OUTPUT
        data = namedtuple('data', ['X', 'other', 'names'])
        return data(X=X, other=sparsified_data.keys, names=features)

    def transform(self, df, low_memory=True):
        """ Pre-process input_files using the same pre-processing parameters used for the training phase.

        Args:
            df (pandas DataFrame): dataframe to be pre-processed.

            low_memory (bool): If True, counterizor will not use parallel computation. default: False

        Returns:
            namedtuple('data', ['X', 'other', 'names'])

                data.X (scipy sparse matrix): model input

                data.other (pandas DataFrame): columns unused
        """
        # PRUNIFICATOR: TRANSFORM
        self.prunificator.transform(df) if self.prunificator else df

        # COUNTERIZOR: TRANSFORM
        df = parallel_counterizor(df, self.counterizor,
                                  n_jobs=-1) if not low_memory else self.counterizor.transform(df)

        # VECTORIZOR: TRANSFORM
        vectorized_data = self.vectorizor.transform(df)

        # IMPUTOR: TRANSFORM
        imputed_df = self.imputor.transform(vectorized_data.df)

        # SPARSIFIOR: TRANSFORM
        sparsified_data = self.sparsifior.transform(imputed_df, vectorized_data.sm, vectorized_data.sm_columns)

        # VARIANCE SELECTOR: TRANSFORM
        if self.variance_selector:
            X = self.variance_selector.transform(sparsified_data.X)
            features = self.variance_selector.select_features(sparsified_data.features)
        else:
            X = sparsified_data.X
            features = sparsified_data.features

        # OUTPUT
        data = namedtuple('data', ['X', 'other', 'names'])
        return data(X=X, other=sparsified_data.keys, names=features)


class CustomStrategyTest(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame([[34.5, None], [21.4, 7], [None, 11], [12.8, 3]], columns=["num1", "num2"])
        self.num_cols = ["num1", "num2"]

    def test_na_dictionary_empty_strategy_default(self):
        cs = CustomStrategy({})
        expected = {"num1": 22.9, "num2": 7}
        actual = cs.na_dictionary(self.df, self.num_cols)
        self.assertEqual(set(expected), set(actual))
        [self.assertAlmostEqual(expected[key], actual[key]) for key in expected]

    def test_na_dictionary_empty_strategy_default_0(self):
        cs = CustomStrategy({}, default=0)
        self.assertDictEqual(cs.na_dictionary(self.df, self.num_cols), {"num1": 0, "num2": 0})

    def test_na_dictionary_empty_numeric_cols(self):
        cs = CustomStrategy({"num1": 0}, default=0)
        self.assertDictEqual(cs.na_dictionary(self.df, []), {})

    def test_na_dictionary_int(self):
        cs = CustomStrategy({"num1": 1}, default=0)
        self.assertDictEqual(cs.na_dictionary(self.df, self.num_cols), {"num1": 1, "num2": 0})

    def test_na_dictionary_float(self):
        cs = CustomStrategy({"num1": 3.14}, default=0)
        self.assertDictEqual(cs.na_dictionary(self.df, self.num_cols), {"num1": 3.14, "num2": 0})

    def test_na_dictionary_bool(self):
        cs = CustomStrategy({"num1": False}, default=0)
        self.assertDictEqual(cs.na_dictionary(self.df, self.num_cols), {"num1": False, "num2": 0})

    def test_na_dictionary_unknown(self):
        cs = CustomStrategy({"num1": "abc"}, default=0)
        self.assertDictEqual(cs.na_dictionary(self.df, self.num_cols), {"num1": 0, "num2": 0})

    def test_na_dictionary_median(self):
        cs = CustomStrategy({"num1": "median"}, default=0)
        self.assertDictEqual(cs.na_dictionary(self.df, self.num_cols), {"num1": 21.4, "num2": 0})

    def test_na_dictionary_mean(self):
        cs = CustomStrategy({"num1": "mean"}, default=0)
        expected = {"num1": 22.9, "num2": 0}
        actual = cs.na_dictionary(self.df, self.num_cols)
        self.assertEqual(set(expected), set(actual))
        [self.assertAlmostEqual(expected[key], actual[key]) for key in expected]


class NoneStrategyTest(unittest.TestCase):

    def test_na_dictionary(self):
        ns = NoneStrategy()
        self.assertEqual(ns.na_dictionary(None, None), {})


class ZeroStrategyTest(unittest.TestCase):

    def setUp(self):
        self.num_cols = ["num1", "num2"]

    def test_na_dictionary(self):
        zs = ZeroStrategy()
        self.assertEqual(zs.na_dictionary(None, self.num_cols), {"num1": 0, "num2": 0})

    def test_na_dictionary_empty_numeric_cols(self):
        zs = ZeroStrategy()
        self.assertDictEqual(zs.na_dictionary(None, []), {})


class CounterizorTest(unittest.TestCase):

    def setUp(self):
        self.c = Counterizor(ignored_columns=['id', 'useless', 'target'],
                             sharp_categorical_dict={"concat_categorical": {"sep": "#", "norm": True}})
        column_names = ['id', 'useless', 'numeric1', 'numeric2', 'concat_categorical', 'gender', 'target']
        self.df = pd.DataFrame([['A34G56G', '001', 34.5, None, "a#b#a#b#c", "F", 0],
                                ['V27H74P', '002', 21.4, 7, None, "M", 1],
                                ['P34N12C', '002', None, 11, None, "", 1],
                                ['N08W87W', '002', 12.8, 3, "b#b#d#d#e#e", None, 0]], columns=column_names)

    def test_fit(self):
        self.c.fit(self.df)
        self.assertEqual(self.c.categorical_columns, ['gender'])

    def test_transform(self):
        df_c = self.c.fit_transform(self.df)
        self.assertListEqual(df_c.cat_gender.values.tolist(), [{'F': 1}, {'M': 1}, {'other': 1}, {'other': 1}])
        self.assertDictEqual(df_c["#_cat_concat_categorical"][3], {"b": 1. / 3, "d": 1. / 3, "e": 1. / 3})


class VarianceSelectorTest(unittest.TestCase):

    def setUp(self):
        self.X = pd.DataFrame([[34.5, 4, 0, 0.01], [21.4, 7, 0, 0.2], [5, 11, 0, 0.04], [12.8, 3, 0, 0.7]], columns=["a", "b", "c", "d"])
        self.features = ["a", "b", "c", "d"]

    def test_VarianceSelector_0(self):
        vs = VarianceSelector(0)
        expected_res = np.array([[3.45000000e+01,   4.00000000e+00,   1.00000000e-02],
                                 [2.14000000e+01,   7.00000000e+00,   2.00000000e-01],
                                 [5.00000000e+00,   1.10000000e+01,   4.00000000e-02],
                                 [1.28000000e+01,   3.00000000e+00,   7.00000000e-01]])
        self.assertTrue((vs.fit_transform(self.X) == expected_res).all())
        self.assertEqual(vs.select_features(["a", "b", "c", "d"]), ['a', 'b', 'd'])

    def test_VarianceSelector_1(self):
        vs = VarianceSelector(1)
        expected_res = np.array([[34.5, 4.],
                                 [21.4, 7.],
                                 [5., 11.],
                                 [12.8, 3.]])
        self.assertTrue((vs.fit_transform(self.X) == expected_res).all())
        self.assertEqual(vs.select_features(["a", "b", "c", "d"]), ['a', 'b'])


if __name__ == '__main__':

    column_names = ['id', 'sc1', 'numeric1', 'numeric2', 'sc2', 'gender', 'target', 'c1', 'c2']
    df = pd.DataFrame([['A34G56G', 'ab_cd', 34.5, None, "a#b#a#b#c", "F", 0, "AA","AAA"],
                       ['P34N12C', 'aé_ef', None, 98, "", "", 1, "BB", "AAA"],
                       ['V27H74P', '', -18.4, -10.45, "#b#a#", "M", 1, "CC", "BBB"],
                       ['B74Q12W', 'cd_ef', 0.75, 1.021, "#b#a#", "M", 1, "", "CCC"],
                       ['G98S45X', 'gh_ij', 78.45, 9874, None, "", 1, "DD", "BBB"],
                       ['R16D00C', 'gh_ij_ij', 0, 4.75, "#b#a#", "F", 1, "EE", ""],
                       ['N08W87W', 'ab____kl', 12.8, 7.54, "b#b#d#d#e#e", "F", 0, "FF", "AAA"],
                       ['U13F20V', '_', 12.8, 98.12, "b#b#d#d#a#a", "F", 0, "GG", "AAA"],
                       ['P42G78B', None, -3.28, 42.36, "b#b###e", "M", 0, "HH", "CCC"],
                       ['M98H12N', 'ab_kl_', 1478, -18.45, "a##c#d#", "M", 0, "II", "BBB"],
                       ['W45J98P', '_gh_cd', 6., 1e-05, "#d", "F", 0, "JJ", None],
                      ],
                      columns=column_names)
    print("\nData Frame (raw)\n")
    print(df)

    p = Preprocessor()
    data = p.fit_transform(df.copy(), pruning_frequency=None, do_not_use=['id'],
                           variance_threshold=0, low_memory=True,
                           sharp_categorical_dict={'sc1': {'sep': "_", 'norm': False}, 'sc2': {'sep': "#", 'norm': True}},
                           na_strategy=MedianStrategy())
    print("\nData Frame (preprocessed)\n")
    print(pd.concat([data.other, pd.DataFrame(data.X.todense(), columns=data.names)], axis=1))

    column_names = ['id', 'sc1', 'numeric1', 'numeric2', 'sc2', 'gender', 'target', 'c1', 'c2']
    df = pd.DataFrame([['A34G56G', 'ab', 34.5, None, "c", "F", 0, "AA","AAA"],
                       ['P34N12C', 'abh', None, 98, "", "", 1, "BB", "AAA"],
                       ['V27H74P', 'wdf', -18.4, -10.45, "a", "M", 1, "CC", "BBB"],
                       ['V27H74P', 'cdz', 0.75, 1.021, "a", "M", 1, "", "CCC"],
                       ['G98S45X', 'ija', 78.45, 9874, None, "", 1, "DD", "BBB"],
                       ['A34G56G', 'ij', 0, 4.75, "a", "F", 1, "EE", ""],
                       ['A34G56G', 'kl', 12.8, 7.54, "e", "F", 0, "FF", "AAA"],
                       ['U13F20V', 'pot', 12.8, 98.12, "d", "F", 0, "GG", "AAA"],
                       ['A34G56G', None, -3.28, 42.36, "b", "M", 0, "HH", "CCC"],
                       ['M98H12N', 'klu', 1478, -18.45, "a", "M", 0, "II", "BBB"],
                       ['U13F20V', 'gh', 6., 1e-05, "d", "F", 0, "JJ", None],
                      ],
                      columns=column_names)
    print("\nData Frame (raw, not aggregated)\n")
    print(df)

    strategy = {"gender": max,
                "sc1": "counter",
                "sc2": "normalized_counter",
                "c1": lambda x: list(set(x)),
                "c2": "counter",
                "target": max,
                "numeric1": np.mean,
                "numeric2": np.median}
    a = Aggregator(aggregation_keys='id', aggregation_strategy=strategy, vectorizor_compatibility=False)
    agg_df = a.fit_transform(df.copy())
    print("\nData Frame (aggregated)\n")
    print(agg_df)

    strategy = {"gender": max,
                "sc1": "counter",
                "sc2": "normalized_counter",
                "c1": "counter",
                "c2": "counter",
                "target": max,
                "numeric1": np.mean,
                "numeric2": np.median}
    a = Aggregator(aggregation_keys='id', aggregation_strategy=strategy)
    agg_df = a.fit_transform(df.copy())
    print("\nData Frame (aggregated, vectorizor compatible)\n")
    print(agg_df)

    v = Vectorizor(ignored_columns=['id', 'target'])
    data = v.fit_transform(agg_df)
    print("\nData Frame (vectorized)\n")
    print(pd.concat([data.df, pd.DataFrame(data.sm.todense(),columns=data.sm_columns)], axis=1))

    print("\n")
    unittest.main()
