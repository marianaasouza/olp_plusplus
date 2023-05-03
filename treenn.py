import numpy as np
from sklearn.base import BaseEstimator

from sklearn.utils import check_X_y
from sklearn.utils import check_array

from deslib.util.prob_functions import softmax

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class TreeNN(BaseEstimator):
    """
    Wrapper class for the K-NN substitute in the region of competence
    definition step of the dynamic selection techniques.
    The TreeNN model retrieves the samples within one or several data partitions
    constructed from tree-based models where a given query sample falls into.

    Parameters
    ----------
    tree_level :  int (negative), default=-1
        Level of the target data partition, starting from the last level in the
        decision path (leaf= -1) upwards. Default is the leaf level.

    model_type : {"dt", "rf"} or class, default="dt"
        Tree model type, "dt" stands for Decision Tree
        (sklearn.tree.DecisionTreeClassifier class) and "rf" for Random Forest
        (sklearn.ensemble.RandomForestClassifier). The tree-based class must
        inherit from sklearn.tree.BaseDecisionTree or
        sklearn.ensemble.ForestClassifier.

    kwargs: dict
        Keyword arguments for fitting the model indicated in model_type.
    """

    def __init__(self,
                 tree_level=-1,
                 model_type='dt',
                 **kwargs):

        self.model_type = model_type
        self.kwargs = kwargs
        self.tree_level = tree_level
        self.n_trees = None

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Data used to fit the model.
        y : array of shape (n_samples)
            class labels of each example in X.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y)

        self.classes_indexes_ = {}
        self.fit_X_ = X
        self.fit_y_ = y
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.size

        # Check inputs
        self._check_tree_level(self.tree_level)

        # Set and fit tree model type
        self._set_model_type()
        self.tree_ = self.model_type_(**self.kwargs)
        self.tree_.fit(X, y)

        # Set ensemble size in case of a tree ensemble
        try:
            self.n_trees = self.tree_.n_estimators
        except AttributeError:
            self.n_trees = 1

        return self

    def kneighbors(self, X=None, tree_level=None, return_distance=True,
                   return_subspaces=False):
        """
        Returns the neighbors of the instances in X, that is, the samples within
        the data partitions they fall into.

        Parameters
        ----------
        X : array-like, shape (n_query, n_features)
            Test samples.

        tree_level : int (negative), default=-1
            Level of the target partition, starting from the last level in the
            decision path (leaf= -1) upwards. Default is the leaf level.

        return_distance :  bool, default=True
            If True, will return the tree-based distances between the test samples
            and the training instances.

        return_subspaces : bool, default=False
            If True, will return a mask indicating the features used in the
            test samples' decision path until the indicated target level.

        Returns
        -------
        mask_inds : array-like, shape (n_query, n_train, n_trees)
            Mask indicating the training samples within the data partition that
            each one of the query instances fall into at the target tree(s) level.

        dists : array-like, shape (n_query, n_train)
            Tree-based distances between the test samples
            and the training instances in each tree.

        mask_subspaces : array-like, shape (n_query, n_features, n_trees)
            Mask indicating the features selected until the target tree(s) level of
            the data partition that each one of the query instances fall into.
        """

        # If no tree level was indicated, use the one previously set
        if tree_level is None or tree_level > 0:
            tree_level = self.tree_level

        if X is None:
            # Get partitions from each training sample
            X = self.fit_X_

        # Obtain the masks indicating the samples within the target data partition
        # and the corresponding feature subspaces
        mask_partitions, mask_subspaces = self._obtain_partitions(X, tree_level)

        # Obtain the tree-based distances from the partitions mask
        # 1.0 for samples outside target partition, 0.0 otherwise
        # averaged over all trees
        dists = self._obtain_distances(mask_partitions)

        # Indexes array
        inds = np.ones_like(dists, dtype=int) * np.arange(mask_partitions.shape[1])

        # If the model is a single tree, reshape the masks to the correct dimensions
        if self.n_trees < 2:
            mask_subspaces = np.repeat(mask_subspaces[:, :, np.newaxis], self.n_trees, axis=2)
            mask_partitions = np.repeat(mask_partitions[:, :, np.newaxis], self.n_trees, axis=2)
        inds = np.repeat(inds[:, :, np.newaxis], self.n_trees, axis=2)

        # Masked array indicating the indexes of the samples within the
        # target partitions of each instance in X
        mask_inds = np.ma.MaskedArray(inds,
                                      mask=np.logical_not(mask_partitions))
        # Return structure
        ret = []
        if return_distance:
            ret.append(dists)
        if return_subspaces:
            ret.append(mask_subspaces)
        ret.append(mask_inds)

        return tuple(ret)

    def predict(self, X):
        """
        Predict the class label for each sample in X based on the class
        distribution in the partitions where the instances from X fall into
        at the pre-set tree level.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The input data.
        Returns
        -------
        preds : array, shape (n_samples,)
                Class labels for samples in X.
        """
        proba = self.predict_proba(X)
        return self.classes_.take(np.argmax(proba, axis=1), axis=0)

    def predict_proba(self, X):
        """
        Return probability estimates for the for each sample in X
        based on the class distribution in the partitions where the
        instances from X fall into at the pre-set tree level.

        Parameters
        ----------
        X : array-like, shape (n_query, n_features)
            Test samples.

        Returns
        -------
        proba : array of shape (n_samples, n_classes), or a list of n_outputs
            of such arrays if n_outputs > 1.
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        X = check_array(X, accept_sparse='csr')

        dists, _ = self.kneighbors(X, return_distance=True,
                                   return_subspaces=False)
        inds = np.ones_like(dists, dtype=int) * np.arange(dists.shape[1])

        classes = self.fit_y_[inds]
        sim_array = np.empty((X.shape[0], self.n_classes_))
        funct = lambda m: np.ma.mean(m, axis=1) if self.n_trees > 1 \
            else np.ma.sum(m, axis=1)

        for c in self.classes_:
            # invalidate instances do not belong to the same partition
            part_class_mask = np.logical_or(classes != c, dists == 1.)
            # Sum (single tree) or average (tree ensemble) class support for
            # the label c
            sim_array[:, c] = funct(np.ma.MaskedArray(1 - dists, part_class_mask))
        probas = softmax(sim_array)
        return probas

    def _obtain_partitions(self, X, tree_level=-1):
        """
        Obtain the masks indicating the samples within the tree(s) partitions
        at the indicated tree level where each instance in X falls,
        as well as the corresponding feature subspace from the tree's
        decision path at the indicated tree level.

        Parameters
        ----------
        X : array-like, shape (n_query, n_features)
            Test samples.

        tree_level : int (negative), default=-1
            Level of the target partition, starting from the last level in the
            decision path (leaf= -1) upwards. Default is the leaf level.

        Returns
        -------
        partition_neighbors_mask : array-like, shape (n_query, n_train, n_trees) or
                                        (n_query, n_train) if self.n_trees = 1
            Mask indicating the training samples within the data partition that
            each one of the query instances fall into at the target tree level.

        subspaces : array-like, shape (n_query, n_features, n_trees) or
                        (n_query, n_features) if self.n_trees = 1
            Mask indicating the features selected until the target tree level of
            the data partition that each one of the query instances fall into.
        """
        # Number of levels to jump from leaf node
        target_tree_level = -tree_level - 1
        # Single tree
        if self.n_trees < 2:
            node_indicator_train = self.tree_.decision_path(self.fit_X_)
            dense_indicator_train = np.asarray(node_indicator_train.todense())
            partition_neighbors_mask, subspaces = self._obtain_partition_from_tree(self.tree_,
                                                                                      dense_indicator_train,
                                                                                      X,
                                                                                      target_tree_level)
        # Ensemble of trees
        else:
            partition_neighbors_mask = np.zeros((X.shape[0],
                                                    self.fit_X_.shape[0],
                                                    len(self.tree_.estimators_)),
                                                   dtype=int)
            subspaces = np.zeros((X.shape[0], X.shape[1], len(self.tree_.estimators_)),
                                 dtype=int)
            # Gather the partitions from each tree in the ensemble
            for j, tree in enumerate(self.tree_.estimators_):
                node_indicator_train = tree.decision_path(self.fit_X_)
                dense_indicator_train = np.asarray(node_indicator_train.todense())
                partition_neighbors_mask[:, :, j], subspaces[:, :, j] = \
                    self._obtain_partition_from_tree(tree,
                                                     dense_indicator_train,
                                                     X,
                                                     target_tree_level)
        return partition_neighbors_mask, subspaces

    def _obtain_distances(self, partition_neighbors_mask):
        """
        Obtain the tree-based distances between the training samples and
        the instances in X from the partitions mask.

        Parameters
        ----------
        partition_neighbors_mask : array-like, shape (n_query, n_train, n_trees) or
                                        (n_query, n_train) if self.n_trees = 1
            Mask indicating the training samples within the data partition that
            each one of the query instances fall into at the target tree level.

        Returns
        -------
        dists : array-like, shape (n_query, n_train)
            Tree-based distances between the test samples
            and the training instances in each tree.
        """
        if self.n_trees < 2:
            dists = np.asarray(np.logical_not(partition_neighbors_mask),
                               dtype=float)

        else:  # ensemble
            dists = np.asarray(np.mean(np.logical_not(partition_neighbors_mask),
                                       axis=2),
                               dtype=float)

        return dists

    @staticmethod
    def _obtain_partition_from_tree(tree, train_indicator, X, target_tree_level=0):
        """
        Obtain the data partitions at the target level from the decision path
        the samples in X traverse in a single tree.

        Parameters
        ----------
        tree : instance of sklearn.tree.DecisionTreeClassifier class
            Tree model.

        train_indicator: sparse matrix of shape (n_train, n_nodes)
            Indicator of the decision paths all training samples
            traverse in the tree.

        X : array-like, shape (n_query, n_features)
            Test samples.

        target_tree_level : int (non-negative), default=0
            Level of the target partition, in number of "jumps" from the
            leaf level necessary to reach it. Default is the leaf level.

        Returns
        -------
        partition_neighbors_mask : array-like, shape (n_query, n_train)
            Mask indicating the training samples within the data partition that
            each one of the query instances fall into at the target tree level.

        subspaces : array-like, shape (n_query, n_features)
            Mask indicating the features selected until the target tree level of
            the data partition that each one of the query instances fall into.
        """
        # Indicator of decision path for each sample in X
        indicator = np.asarray(tree.decision_path(X).todense())
        # Zero for unvisited nodes, node id + 1 for visited nodes
        added_indicator = indicator * np.arange(1, indicator.shape[1] + 1)
        # Sort nodes from leaf to root
        ascending_nodes_tst = np.argsort(-added_indicator, axis=1)

        # Identify subspaces based on indicator/nodes
        feature = tree.tree_.feature
        feature_indicator = np.ones_like(indicator) * np.arange(indicator.shape[1])

        mask_all_nodes_tst = np.ma.MaskedArray(feature[feature_indicator],
                                               mask=np.logical_not(
                                                   np.logical_and(indicator,
                                                                  feature[feature_indicator] >= 0)))
        subspaces = np.zeros((X.shape[0], X.shape[1]), dtype=int)
        xs, ys = np.ma.where(mask_all_nodes_tst >= 0)
        yfs = mask_all_nodes_tst[xs, ys]
        subspaces[xs, yfs] = 1

        # Identify depth of path from the root node position
        path_depth_tst = np.argwhere(ascending_nodes_tst == 0)[:, 1]
        mask_too_shallow = path_depth_tst < target_tree_level
        target_nodes_tst = ascending_nodes_tst[np.arange(mask_too_shallow.shape[0]),
                                               np.min(np.c_[target_tree_level*np.ones_like(path_depth_tst),
                                                            path_depth_tst], axis=1)]

        target_nodes_tst[mask_too_shallow] = 0
        indicator_idxs = np.meshgrid(
            np.arange(train_indicator.shape[0]),
            target_nodes_tst,
            indexing='ij')
        # Aggregate information on the presence of all training
        # instances in the target nodes
        partition_neighbors_mask = train_indicator[indicator_idxs[0],
                                                      indicator_idxs[1]].T

        return partition_neighbors_mask, subspaces

    def _set_model_type(self):
        if self.model_type is None or self.model_type == 'dt':
            self.model_type_ = DecisionTreeClassifier

        elif callable(self.model_type):
            self.model_type_ = self.model_type
        else:
            raise ValueError('"model_type" should be one of the following '
                             '["dt", "rf", None] or an estimator class.')

    def _check_tree_level(self, tree_level):
        if tree_level is None:
            raise ValueError('"tree_level" is required for the Tree-NN model.')

        if tree_level > 0:
            raise ValueError('"tree_level" must be a negative integer. Got {}.'
                             .format(tree_level))

        if not np.issubdtype(type(tree_level), np.integer):
            raise TypeError(
                "tree_level does not take {} value, "
                "enter integer value".format(type(tree_level)))


if __name__ == "__main__":

    # High-dimensional toy example

    X, y = make_classification(n_samples=1000,
                               n_features=2000,
                               n_informative=80,
                               random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.25,
                                                        random_state=42)

    dict_clf = {
        'TreeNN(RF)':
            {
                'model_type': RandomForestClassifier,
                'min_samples_leaf': 7,
                'n_estimators': 10,
                'bootstrap': False,
                'random_state': 20,
             },
        'TreeNN(DT)':
            {
                'model_type': DecisionTreeClassifier,
                'min_samples_leaf': 7,
                'random_state': 20,
             },
    }

    for clf in dict_clf.keys():
        treeNN = TreeNN(tree_level=-2,
                        **dict_clf[clf])
        treeNN.fit(X_train, y_train)
        y_out = treeNN.predict(X_test)
        print(clf + ' accuracy rate:')
        print(np.mean(y_test == y_out))

    rf = RandomForestClassifier(random_state=20)
    rf.fit(X_train, y_train)
    print('RF accuracy rate:')
    print(rf.score(X_test, y_test))

