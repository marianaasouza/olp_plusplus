import numpy as np
import functools
from copy import deepcopy

from sklearn.utils.validation import check_X_y
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from deslib.base import BaseDS

from sgh import SGH
from treenn import TreeNN


class OLPpp(BaseDS):
    """
    Online Local Pool++ (OLP++) technique.

    This technique dynamically generates linear classifiers over the
    data partitions where a given query instance falls into,
    if in the presence of class overlap.
    The data partitions are obtained using one or more tree-based algorithms.
    The responses of the local experts are then used to label the
    unknown sample.

    Parameters
    ----------

    k : int (Default = 7)
        Minimum number of samples within a leaf node.

    n_classifiers : int (default = 2)
        Number of local classifiers (or local pool size) to be
        generated within a given decision path the query traverses.
        This also indicates how many nodes/levels of the tree structure
        will be used for generating the local pool.

    n_trees : int (default = 1)
        Number of trees to be used for the region definition step.

    knn_classifier : {'sklearn'} or None
        Algorithm to be used for generating the data partitions.

    References
    ----------
    Souza, M. A., Sabourin, R., Cavalcanti, G. D., & Cruz, R. M. (2023).
    OLP++: An online local classifier for high dimensional data.
    Information Fusion, 90, 120-137.
    """
    def __init__(self,
                 k=7,
                 n_classifiers=2,
                 n_trees=1,
                 knn_classifier=None,
                 n_jobs=1,
                 random_state=None):

        super().__init__(pool_classifiers=None, k=k, DFP=False,
                         with_IH=False, knne=False,
                         random_state=random_state,
                         knn_classifier=knn_classifier)

        self.name = 'OLPpp'
        self.pool_classifiers = None

        self.hardness_function = None
        self.n_trees = n_trees
        self.knn_classifier = knn_classifier
        self.n_classifiers = n_classifiers
        self.n_jobs = n_jobs

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
        check_X_y(X, y)
        self._set_dsel(X, y)
        self._set_region_of_competence_algorithm()
        self._fit_region_competence(X, y)

        return self

    def _set_region_of_competence_algorithm(self):
        if self.knn_classifier is None or self.knn_classifier == 'sklearn':
            if self.n_trees == 1:
                self.knn_class_ = functools.partial(TreeNN,
                                                    model_type=DecisionTreeClassifier,
                                                    random_state=self.random_state)

            else:
                self.knn_class_ = functools.partial(TreeNN,
                                                    model_type=RandomForestClassifier,
                                                    n_estimators=self.n_trees,
                                                    n_jobs=self.n_jobs,
                                                    bootstrap=False,
                                                    random_state=self.random_state)
        else:
            raise ValueError('"knn_classifier" should be "sklearn" or None.')

        self.roc_algorithm_ = self.knn_class_(min_samples_leaf=self.k)

    def _set_dsel(self, X, y):
        """
        Get information about the structure of the data
        (e.g., n_classes, n_samples, classes)

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The Input data.

        y : array of shape (n_samples)
            class labels of each sample in X.
        
        Returns
        -------
        self
        """
        self.DSEL_data = X
        self.DSEL_target = y
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]
        self.n_samples = self.DSEL_target.size

        return self

    def _check_parameters(self):
        """
        Verifies if the input parameters are correct
        raises an error if values are below 1.
        """
        if self.k is not None:
            if not isinstance(self.k, int):
                raise TypeError("parameter k should be an integer")
            if self.k < 1:
                raise ValueError("parameter k must equal or higher than 1."
                                 "input k is {} ".format(self.k))
        if self.n_trees is not None:
            if not isinstance(self.n_trees, int):
                raise TypeError("parameter n_trees should be an integer")
            if self.n_trees < 1:
                raise ValueError("parameter n_trees must be equal or higher than 1."
                                 "input n_trees is {} ".format(self.n_trees))
        if self.n_classifiers is not None:
            if not isinstance(self.n_classifiers, int):
                raise TypeError("parameter n_classifiers should be an integer")
            if self.n_classifiers < 1:
                raise ValueError("parameter n_classifiers must be equal or higher than 1."
                                 "input n_classifiers is {} ".format(self.n_trees))

    def _check_is_fitted(self):

        if self.roc_algorithm_ is None:
            raise NotFittedError("DS method not fitted, "
                                 "call `fit` before exploiting the model.")

    def _generate_local_pool(self, query, masks_neighbors):
        """
        Generates the local linear classifiers over the target data
        partitions where the query instances fall into.

        Parameters
        ----------
        query : array of shape (n_query, n_features)
            The test samples

        masks_neighbors : list containing arrays of shape (n_query, n_train, n_trees)
            Mask indicating the training samples within the data partition that
            each one of the query instances fall into at the target tree(s) levels.

        Returns
        -------
        pool_classifiers : list of length (n_query)
            Local pool of classifiers obtained for each instance.
            Decision paths that end in homogeneous leaf nodes yield an empty list
            instead of a local pool.

        classifiers_outputs_ : arrays of shape (n_trees, n_classifiers, n_query)
            Label votes of the local decision rules obtained from the local
            pool and target data partitions.

        """

        query = np.atleast_2d(query)
        pool_classifiers = []
        classifiers_outputs_ = np.zeros((self.n_trees,
                                         self.n_classifiers,
                                         query.shape[0]),
                                        dtype=int)

        for index, instance in enumerate(query):

            # Keep record of the branches that end in homogeneous leaves
            local_pool = []

            for i in range(self.n_trees):

                for j in range(self.n_classifiers):
                    # Mask indicating neighbors of query at the j-th level of the trees
                    mask_neighbors = masks_neighbors[j][index]

                    included_samples = np.logical_not(np.ma.getmaskarray(mask_neighbors))[:, i]

                    curr_classes, count_classes = np.unique(self.DSEL_target[included_samples],
                                                            return_counts=True)

                    # If there are two or more classes in the local region
                    if len(curr_classes) > 1:

                        # Obtain SGH pool
                        subpool = SGH()
                        subpool.fit(self.DSEL_data,
                                    self.DSEL_target, included_samples[:])

                        # Save selected classifier from subpool
                        local_pool.append(deepcopy(subpool[0]))

                        classifiers_outputs_[i, j, index] = int(
                            subpool.estimators_[0].predict(instance.reshape(1, -1)))

                    else:
                        # Branch with homogeneous leaf
                        classifiers_outputs_[i, :, index] = curr_classes[0]
                        break

            pool_classifiers.append(local_pool)

        return pool_classifiers, classifiers_outputs_

    def select(self, query):
        """
        Returns the generated local pool for the query instances.

        Parameters
        ----------
        query : array of shape (n_query, n_features)
            The test samples

        Returns
        -------
        pool_classifiers : list of length (n_query)
            Local pool of classifiers obtained for each instance.
            Decision paths that end in homogeneous leaf nodes yield an empty list
            instead of a local pool.

        """
        masks_neighbors = self._obtain_local_regions(query)
        pool_classifiers, _ = \
            self._generate_local_pool(query, masks_neighbors)

        return pool_classifiers

    def _obtain_local_regions(self, query):
        """
        Returns masks indicating the training samples within
        the data partitions the query instances fall into.

        Parameters
        ----------
        query : array of shape = (n_query, n_features)
                The test samples

        Returns
        -------
        masks_neighbors : list containing arrays of shape (n_query, n_train, n_trees)
            Mask indicating the training samples within the data partition that
            each one of the query instances fall into at the target tree(s) levels.
        """
        masks_neighbors = []
        for j in range(self.n_classifiers):
            # Current level of tree(s), from the leaf upwards
            curr_tree_level = -(j + 1)
            mask_neighbors_at_level = self.roc_algorithm_.kneighbors(X=query,
                                                                     tree_level=curr_tree_level,
                                                                     return_distance=False,
                                                                     return_subspaces=False)[0]
            masks_neighbors.append(mask_neighbors_at_level)
        return masks_neighbors

    def classify_with_ds(self, query):
        """
        Predicts the label of the query samples.

        The prediction is made by aggregating the votes obtained from
        all data partitions considered in the decision path(s).

        Parameters
        ----------
        query : array of shape = (n_query, n_features)
                The test samples

        Returns
        -------
        predicted_label : Prediction of the ensemble for the input query.
        """

        masks_neighbors = self._obtain_local_regions(query)

        # Generate local pool and obtain their outputs
        pool_classifiers, classifiers_outputs_ = \
            self._generate_local_pool(query, masks_neighbors)

        votes = classifiers_outputs_.reshape(-1, query.shape[0])
        weights = np.ones_like(votes)
        class_support = np.zeros((len(self.classes), query.shape[0]))
        for i, c in enumerate(self.classes):
            mask_class = votes == c
            class_support[i, :] = np.sum(np.ma.MaskedArray(weights,
                                                           mask=np.logical_not(mask_class)),
                                         axis=0)
        class_support /= np.sum(weights, axis=0)

        predicted_label = self.classes[np.argmax(class_support, axis=0)]

        return predicted_label

    def predict(self, X):
        """
        Predicts the class label for each sample in X.

        Parameters
        ----------
        X : array of shape = (n_query, n_features)
            The input data.

        Returns
        -------
        predicted_labels : array of shape = (n_query)
            Predicted class label for each sample in X.
        """
        # Check if the roc algorithm was trained
        self._check_is_fitted()
        predicted_labels = self.classify_with_ds(X)

        return predicted_labels


if __name__ == "__main__":

    # High-dimensional toy example

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=100,
                               n_features=200,
                               n_informative=100,
                               random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.25,
                                                        random_state=42)

    _olppp = {
        'OLP++(DT)':
            {
                'n_trees': 1,
             },
        'OLP++(RF)':
            {
                'n_trees': 10,
            },
    }
    for clf in _olppp.keys():
        partolp = OLPpp(k=7,  # min_leaf_size
                        n_classifiers=2,
                        n_trees=_olppp[clf]['n_trees'],
                        random_state=1)
        partolp.fit(X_train, y_train)
        y_out = partolp.predict(X_test)
        print(clf + ' accuracy rate:')
        print(np.mean(y_test == y_out))

    rf = RandomForestClassifier(n_estimators=100,
                                random_state=1)
    rf.fit(X_train, y_train)
    print('RF accuracy rate:')
    print(rf.score(X_test, y_test))

    