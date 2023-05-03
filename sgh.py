from sklearn.ensemble import BaseEnsemble
from sklearn.linear_model import SGDClassifier
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.utils.validation import check_X_y

from math import sqrt

from copy import deepcopy
import warnings
warnings.filterwarnings("ignore", message="Maximum number of iteration")

class SGH(BaseEnsemble):
    """
    Self-Generating Hyperplanes (SGH).

    Generates a pool of classifiers which guarantees an Oracle
    accuracy rate of 100% over the training (input) set.
    That is, for each instance in the training set, there is at
    least one classifier in the pool able to correctly label it.
    The generated classifiers are always two-class hyperplanes. 


    References
    ----------

    L. I. Kuncheva, A theoretical study on six classier fusion
    strategies, IEEE Transactions on
    Pattern Analysis and Machine Intelligence 24 (2) (2002) 281-286.

    M. A. Souza, G. D. Cavalcanti, R. M. Cruz, R. Sabourin,
    On the characterization of the
    oracle for dynamic classier selection, in: International
    Joint Conference on Neural Networks,
    IEEE, 2017, pp. 332-339.

    """

    def __init__(self,
                 base_estimator=SGDClassifier,
                 n_estimators=1,
                 correct_classif_label=[]
                 ):

        super(SGH, self).__init__(base_estimator=base_estimator,
                                  n_estimators=1)

        # Pool initially empty
        self.estimators_ = []
        self.base_estimator_ = base_estimator


    def fit(self, X, y, included_samples=np.array([]), sample_weights=None):
        """
        Populates the SHG ensemble.

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The training data.

        y : array of shape = [n_samples]
            class labels of each example in X.

        included_samples : array of shape = [n_samples]
            array of ones and zeros ('1','0'), indicating which
            samples in X are to be used for training.
            If all, leave blank.

        sample_weights : array of shape = [n_samples]
            array of float indicating the weight of each sample in X.
            Default is None.

        Returns
        -------
        self

        """

        check_X_y(X, y)
        return self._fit(X, y, included_samples, sample_weights=sample_weights)

    def _fit(self, X, y, included_samples, sample_weights=None):

        # Set base estimator as the Perceptron
            # SGDClassifier(loss="perceptron", eta0=1.e-17,
            #                                  max_iter=1,
            #                                  learning_rate="constant",
            #                                  penalty=None)

        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.size

        # If there is no indication of which instances to
        # include in the training, include all
        if included_samples.sum() == 0:
            included_samples = np.ones((X.shape[0]), int)

        # Generate pool
        self._generate_pool(X, y, included_samples,
                            sample_weights=sample_weights)

        return self

    def _build_Perceptron(self, X, y, curr_training_samples, centroids, sample_weights):
        """
        Calculates the parameters (weight and bias) of the hyperplane
        placed in the midpoint between the centroids of most distant
        classes in X[curr_training_samples].


        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The training data.

        y : array of shape = [n_samples]
            class labels of each example in X.

        curr_training_samples : array of shape = [n_samples]
            array of ones and zeros ('1','0'), indicating which
            samples in X are to be used for placing the hyperplane.

        centroids : array of shape = [n_classes,n_features]
            centroids of each class considering the previous
            distribution of X[curr_training_samples].

        sample_weights : array of shape [n_samples]
            weights used to compute the centroid of
            each class in X[curr_training_samples], as well as
            increase/decrease their margin.


        Returns
        -------

        perc : SGDClassifier
            perceptron placed between the centroids
            of X[curr_training_samples].

        centroids : array of shape = [n_classes,n_features]
            updated centroids of each class considering the
            distribution of X[curr_training_samples].

        """

        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = classes.size
        weights = np.zeros((n_classes, n_features), float)
        bias = np.zeros((n_classes), float)

        # Vector indicating the remaining classes in eval_X/eval_y
        curr_classes = np.zeros((n_classes), int)

        mask_incl_samples = np.asarray(curr_training_samples, dtype=bool)

        if sample_weights is None:
            sample_weights = np.ones(n_samples) / n_samples

        midpt_weights = np.ones(n_classes)

        for i in range(0, n_classes):
            # Select instances from a single class
            mask_c = classes[i] == y

            mask_sel = np.logical_and(mask_c, mask_incl_samples)

            c = X[mask_sel]
            # midpt_weights[i] = 1 - np.mean(sample_weights[mask_sel])
            if c.size:
                # Update centroid of class
                if np.sum(sample_weights[mask_sel]) == 0:
                    centroids[i,] = np.average(c, axis=0,
                                               weights=None)
                else:
                    centroids[i,] = np.average(c, axis=0,
                                               weights=sample_weights[mask_sel])
                # Indicate its presence
                curr_classes[i] = 1

        idx_curr_classes = np.where(curr_classes > 0)

        if curr_classes.sum() >= 2:  # More than 2 classes remain
            # Pairwise distance between current classes
            dist_classes = squareform(pdist(centroids[idx_curr_classes[0]]))
            np.fill_diagonal(dist_classes, np.inf)

            # Identify the two farthest away
            closest_dist = np.unravel_index(np.argmin(dist_classes),
                                            dist_classes.shape)

            idx_class_1 = idx_curr_classes[0][closest_dist[0]]
            idx_class_2 = idx_curr_classes[0][closest_dist[1]]

        else:  # Only one class remains
            # Pairwise distance between all classes in the problem
            dist_classes = squareform(pdist(centroids))
            np.fill_diagonal(dist_classes, np.inf)

            # Remaining class
            idx_class_1 = idx_curr_classes[0][0]
            # Most distant from class_1
            idx_class_2 = np.argmin(dist_classes[idx_class_1,])

            # Difference vector between selected classes
        diff_vec = centroids[idx_class_1,] - centroids[idx_class_2,]

        if not np.any(diff_vec):
            # print('Equal classes centroids!')
            w_p = 0.01 * np.ones((n_features), float)
            w_p = w_p / sqrt(((w_p) ** (2)).sum())
        else:
            # Normal vector of diff_vec
            w_p = diff_vec / sqrt(((diff_vec) ** (2)).sum())

        sum_vec = (midpt_weights[idx_class_1] * centroids[idx_class_1,] +
                   midpt_weights[idx_class_2] * centroids[idx_class_2,]) / np.sum(midpt_weights)

        theta_p = np.dot(-w_p, sum_vec)

        # Weights of linear classifier
        weights[idx_class_1,] = w_p
        weights[idx_class_2,] = -w_p

        # Bias of linear classifier
        bias[idx_class_1,] = theta_p
        bias[idx_class_2,] = -theta_p

        assert not np.isnan(theta_p)

        # Generate classifier
        perc = self.base_estimator_(max_iter=1)
        #     SGDClassifier(loss="perceptron", eta0=1.e-17, max_iter=1,
        #                      learning_rate="constant", penalty=None)

        perc.classes_ = classes
        perc.fit(X, y)

        # Set classifier's weigths and bias
        perc.coef_ = weights
        perc.intercept_ = bias

        # Return perceptron
        return perc, centroids

    def _generate_pool(self, X, y, curr_training_samples,
                       sample_weights=None):
        """
        Generates the classifiers in the pool of classifiers
        ("estimators_") using the SGH method.

        In each iteration of the method, a hyperplane is
        placed in the midpoint between the controids of the
        two most distant classes in the training data.
        Then, the newly generated classifier is tested over
        all samples and the ones it correctly labels are
        removed from the set.
        In the following iteration, a new hyperplane is
        created based on the classes of the remaining samples
        in the training set.
        The method stops when no sample remains in the training set.

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The training data.

        y : array of shape = [n_samples]
            class labels of each example in X.

        curr_training_samples : array of shape = [n_samples]
            array of ones and zeros ('1','0'), indicating which
            samples in X are to be used for training.
            If all, leave blank.
            

        Returns
        -------
        self
        """

        # Input data and samples included in the training
        n_samples, n_features = X.shape

        # Labels of the correct classifier for each training sample
        corr_classif_lab = np.zeros((n_samples), int)

        # Pool size
        n_perceptrons = 0

        n_err = 0
        max_err = 3

        # Problem's classes
        classes = np.unique(y)
        n_classes = classes.size

        # Centroids of each class
        centroids = np.zeros((n_classes, n_features), float)

        self.input_data = []

        # While there are still misclassified samples
        while curr_training_samples.sum() > 0 and n_err < max_err:
            # Generate classifier
            self.input_data.append(deepcopy(curr_training_samples))

            curr_perc, centroids = self._build_Perceptron(X, y,
                                                          curr_training_samples,
                                                          centroids, sample_weights)

            # Add classifier to pool
            self.estimators_.append(deepcopy(curr_perc))

            # Obtain set with instances that weren't correctly classified yet
            idx_curr_training_samples = np.where(curr_training_samples > 0)
            eval_X = X[idx_curr_training_samples[0]]
            eval_y = y[idx_curr_training_samples[0]]

            # Evaluate generated classifier over eval_X
            out_curr_perc = self.estimators_[n_perceptrons].predict(eval_X)

            # Identify correctly classified samples
            idx_correct_eval = (out_curr_perc == eval_y).nonzero()

            # Exclude correctly classified samples from current training set
            curr_training_samples[
                idx_curr_training_samples[0][idx_correct_eval[0]]] = 0

            # Set classifier label for the correctly classified instances
            corr_classif_lab[idx_curr_training_samples[0][
                idx_correct_eval[0]]] = n_perceptrons
            # Increase pool size
            n_perceptrons += 1
            n_err += 1

        # Update pool size
        self.n_estimators = n_perceptrons
        # Update classifier labels
        self.correct_classif_label = corr_classif_lab

        return self
