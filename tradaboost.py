import numpy as np
import numbers
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import _check_sample_weight
from sklearn.base import clone
from dataset.twenty_newsgroups import fetch_transfer


def _validate_data(X, y=None):
    accept_sparse = ['csr', 'csc']
    if y is None:
        ret = check_array(X,
                          accept_sparse=accept_sparse,
                          ensure_2d=False,
                          allow_nd=True,
                          dtype=None)
    else:
        ret = check_X_y(X, y,
                        accept_sparse=accept_sparse,
                        ensure_2d=False,
                        allow_nd=True,
                        dtype=None)
    return ret


class TrAdaBoost(BaseEstimator, ClassifierMixin):
    """
    Attribute
    ------
    estimators_ : list of estimator
    estimator_weights_ : ndarray of shape (n_estimators, )
    estimator_errors_ : ndarray of shape (n_estimators, )
    n_estimators : number of estimators
    n_classes : number of classes
    classes_ : labels for classifier
    """
    def __init__(self, n_estimators=30, base_estimator=LinearSVC(), estimator_params=tuple()):
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.estimator_params = estimator_params

    def _set_param(self, X, y, partition, sample_weight=None):
        X, y = _validate_data(X, y)

        sample_weight = _check_sample_weight(sample_weight, X, np.float64)
        sample_weight /= sample_weight.sum()
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative weights")

        self._validate_estimator()
        self.n_samples = X.shape[0]
        self.n_task = self.n_samples - partition
        self.n_source = partition

        self.estimators_ = []
        self.beta_t = np.ones(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.beta = 1. / (1. + np.sqrt(2. * np.log(partition) / self.n_estimators))
        return X, y, sample_weight

    def fit(self, X, y, partition, sample_weight=None):
        """
        :param X: {array-like, sparse matrix} of shape (n_samples, n_features)
        :param y: array-like of shape (n_samples, )
        :param partition: integer
            parting the data to source domain and destination domain
        :param sample_weight: array-like of shape (n_samples, ), default=None
        :return: self: object
        """
        # Check that X and y have correct shape
        X, y, sample_weight = self._set_param(X, y, partition, sample_weight)
        for iboost in range(self.n_estimators):
            sample_weight, beta_t, estimator_error = self._boost(
                iboost,
                X, y,
                sample_weight,
                partition
            )

            if sample_weight is None:
                pass

            # print('After %d round, the training error is %.16lf' % (iboost, estimator_error))
            self.beta_t[iboost] = beta_t
            self.estimator_errors_[iboost] = estimator_error
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                sample_weight /= sample_weight_sum
        self.sample_weight = sample_weight
        return self

    def predict(self, X):
        """Predict classes for X.
        :param X : {array-like, sparse matrix} of shape(n_samples, n_features)
        :return y : ndarray of shape (n_samples, )
        """
        check_is_fitted(self)
        X = _validate_data(X)

        n_estimators = len(self.estimators_)
        n_samples = X.shape[0]
        ans_pre = np.ones(n_samples)
        ans_random = np.ones(n_samples)
        # 计算后N/2个基分类器的结果，并于随机算法相比较
        for estimator, weight, i in zip(self.estimators_, self.beta_t, range(1, n_estimators + 1)):
            # print(i)
            if i >= n_estimators / 2:
                ans_pre *= pow(self.beta_t[i - 1], -estimator.predict(X))
                ans_random *= pow(self.beta_t[i - 1], -0.5)
        return (ans_pre > ans_random).astype(np.int)

    def _boost(self, iboost, X, y, sample_weight, partition):
        """ training in each iteration
        :param iboost : int
            The index of the current boost iteration
        :param X: {array-like, sparse matrix} of shape (n_samples, n_features)
        :param y: ndarray of shape (n_samples, )
        :param sample_weight:
        :param partition: int
            splitting the same-distribution domain and diff-distribution domain

        :return:
        sample_weight : array-like of shape (n_samples, ) or None
            if None then boosting has terminated early

        beta_t : float
            the parameter for each iteration
            if None then boosting has terminated early

        estimator_error : float
            the classification error for the current boost
            if None then boosting has terminated early
        """
        estimator = self._make_estimator()
        if np.fabs(sample_weight.sum() - 1.) > 1e-10:
            raise ValueError('weight is invalid')
        # print('training of %d round start' % (iboost))
        estimator.fit(X, y, sample_weight=sample_weight)
        n_samples = self.n_samples
        partition = int(partition)
        if partition > n_samples or partition <=0:
            raise ValueError('partition is invalid')
        y_predict = estimator.predict(X)
        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        incorrect = y_predict != y
        estimator_error = np.average(incorrect, weights=sample_weight, axis=0)
        weight_sum = np.sum(sample_weight[partition:])
        #estimator_error = np.average(incorrect[partition:], weights=sample_weight[partition:]/weight_sum, axis=0)
        #print(estimator_error, sum(sample_weight[partition:]/weight_sum))
        if estimator_error <= 1e-12:
            return sample_weight, 0, 0

        if estimator_error >= 0.5:
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError('BaseClassifier in TrAdaBoostClassifier '
                                 'ensemble is worse than random, ensemble '
                                 'can not be fit.')

        beta_t = estimator_error / (1. - estimator_error)
        incorrect = incorrect.astype(np.int)
        if not iboost == self.n_estimators - 1:
            sample_weight[:(partition - 1)] *= np.power(self.beta, incorrect[:(partition - 1)])
            sample_weight[partition:] *= np.power(beta_t, incorrect[partition:] * -1)

        return sample_weight, beta_t, estimator_error

    def _validate_estimator(self, default=None):
        if not isinstance(self.n_estimators, numbers.Integral):
            raise ValueError("n_estimators must be an integer, "
                             "got {0}.".format(type(self.n_estimators)))
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero, "
                             "got {0}.".format(self.n_estimators))

        if self.base_estimator is not None:
            self.base_estimator_ = self.base_estimator
        else:
            self.base_estimator_ = default

        if self.base_estimator_ is None:
            raise ValueError("base_estimator cannot be None")

    def _make_estimator(self, append=True):
        estimator = clone(self.base_estimator_)
        estimator.set_params(**{p: getattr(self, p) for p in self.estimator_params})
        if append:
            self.estimators_.append(estimator)
        return estimator


class DynamicTrAdaBoostClassifier(TrAdaBoost):
    """
    implementation of Dynamic-TrAdaBoost algorithm
    """
    def __init__(self, n_estimators=100, base_estimator=LinearSVC(), estimator_params=tuple()):
        super().__init__(
            n_estimators=n_estimators,
            base_estimator=base_estimator,
            estimator_params=tuple()
        )

    def _boost(self, iboost, X, y, sample_weight, partition):

        estimator = self._make_estimator()
        if np.fabs(sample_weight.sum() - 1.) > 1e-10:
            raise ValueError('weight is invalid')
        estimator.fit(X, y, sample_weight=sample_weight)
        n_samples = self.n_samples
        partition = int(partition)
        if partition > n_samples or partition <=0:
            raise ValueError('partition is invalid')
        y_predict = estimator.predict(X)
        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        incorrect = y_predict != y

        weight_sum = np.sum(sample_weight[partition:])
        #estimator_error = np.average(incorrect[partition:], weights=sample_weight[partition:], axis=0)
        estimator_error = np.average(incorrect, weights=sample_weight, axis=0)
        weight_sum = np.sum(sample_weight[partition:])
        # estimator_error = np.average(incorrect[partition:], weights=sample_weight[partition:]/weight_sum, axis=0)
        if estimator_error <= 1e-12:
            return sample_weight, 1., 0

        n_classes = self.n_classes_

        if estimator_error >= 0.5:
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError('BaseClassifier in TrAdaBoostClassifier '
                                 'ensemble is worse than random, ensemble '
                                 'can not be fit.')
            # return None, None, None

        beta_t = estimator_error / (1. - estimator_error)
        c_t = 2 * (1 - estimator_error)
        incorrect = incorrect.astype(np.int)
        if not iboost == self.n_estimators - 1:
            sample_weight[:(partition - 1)] *= np.power(self.beta, incorrect[:(partition - 1)])
            sample_weight[partition:] *= np.power(beta_t, incorrect[partition:] * -1) * c_t
        return sample_weight, beta_t, estimator_error


from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC

def _test_tra():
    for i in range(10):
        X_train, y_train, X_test, y_test, partition = fetch_transfer()
        clf_tradaboost = TrAdaBoost(n_estimators=15, base_estimator=LinearSVC())
        clf_dynamictra = DynamicTrAdaBoostClassifier(n_estimators=15, base_estimator=LinearSVC())
        clf_adaboost   = AdaBoostClassifier(n_estimators=15, base_estimator=LinearSVC(), algorithm='SAMME')

        clf_tradaboost.fit(X_train, y_train, partition)
        clf_adaboost.fit(X_train, y_train, partition)
        clf_dynamictra.fit(X_train, y_train, partition)

        error_tradaboost = np.mean(y_test == clf_tradaboost.predict(X_test))
        error_adaboost   = np.mean(y_test == clf_adaboost.predict(X_test))
        error_dynamictra = np.mean(y_test == clf_dynamictra.predict(X_test))
        print(error_adaboost, error_tradaboost, error_dynamictra)


if __name__ == '__main__':
    _test_tra()

