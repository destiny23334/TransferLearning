from TrAdaBoost.tradaboost import TrAdaBoost, DynamicTrAdaBoostClassifier
from TrAdaBoost.filter import DynamicFilterTrAdaBoostClassifier, SimpleFilterTrAdaBoostClassifier, NewFilterClassifier
from dataset.twenty_newsgroups import fetch_transfer
import numpy as np
from scipy.sparse import vstack
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import AdaBoostClassifier


class MyTrAdaBoostClassifier(TrAdaBoost):
    def fit(self, X, y, partition, sample_weight=None):
        X, y, sample_weight = self._set_param(X, y, partition, sample_weight)
        self.right_times = np.zeros([self.n_samples])
        self.is_source = np.zeros([self.n_samples]).astype(np.bool)
        self.is_task = np.ones([self.n_samples]).astype(np.bool)
        for i in range(partition):
            self.is_task[i] = False
            self.is_source[i] = True

        for iboost in range(self.n_estimators):
            sample_weight, beta_t, estimator_error = self._boost(
                iboost,
                X, y,
                sample_weight,
                partition
            )
            # print('After %d round, the training error is %.16lf' % (iboost, estimator_error))
            if sample_weight is None:
                raise ValueError('sample_weight is None!')

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
        self.right_times[np.bitwise_not(incorrect)] += 1
        self._check_error_time(partition)
        estimator_error = np.average(incorrect, weights=sample_weight, axis=0)
        if estimator_error <= 1e-12:
            return sample_weight, 1., 0

        if estimator_error >= 0.5:
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError('BaseClassifier in TrAdaBoostClassifier '
                                 'ensemble is worse than random, ensemble '
                                 'can not be fit.')
            # return None, None, None

        beta_t = estimator_error / (1. - estimator_error)
        incorrect = incorrect.astype(np.int)
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight[self.is_source] *= np.power(self.beta, incorrect[self.is_source])
            sample_weight[self.is_task] *= np.power(beta_t, incorrect[self.is_task] * -1)
        return sample_weight, beta_t, estimator_error

    def _check_error_time(self, partition):
        for i in range(partition):
            if self.right_times[i] >= self.n_estimators / 3:
                self.is_task[i] = True
                self.is_source[i] = False


def test():
    clf_tradaboost = TrAdaBoost(n_estimators=20, base_estimator=LinearSVC())
    clf_mytradaboo = MyTrAdaBoostClassifier(n_estimators=20, base_estimator=LinearSVC())
    clf_adaboost   = AdaBoostClassifier(base_estimator=LinearSVC(), n_estimators=20, algorithm='SAMME')
    clf_dynamictra = DynamicTrAdaBoostClassifier(n_estimators=20, base_estimator=LinearSVC())
    clf_simplefile = SimpleFilterTrAdaBoostClassifier(n_estimators=20, base_estimator=LinearSVC())
    clf_dynamicfil = DynamicFilterTrAdaBoostClassifier(n_estimators=20, base_estimator=LinearSVC())
    clf_hybricfilt = NewFilterClassifier(n_estimators=20, base_estimator=LinearSVC())

    errors_my  = 0
    errors_ada = 0
    errors_tra = 0
    errors_dyf = 0
    errors_sfi = 0
    errors_dyn = 0
    errors_hyb = 0
    iter = 10
    for i in range(iter):
        X_train, y_train, X_test, y_test, partition = fetch_transfer(dataset='svt', all=2000, ratio=0.05)
        clf_tradaboost.fit(X_train, y_train, partition)
        clf_mytradaboo.fit(X_train, y_train, partition)
        clf_adaboost.fit(X_train, y_train)
        clf_dynamictra.fit(X_train, y_train, partition)
        clf_simplefile.fit(X_train, y_train, partition)
        clf_dynamicfil.fit(X_train, y_train, partition)
        clf_hybricfilt.fit(X_train, y_train, partition)

        errors_my += np.mean(clf_mytradaboo.predict(X_test) == y_test)
        errors_ada += np.mean(clf_adaboost.predict(X_test) == y_test)
        errors_tra += np.mean(clf_tradaboost.predict(X_test) == y_test)
        errors_dyf += np.mean(clf_dynamicfil.predict(X_test) == y_test)
        errors_sfi += np.mean(clf_simplefile.predict(X_test) == y_test)
        errors_dyn += np.mean(clf_dynamictra.predict(X_test) == y_test)
        errors_hyb += np.mean(clf_hybricfilt.predict(X_test) == y_test)

    print("ada: %.6lf\ttra: %.6lf\tdynamic: %.6lf\tmydyna: %.6lf\tsim: %.6lf\tdynaf: %.6lf\thyb: %.6lf" %
              (errors_ada / iter, errors_tra / iter, errors_dyn / iter, errors_my / iter, errors_sfi / iter, errors_dyf / iter, errors_hyb / iter))



if __name__ == '__main__':
    test()