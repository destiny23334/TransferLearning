from TrAdaBoost.tradaboost import TrAdaBoost
from dataset.twenty_newsgroups import get_rec_sci, get_sci_talk, get_rec_talk, get_comp_talk

import numpy as np
from scipy.sparse import vstack
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import AdaBoostClassifier


class SimpleFilterTrAdaBoostClassifier(TrAdaBoost):
    def fit(self, X, y, partition, sample_weight=None):
        super().fit(X, y, partition, sample_weight)

    def _boost(self, iboost, X, y, sample_weight, partition):
        estimator = self._make_estimator()
        if np.fabs(sample_weight.sum() - 1.) > 1e-10:
            raise ValueError('weight is invalid')
        # print('training of %d round start' % iboost)
        estimator.fit(X, y, sample_weight=sample_weight)
        n_samples = self.n_samples
        partition = int(partition)
        if partition > n_samples or partition <= 0:
            raise ValueError('partition is invalid')

        y_predict = estimator.predict(X)
        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        incorrect = y_predict != y

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
            sample_weight[:(partition - 1)] *= np.power(self.beta, incorrect[:(partition - 1)])
            sample_weight[partition:] *= np.power(beta_t, incorrect[partition:] * -1)

        idx_filed = sample_weight < 1. / n_samples / self.n_estimators
        sample_weight[idx_filed] = 0

        return sample_weight, beta_t, estimator_error


class DynamicFilterTrAdaBoostClassifier(TrAdaBoost):
    def fit(self, X, y, partition, sample_weight=None):
        X, y, sample_weight = self._set_param(X, y, partition, sample_weight)
        self.error_times = np.zeros([self.n_samples])
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
            # print('In %d iteration, the training error is %.10f' % (iboost, estimator_error))
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
        # print('training of %d round start' % iboost)
        estimator.fit(X, y, sample_weight=sample_weight)
        n_samples = self.n_samples
        partition = int(partition)
        if partition > n_samples or partition <= 0:
            raise ValueError('partition is invalid')

        y_predict = estimator.predict(X)
        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        incorrect = y_predict != y
        self.error_times += incorrect


        estimator_error = np.average(incorrect, weights=sample_weight, axis=0)
        if estimator_error <= 1e-12:
            return sample_weight, 1., 0

        if estimator_error >= 0.5:
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError('BaseClassifier in TrAdaBoostClassifier '
                                 'ensemble is worse than random, ensemble '
                                 'can not be fit.')
            raise Warning('estimator error is great than 0.5')
            # return None, None, None

        beta_t = estimator_error / (1. - estimator_error)
        incorrect = incorrect.astype(np.int)
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight[:(partition - 1)] *= np.power(self.beta, incorrect[:(partition - 1)])
            sample_weight[partition:] *= np.power(beta_t, incorrect[partition:] * -1)

        for i in range(self.n_samples):
            if self.error_times[i] >= self.n_estimators / 2:
                sample_weight[i] = 0

        return sample_weight, beta_t, estimator_error


class NewFilterClassifier(TrAdaBoost):
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
        # print('training of %d round start' % iboost)
        estimator.fit(X, y, sample_weight=sample_weight)
        n_samples = self.n_samples
        partition = int(partition)
        if partition > n_samples or partition <= 0:
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

        idx_filed = sample_weight < 1. / n_samples / self.n_estimators
        sample_weight[idx_filed] = 0

        return sample_weight, beta_t, estimator_error

    def _check_error_time(self, partition):
        for i in range(partition):
            if self.right_times[i] >= self.n_estimators / 3:
                self.is_task[i] = True
                self.is_source[i] = False

def data_processing(data=1, step=5):
    if data == 1:
        X_source, X_task, y_source, y_task = get_rec_sci()
    elif data == 2:
        X_source, X_task, y_source, y_task = get_rec_talk()
    elif data == 3:
        X_source, X_task, y_source, y_task = get_sci_talk()
    else:
        X_source, X_task, y_source, y_task = get_comp_talk()
    X_source = X_source[::3]
    y_source = y_source[::3]
    X_task = X_task[::]
    y_task = y_task[::]
    select_task = np.random.random([X_task.shape[0]]) >= 0.77
    select_source = np.random.random([X_source.shape[0]]) >= 0
    X_task = X_task[select_task]
    y_task = y_task[select_task]
    X_source = X_source[select_source]
    y_source = y_source[select_source]
    print(X_task.shape[0] / X_source.shape[0])
    X_train_task, X_test, y_train_task, y_test = train_test_split(X_task, y_task)
    X_train = vstack([X_source, X_train_task])
    y_train = np.concatenate([y_source, y_train_task])
    partition = X_source.shape[0]
    return X_train, X_test, y_train, y_test, partition


def test():
    clf_tradaboost = TrAdaBoost(n_estimators=30, base_estimator=LinearSVC())
    clf_simplefilt = SimpleFilterTrAdaBoostClassifier(n_estimators=30, base_estimator=LinearSVC())
    clf_dynamicfil = DynamicFilterTrAdaBoostClassifier(n_estimators=30, base_estimator=LinearSVC())
    clf_adaboost   = AdaBoostClassifier(base_estimator=LinearSVC(), n_estimators=30, algorithm='SAMME')
    clf_hybricfilt = NewFilterClassifier(base_estimator=LinearSVC(), n_estimators=30)
    errors_tra = np.zeros([10])
    errors_sim = np.zeros([10])
    errors_dym = np.zeros([10])
    errors_ada = np.zeros([10])
    errors_hyb = np.zeros([10])
    iter = 10
    for i in range(iter):
        X_train, X_test, y_train, y_test, partition = data_processing(4, 1)
        clf_tradaboost.fit(X_train, y_train, partition)
        clf_simplefilt.fit(X_train, y_train, partition)
        clf_dynamicfil.fit(X_train, y_train, partition)
        clf_hybricfilt.fit(X_train, y_train, partition)
        clf_adaboost.fit(X_train, y_train)

        errors_tra[i] = np.mean(clf_tradaboost.predict(X_test) == y_test)
        errors_sim[i] = np.mean(clf_simplefilt.predict(X_test) == y_test)
        errors_dym[i] = np.mean(clf_dynamicfil.predict(X_test) == y_test)
        errors_ada[i] = np.mean(clf_adaboost.predict(X_test) == y_test)
        errors_hyb[i] = np.mean(clf_hybricfilt.predict(X_test) == y_test)

        print('sim：%.10lf\tdym: %.10lf\ttra: %.10lf\tada: %.10lf' % (errors_sim[i], errors_dym[i], errors_tra[i], errors_ada[i]))
    print('%.10lf\t%.10lf\t%.10lf\t%.10lf\t' % (errors_sim.sum()/iter, errors_dym.sum()/iter, errors_tra.sum()/iter, errors_ada.sum()/iter))


if __name__ == '__main__':
    test()



"""
ada: 0.844758	tra: 0.653226	dynamic: 0.895161	mydyna: 0.929435	sim: 0.909274	dyna: 0.766129	
ada: 0.760081	tra: 0.647177	dynamic: 0.921371	mydyna: 0.939516	sim: 0.925403	dyna: 0.743952	
ada: 0.850806	tra: 0.675403	dynamic: 0.913306	mydyna: 0.913306	sim: 0.895161	dyna: 0.733871	
ada: 0.733871	tra: 0.717742	dynamic: 0.923387	mydyna: 0.933468	sim: 0.953629	dyna: 0.800403	
ada: 0.818548	tra: 0.691532	dynamic: 0.891129	mydyna: 0.913306	sim: 0.905242	dyna: 0.715726	
ada: 0.820565	tra: 0.707661	dynamic: 0.925403	mydyna: 0.921371	sim: 0.937500	dyna: 0.768145	
ada: 0.870968	tra: 0.697581	dynamic: 0.891129	mydyna: 0.923387	sim: 0.784274	dyna: 0.733871	
ada: 0.814516	tra: 0.709677	dynamic: 0.921371	mydyna: 0.945565	sim: 0.943548	dyna: 0.766129	
ada: 0.844758	tra: 0.953629	dynamic: 0.931452	mydyna: 0.895161	sim: 0.721774	dyna: 0.856855	
ada: 0.907258	tra: 0.703629	dynamic: 0.909274	mydyna: 0.919355	sim: 0.899194	dyna: 0.711694	
"""