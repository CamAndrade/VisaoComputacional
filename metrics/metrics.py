import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


class Metrics:
    def __init__(self, y_true, y_pred):
        self.__y_true = y_true
        self.__y_pred = y_pred

    def matriz_confusao(self):
        return confusion_matrix(self.__y_true, self.__y_pred)

    def sensibilidade(self):
        return np.mean(recall_score(self.__y_true, self.__y_pred, average=None))

    def especificidade(self):
        cm = confusion_matrix(self.__y_true, self.__y_pred)
        fp = cm.sum(axis=0) - np.diag(cm)
        fn = cm.sum(axis=1) - np.diag(cm)
        vp = np.diag(cm)
        vn = cm.sum() - (fp + fn + vp)
        especificidade = vn/(vn+fp)
        return np.mean(especificidade)

    def precisao(self):
        return np.mean(precision_score(self.__y_true, self.__y_pred, average=None))

    def medida_f(self):
        return np.mean(f1_score(self.__y_true, self.__y_pred, average=None))

    def acuracia(self):
        return accuracy_score(self.__y_true, self.__y_pred)
