from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import backend as K

import os
import sys
import itertools
import numpy as np
import pandas as pd
import scipy.stats
import math


import matplotlib.pyplot as plt


class PerformanceMeasurement():
    '''
    ...
    '''
    def __init__(self) -> None:
        ...
        
    def plot_confusion_matrix(self, y_true: list, y_pred: list, class_lst:list , exp_path: str, title='Confusion Matrix', cmap=plt.cm.get_cmap('Blues'), normalize=False):
        # confustion_matrix
        con_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
        
        plt.imshow(con_mat, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        marks = np.arange(len(class_lst))
        nlabels = []
        for k in range(len(con_mat)):
            n = sum(con_mat[k])
            nlabel = f'{class_lst[k]}'
            nlabels.append(nlabel)
        plt.xticks(ticks=marks, labels=nlabels)
        plt.yticks(ticks=marks, labels=nlabels)

        thresh = con_mat.max() / 2.
        if normalize:
            for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
                plt.text(j, i, '{0}%'.format(con_mat[i, j] * 100 / n), horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
        else:
            for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
                plt.text(j, i, con_mat[i, j], horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(os.path.join('bin', 'exp', exp_path, 'con_mat.jpg'))   
        
        plt.close()
                  
    def recall(self, y_true: list, y_pred: list):
        """Recall for each labels.
            
        Args:
            y_ture: 1d array-like, or label indicator array.
            y_pred: 1d array-like, or label indicator array.

        Return:
            recall: 1d array.
        """
        true_positives = K.sum(K.round(K.clip(y_true.astype('float32') * y_pred.astype('float32'), 0, 1)), axis=0)
        possible_positives = K.sum(K.round(K.clip(y_true.astype('float32'), 0, 1)), axis=0)
        recall = true_positives / (possible_positives + K.epsilon())
        
        return recall

    def precision(self, y_true: list, y_pred: list):
        """"Precision.
            
        Args:
            y_ture: 1d array-like, or label indicator array.
            y_pred: 1d array-like, or label indicator array.

        Return:
            precision: 1d array.
        """
        true_positives = K.sum(K.round(K.clip(y_true.astype('float32') * y_pred.astype('float32'), 0, 1)), axis=0)
        predicted_positives = K.sum(K.round(K.clip(y_pred.astype('float32'), 0, 1)), axis=0)
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1(self, y_true: list, y_pred: list):
        """Macro F1.
        
        Args:
            y_ture: 1d array-like, or label indicator array.
            y_pred: 1d array-like, or label indicator array.
        
        Return:
            macro f1
        """
        p = self.precision(y_true.astype('float32'), y_pred.astype('float32'))
        r = self.recall(y_true.astype('float32'), y_pred.astype('float32'))
        return K.mean(2*((p*r)/(p+r+K.epsilon())), axis=None)

    def clopper_pearson(self, x, n, alpha=0.05):
        """
        reference: https://gist.github.com/sampsyo/c073c089bde311a6777313a4a7ac933e
        stimate the confidence interval for a sampled Bernoulli random variable.
        `x` is the number of successes and `n` is the number trials (x <= n). 
        `alpha` is the confidence level (i.e., the true probability is
        inside the confidence interval with probability 1-alpha). The
        function returns a `(low, high)` pair of numbers indicating the
        interval on the probability.
        """
        b = scipy.stats.beta.ppf
        lo = b(alpha / 2, x, n - x + 1)
        hi = b(1 - alpha / 2, x + 1, n - x)
        return 0.0 if math.isnan(lo) else lo, 1.0 if math.isnan(hi) else hi
    
    def calc_CL(self, indicator):
        """
        Confidence intervals for sensitivity, specificity and accuracy are "exact" Clopper-Pearson confidence intervals.
        """
        confidence_lower, confidence_upper = self.clopper_pearson(indicator['x'], indicator['n'])
        return float(indicator['value']), float(confidence_lower), float(confidence_upper)

    def calc_cm(self, y_pred: list, y_true: list):
        cm = confusion_matrix(y_true, y_pred)
        
        tp, fn, fp, tn = cm.ravel()
        
        acc = {'value':(tp+tn)/(tp+fp+fn+tn), 'x':tp+tn, 'n':tp+fp+fn+tn}
        sens = {'value':tp/(tp+fn), 'x':tp, 'n':tp+fn}
        spec = {'value':tn/(tn+fp), 'x':tn, 'n':tn+fp}

        print("sensitivity [95% CI] = {0[0]:0.4f} [{0[1]:0.4f} - {0[2]:0.4f}]".format(self.calc_CL(sens)))   
        print("specificity [95% CI] = {0[0]:0.4f} [{0[1]:0.4f} - {0[2]:0.4f}]".format(self.calc_CL(spec)))
        print("accuracy [95% CI]= {0[0]:0.4f} [{0[1]:0.4f} - {0[2]:0.4f}]".format(self.calc_CL(acc)))
        
        return  acc, sens, spec
