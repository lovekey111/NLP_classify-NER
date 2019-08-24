import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings("ignore")


def score(y_true, y_pre):
    a = accuracy_score(y_true, y_pre)
    p = precision_score(y_true, y_pre, average='macro')
    r = recall_score(y_true, y_pre, average='macro')
    f = f1_score(y_true, y_pre, average='macro')
    return a, p, r, f


def evaluate(y_true, y_pred, kind='train_label', verbose=2):
    a = accuracy_score(y_true, y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
    avg_p, avg_r, avg_f = np.mean(p), np.mean(r), np.mean(f)
    if verbose == 2:
        print('='*40)
        print(kind)
        print('accuray:%s\navg_precision:%s\navg_recall:%s\navg_f:%s' % (a, avg_p, avg_r, avg_f))
        print('='*40)
    return avg_p, avg_r, avg_f


