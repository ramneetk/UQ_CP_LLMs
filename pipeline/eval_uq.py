import functools
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import pdb

def area_under_accuracy_coverage_curve(u, a):
    # area under the rejection-VALUE curve, where VALUE could be accuracy, etc.
    df = pd.DataFrame({"u": u, 'a': a}).sort_values('u', ascending=True) 
    df['amean'] = df['a'].expanding().mean() # an expanding window that yields the value of an aggregation statistic with all the data available up to that point in time.
    return metrics.auc(np.linspace(0,1,len(df)), df['amean'])

def area_under_accuracy_rejection_curve(u, a):
    df = pd.DataFrame({"u": u, 'a': a}).sort_values('u', ascending=True) 
    cusum = df['a'].expanding().sum()
    total_acc = sum(list(a))
    rejected_samples_acc = np.array([total_acc]*len(u)) - cusum # will start with ~accuracy on original set, and keep on reducing as we will be considering accuracy on only highly uncertain samples
    rejected_samples_acc = rejected_samples_acc[:-1] # as the accuracy on a set with no samples (all samples rejected) does not make sense
    no_rejected_samples = np.array([len(a)-i for i in range(1,len(a)+1)])
    no_rejected_samples = no_rejected_samples[:-1] 
    assert len(rejected_samples_acc) == len(no_rejected_samples)
    avg_acc_rejected_samples = rejected_samples_acc/no_rejected_samples
    return metrics.auc(np.linspace(0,1,len(avg_acc_rejected_samples)), avg_acc_rejected_samples)

class Summarizer:
    def __init__(self, uqs, acc, lengths=None) -> None:
        self.uqs = uqs
        self.acc = acc
        self.lengths = lengths

        self.mem = {}

    @classmethod
    def compute_metric(cls, u, a, metric):
        if metric == 'auarc':
            return area_under_accuracy_coverage_curve(u, a) 
        elif metric == 'rej_acc':
            return area_under_accuracy_rejection_curve(u,a)
        elif metric == 'auroc':
            fpr, tpr, thresholds = metrics.roc_curve(a.astype(int), -u, pos_label=1)
            return metrics.auc(fpr, tpr)
        raise NotImplementedError()

    def _summarize_one_exp(self, uqs, acc:pd.Series, metric:str='auarc',
                           breakdown:np.ndarray=None, breakdown_by:List=None):
        # area under accuracy-coverage curve
        df = pd.DataFrame({'acc': acc, 'breakdown': breakdown})
        for uq_name, uq in uqs.items():
            df[uq_name] = uq
        df = df.dropna(subset=[_ for _ in df.columns if _ != 'breakdown'], how='any')
        def _make_one_ser(tdf):
            ret = pd.Series({ 'acc': tdf['acc'].mean(),
                        '_cnt': len(tdf),
                        })
            if ret['acc'] == 0 or ret['acc'] == 1:
                return ret
            ret['oracle'] =self.compute_metric(-tdf['acc'], tdf['acc'], metric)
            ret['blind'] = tdf['acc'].mean()
            for name, _ in uqs.items():
                ret[f'{name}'] = self.compute_metric(tdf[name], tdf['acc'], metric)
            return ret

        ret = {'main':  _make_one_ser(df)}
        if breakdown is not None:
            tres = {'overall': ret['main']}
            for i, min_len in enumerate(breakdown_by[:-1]):
                max_len = breakdown_by[i + 1]
                tdf = df[(df['breakdown'] > min_len) & (df['breakdown'] <= max_len)]
                tres[f'({min_len},{max_len}]'] = _make_one_ser(tdf)
            ret['breakdown'] = pd.DataFrame(tres)
        return ret
    def plot(self, curve='arc', iloc=None, **kwargs):
        assert curve in {'arc', 'roc'}
        _uqs = {k:v[0] for k,v in self.uqs.items()}
        _uqs.update({'oracle': -self.acc[0]})
        if curve == 'arc':
            plot_rejection_curve(_uqs, self.acc[0], **kwargs)
        elif curve == 'roc':
            plot_roc(_uqs, self.acc[0] if iloc is None else self.acc[1][iloc], **kwargs)

    def find_best_uq_name(self, metric:str='auarc', overall=False, use_conf=True): # metric will always be auarc as we are fixing best params according to this curve
        if metric == 'auarc' and overall:
            summ = self.summ_overall(metric)
        else:
            summ = self.summ_individual(metric, use_conf=use_conf)
            summ = sum(summ) / len(summ)
        summ = summ.sort_values()
        summ = summ.drop(['_cnt', 'acc'])
        assert summ['oracle'] == summ.max()
        summ = summ.drop(['oracle', 'blind'])
        return summ.idxmax()

    def _maximize_acc(self):
        acc = self.acc[1]
        # maximize accuracy, by choosing the prediction with the highest confidence
        ret = {}
        for uq_name, uq in self.uqs.items():
            assert uq[1].shape[1] <= acc.shape[1]
            uq = uq[1].reindex(acc.index).values#[:, :acc.shape[1]]
            idx = np.argmin(uq, axis=1)
            ret[uq_name] = pd.Series([row[_] for row, _ in zip(acc.values, idx)], acc.index)
        return pd.DataFrame(ret).dropna(how='any')

    @functools.lru_cache(maxsize=2)
    def combine_conf_uq(self, metric:str='auarc'):
        individual_summs = []
        for i in range(min(next(iter(self.uqs.values()))[1].shape[1], self.acc[1].shape[1])):
            uqs = {k:v[1][i].rank() + v[0].rank() for k,v in self.uqs.items()}
            individual_summs.append(self._summarize_one_exp(uqs, self.acc[1][i], metric)['main'])
        return individual_summs

    @functools.lru_cache(maxsize=1000)
    def summ_overall(self, metric:str='auarc'):
        return self._summarize_one_exp({k:v[0] for k,v in self.uqs.items()},
                                       self.acc[0], metric)['main']
    @functools.lru_cache(maxsize=1000)
    def summ_individual(self, metric:str='auarc', use_conf=True):
        individual_summs = []
        for i in range(min(next(iter(self.uqs.values()))[1].shape[1], self.acc[1].shape[1])):
            if use_conf:
                uqs = {k:v[1][i] for k,v in self.uqs.items()}
            else:
                uqs = {k:v[0] for k,v in self.uqs.items()}
            individual_summs.append(self._summarize_one_exp(uqs, self.acc[1][i], metric)['main'])
            if metric == 'auroc':
                individual_summs[-1].drop(['oracle', 'blind'], inplace=True)
        return individual_summs

    def summ_overall_by_length(self, metric:str='auarc'):
        return self._summarize_one_exp({k:v[0] for k,v in self.uqs.items()},
                                       self.acc[0], metric, self.lengths.mean(1), [0, 2, 4, 8])['breakdown']

def plot_rejection_curve(uqs, acc, name_map=None, methods=None, cutoff=1, **kwargs):
    if name_map is None:
        name_map = lambda x: x


    for uq_name, uq in uqs.items():
        if methods is not None and uq_name not in methods: continue
        df = pd.DataFrame({"u": uq, 'a': acc}).sort_values('u', ascending=True)
        df['amean'] = df['a'].expanding().mean()
        uq_name = uq_name.replace("generations|", "")
        x = np.linspace(0,1,len(df))
        mask = x <= cutoff
        plt.plot(x[mask], df['amean'].values[::-1][mask], '--' if uq_name == 'oracle' else '-',
                 label=name_map(uq_name))
    plt.ylim(acc.mean()-.05, 1.02)
    plt.hlines(acc.mean(), 0, 1, label='Base Accuracy', linestyles='dashed')

    plt.legend()

def plot_roc(uqs, acc, name_map=None, methods=None, cutoff=1, **kwargs):
    if name_map is None:
        name_map = lambda x: x
    for uq_name, uq in uqs.items():
        if methods is not None and uq_name not in methods: continue
        uq_name = uq_name.replace("generations|", "")
        fpr, tpr, thresholds = metrics.roc_curve(acc.astype(int), -uq, pos_label=1)
        plt.plot(fpr, tpr, '--' if uq_name == 'oracle' else '-', label=name_map(uq_name))
    plt.legend()
