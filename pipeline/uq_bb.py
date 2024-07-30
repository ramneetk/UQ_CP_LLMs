import functools
import os
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
import persist_to_disk as ptd
import torch
import tqdm
from scipy.special import softmax

import sys
import _settings
import dataeval.load as dload
import pipeline.clustering as pc
import pipeline.eval_uq as eval_uq

from sement.seq_clustering import alpha_clustering

import pdb

CONTRADICT, NEUTRAL, AGREE = 0, 1, 2
DEVICE = 'cuda:1'

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama-13b-hf')
parser.add_argument('--dataset', type=str, default='coqa') # coqa/triviaqa
parser.add_argument('--cal_size', type=int, default=1000) # 2000 for triviaqa
parser.add_argument('--acc_name', type=str, default='deberta_entailment') # 'deberta_entailment/rougeL/gpt'
parser.add_argument('--metric', type=str, default='auarc') # auarc/auroc/rej_acc (rej_acc is aurac in the paper)
parser.add_argument('--output_pred_sets', type=bool, default=False)
parser.add_argument('--norm_sets', type=bool, default=True)
parser.add_argument('--num_gens', type=int, default=20)

args = parser.parse_args()

def _clean_path(path):
    base_dir = os.path.normpath(_settings.GENERATION_FOLDER)
    path = os.path.normpath(path)
    assert path.startswith(base_dir)
    return path[len(base_dir):]

def _compute_lexical_sim(sample, num_gens=3):
    locs = sample['mapping'][:num_gens]
    sim_mat = sample['sim_mat']
    ret = 0.
    denom = 0
    for i in locs:
        for j in locs:
            if i != j:
                ret += sim_mat[i,j]
                denom += 1
    if denom == 0: return 1.
    return ret/denom

def recover_sim_mat_new(sim):
    sim_mat = sim['sim_mat'].clone()
    sim_mat[torch.arange(sim_mat.shape[0]), torch.arange(sim_mat.shape[0]), :] = torch.tensor([-torch.inf, -torch.inf, 100])
    mapping = sim['mapping']
    # a len(ans) x len(ans) x 3 tensor
    ret = torch.zeros((len(mapping), len(mapping), 3))
    for i, ans_i in enumerate(mapping):
        for j, ans_j in enumerate(mapping):
            ret[i,j] = torch.tensor(sim_mat[mapping[i], mapping[j]])
    return None, ret

def _create_semantic_sets(sample):
    # https://github.com/lorenzkuhn/semantic_uncertainty
    generated_texts = sample['mapping']
    sim_mat = sample['sim_mat'].argmax(axis=-1)
    # unique_ans is also a list of integers.
    unique_generated_texts = sorted(list(set(generated_texts)))
    semantic_set_ids = {ans: i for i, ans in enumerate(unique_generated_texts)} # one id for each exact-match answer
    for i, ans_i in enumerate(unique_generated_texts):
        for j, ans_j in enumerate(unique_generated_texts[i+1:], i+1):
            if min(sim_mat[ans_i,ans_j], sim_mat[ans_j,ans_i]) > CONTRADICT:
                semantic_set_ids[ans_j] = semantic_set_ids[ans_i]

    list_of_semantic_set_ids = [semantic_set_ids[x] for x in generated_texts] # this step is for allocating same cluster to the same/repeated answer
    # map according to the order of appearance
    _map = defaultdict(int)
    ret = []
    for i, ans in enumerate(list_of_semantic_set_ids):
        if ans not in _map:
            _map[ans] = len(_map)
        ret.append(_map[ans])
    return ret

# our approach for creating semantic sets using alpha clustering
def _create_alpha_semantic_sets(sample):
    generated_texts = sample['mapping']
    cls_logits = sample['sim_mat'] 

    cls_probs = torch.softmax(cls_logits, dim=-1)
    entailment_probs = cls_probs[0:len(cls_probs), 0:len(cls_probs), -1]
    entailment_probs = torch.max(entailment_probs, entailment_probs.T) #OR

    clusters_alpha = alpha_clustering(entailment_probs, alpha=0.5)
    num_elements = len(cls_probs)
    cluster_assignments = torch.zeros(num_elements)
    for i, cluster in enumerate(clusters_alpha):
        for element in cluster:
            cluster_assignments[element] = i

    list_of_semantic_set_ids = [cluster_assignments[x] for x in generated_texts] 
    
    _map = defaultdict(int)
    ret = []
    for i, ans in enumerate(list_of_semantic_set_ids):
        if ans.item() not in _map:
            _map[ans.item()] = len(_map)
        ret.append(_map[ans.item()])

    return ret

# whitebox methods
def _logmeanexp(x, dim, ignore_negative_inf=False):
    if ignore_negative_inf:
        cnt = (x > -torch.inf).sum(dim)
    else:
        cnt = torch.tensor(x.shape[dim])
    return torch.logsumexp(x, dim=dim) - torch.log(cnt)

def _hard_semantic_entropies(neg_log_likelihoods, semantic_set_ids, **kwargs):
    num_samples, num_gens = neg_log_likelihoods.shape

    log_likelihoods = -neg_log_likelihoods
    # initilaize to -inf for all possible semantic ids
    max_num_semantic_ids = semantic_set_ids.max().item() + 1 + 1
    aggregated_likelihoods = torch.log(torch.zeros((num_samples, max_num_semantic_ids)))
    for semantic_set_id in torch.unique(semantic_set_ids):
        temp = torch.where(semantic_set_ids == semantic_set_id, log_likelihoods, -torch.inf)
        aggregated_likelihoods[:, semantic_set_id] = torch.logsumexp(temp, 1)
    return -_logmeanexp(aggregated_likelihoods, dim=1, ignore_negative_inf=True)

def _hard_semantic_alpha_cluster_entropies(neg_log_likelihoods, semantic_set_ids, **kwargs):
    num_samples, num_gens = neg_log_likelihoods.shape

    log_likelihoods = -neg_log_likelihoods
    # initilaize to -inf for all possible semantic ids
    max_num_semantic_ids = semantic_set_ids.max().item() + 1 + 1
    aggregated_likelihoods = torch.log(torch.zeros((num_samples, max_num_semantic_ids)))
    for semantic_set_id in torch.unique(semantic_set_ids):
        temp = torch.where(semantic_set_ids == semantic_set_id, log_likelihoods, -torch.inf)
        aggregated_likelihoods[:, semantic_set_id] = torch.logsumexp(temp, 1)
    return -1*aggregated_likelihoods 

class UQ_computer:
    def __init__(self, path, clean=False,
                 split=None, cal_size=None, seed=None) -> None:

        print("Seed for UQ_computer: ", seed)
        if isinstance(path, str):
            self.path = path
            self.key = (_clean_path(path), clean)
            self.generations = dload.read_cleaned_outputs_new(path)
        else:
            assert isinstance(path, list)
            self.generations, self.path = path, None
            self.key = (None, clean)

        self.keep_indices = None
        if split is not None:
            assert split in ['val', 'test'] and cal_size is not None and seed is not None
            self.key = (_clean_path(path) if self.path is not None else None, clean, split, cal_size, seed)
    
            if cal_size == 2000: # true in case of triviaqa for llama-13b as conformal modeling paper
                triviaQA_generations_path = os.path.join(_settings.GENERATION_FOLDER, 'llama-13b-hf_triviaqa_0')
                import pickle
                with open(f'{triviaQA_generations_path}/cal_test_info/split_indices.pkl', 'rb') as f: 
                    trivia_split_indices = pickle.load(f)
                self.keep_indices = trivia_split_indices['val']
                if split == 'test':
                    self.keep_indices = trivia_split_indices['test']
            else:
                print('ValueError', len(self.generations), cal_size)
                self.keep_indices = np.random.RandomState(seed).choice(len(self.generations), cal_size, replace=False)
                if split == 'test':
                    self.keep_indices = set(np.arange(len(self.generations))) - set(self.keep_indices)
        
        self.generations = [self.generations[_] for _ in self.keep_indices]
        self.ids = [_['id'] for _ in self.generations]

        self.mem = defaultdict(dict)
        self._summ_mem = {}

    @functools.cached_property
    def similarities(self):
        if self.key[0] is None:
            text_key = 'text_cleaned' if self.key[1] else 'text'
            import models.nli as sc
            nli_model = sc.ClassifyWrapper(device=DEVICE)
            sims = [nli_model.create_sim_mat_batched(_['question'], _['generations'][text_key])
                    for _ in tqdm.tqdm(self.generations, desc="computing similarities")]
        else:
            sims = dload.read_semantic_similarities_new(self.path, clean=self.key[1], debug=False)
            sims = [sims[_] for _ in self.ids]
        return sims

    @functools.cached_property
    def rougeLsims(self):
        if self.key[0] is None:
            import dataeval.load_worker as lw
            ret = lw._get_lexical_similarities(self.generations, self.key[1])
        else:
            ret = dload.read_lexical_sim(self.path, clean=self.key[1], debug=False, read_only=True)
            ret = [ret[_] for _ in self.ids]
        return ret

    @functools.cached_property
    def likelihoods(self):
        assert self.path is not None, "likelihoods are not available for black-box data"
        print("load likelihoods")
        likelihoods = dload.read_loglikelihoods_and_more_new(self.path, device=DEVICE, clean=self.key[1], debug=False)
        if likelihoods is not None:
            likelihoods = {_['id']: _ for _ in likelihoods}
            likelihoods = [likelihoods[_] for _ in self.ids]
            likelihoods = self.batchify(likelihoods)
        return likelihoods

    @functools.cached_property
    def self_eval(self):
        assert self.path is not None, "self evaluatinn (P(true)) is not available for black-box data"
        print("load self eval")
        self_eval = dload.read_self_eval(self.path, None, self.key[1])
        if self_eval is not None:
            self_eval = {_['id']: _ for _ in self_eval}
            self_eval = [self_eval[_] for _ in self.ids]
        return self_eval

    @classmethod
    def batchify(cls, likelihoods):
        result_dict = defaultdict(list)
        to_stack = set()
        for sample in tqdm.tqdm(likelihoods, 'reading'):
            result_dict['id'].append(sample['id'])
            for pref, sub_dict in sample.items():
                if pref == 'id':
                    continue
                for key, val in sub_dict.items():
                    if isinstance(val, list) and (isinstance(val[0], int) or isinstance(val[0], float)):
                        val = torch.tensor(val)
                        to_stack.add(pref + '|' + key)
                    result_dict[pref + '|' + key].append(val)
        result_dict = dict(result_dict)
        for key, val in result_dict.items():
            if key in to_stack:
                result_dict[key] = torch.stack(val)
            else:
                if isinstance(val, list) and (isinstance(val[0], int) or isinstance(val[0], float)):
                    val = torch.tensor(val)
                result_dict[key] = val
        return result_dict

    def _get_recovered_logits(self, num_gens:int):
        if '_get_recovered_logits' not in self.mem:
            self.mem['_get_recovered_logits'] = [recover_sim_mat_new(_)[1] for _ in self.similarities]
        return [_[:num_gens, :num_gens] for _ in self.mem['_get_recovered_logits']]

    def _get_jaccard_matrix(self, num_gens:int):
        def jaccard_one(all_answers):
            all_answers = [set(ans.lower().split()) for ans in all_answers]
            ret = np.eye(len(all_answers))
            for i, ans_i in enumerate(all_answers):
                for j, ans_j in enumerate(all_answers[i+1:], i+1):
                    ret[i,j] = ret[j,i] = len(ans_i.intersection(ans_j)) / max(len(ans_i.union(ans_j)),1)
            return ret
        if '_get_jaccard_matrix' not in self.mem:
            text_key = 'text_cleaned' if self.key[1] else 'text'
            self.mem['_get_jaccard_matrix'] = [jaccard_one(_['generations'][text_key]) for _ in self.generations]
        return [_[:num_gens, :num_gens] for _ in self.mem['_get_jaccard_matrix']]

    def _get_semantic_ids(self, num_gens): # returns list (of dataset len) of list (of length equal to num_gen) of cluster ids
        if num_gens not in self.mem['_get_gal_semantic_ids']:
            # We must filter sims first before passing to _create_gal_semantic_ids
            sims = [{
                'mapping': _['mapping'][:num_gens], # mapping is unique answer ID, so shape = num_gens
                'sim_mat': _['sim_mat'], # sim_mat is similarity between unique answers, so if no. of unique answers = 15, then shape will be 15X15X3
                } for _ in self.similarities]
            self.mem['_get_gal_semantic_ids'][num_gens] = [_create_semantic_sets(_) for _ in sims]
        return self.mem['_get_gal_semantic_ids'][num_gens]

    def _get_semantic_ids_using_alpha_clustering(self, num_gens): # returns list (of dataset len) of list (of length equal to num_gen) of cluster ids
        sims = [{
                'mapping': _['mapping'][:num_gens], # mapping is unique answer ID, so shape = num_gens
                'sim_mat': _['sim_mat'], # sim_mat is similarity between unique answers, so if no. of unique answers = 15, then shape will be 15X15X3
                } for _ in self.similarities]
        return [_create_alpha_semantic_sets(_) for _ in sims]
        
    # for getting eigenvectors
    def _get_spectral_projected(self, num_gens:int, eigv_threshold:float, affinity_mode:str, temperature:float):
        key = (num_gens,eigv_threshold, temperature,affinity_mode)
        if key not in self.mem['_get_spectral_projected']:
            clusterer = pc.SpetralClusteringFromLogits(affinity_mode=affinity_mode, eigv_threshold=eigv_threshold, cluster=False, temperature=temperature)
            sim_mats = getattr(self, '_get_jaccard_matrix' if affinity_mode == 'jaccard' else '_get_recovered_logits')(num_gens)
            self.mem['_get_spectral_projected'][key] = [clusterer.orig_proj(logits=_) for _ in tqdm.tqdm(sim_mats, desc='projecting')] 

        return self.mem['_get_spectral_projected'][key]

    def get_length(self, num_gens:int):
        text_key = 'text_cleaned' if self.key[1] else 'text'
        lengths = [[len(set(_.split())) for _ in sample['generations'][text_key][:num_gens]] for sample in self.generations]
        lengths = np.asarray(lengths)
        return lengths.mean(1), lengths

    def get_spectral_eigv(self, num_gens:int, affinity_mode:str, temperature:float, adjust:bool) -> List:
        clusterer = pc.SpetralClusteringFromLogits(affinity_mode=affinity_mode, eigv_threshold=None,
                                                   cluster=False, temperature=temperature)
        sim_mats = getattr(self, '_get_jaccard_matrix' if affinity_mode == 'jaccard' else '_get_recovered_logits')(num_gens)
        return [clusterer.get_eigvs(logits=_).clip(0 if adjust else -1).sum() for _ in tqdm.tqdm(sim_mats)]

    def get_numsets(self, num_gens):
        return [len(set(_)) for _ in self._get_semantic_ids(num_gens)]

    def get_lexicalsim(self, num_gens = None):
        return [-_compute_lexical_sim(_, num_gens) for _ in self.rougeLsims]

    def get_eccentricity(self, num_gens:int, eigv_threshold:float, affinity_mode:str, temperature:float) -> List:
        projected = self._get_spectral_projected(num_gens, eigv_threshold, affinity_mode, temperature)
        ds = np.asarray([np.linalg.norm(x -x.mean(0)[None, :],2,axis=1) for x in projected])
        return np.linalg.norm(ds, 2,1), ds # first return is for overall, second for each response, they are just using overall for reporting AUARC

    def get_degreeuq(self, num_gens:int, affinity_mode:str, temperature:float):
        sim_mats = getattr(self, '_get_jaccard_matrix' if affinity_mode == 'jaccard' else '_get_recovered_logits')(num_gens)
        Ws = [pc.get_affinity_mat(_, affinity_mode, temperature, symmetric=False) for _ in sim_mats]
        ret = np.asarray([np.sum(1-_, axis=1) for _ in Ws])
        return ret.mean(1), ret

    def get_alpha_semantic_cluster_ids_and_nlls(self, num_gens:int, normalize:bool):
        if self.likelihoods is None:
            return None
        semantic_set_ids = self._get_semantic_ids_using_alpha_clustering(num_gens)
        nlls = self.likelihoods['generations|neg_log_likelihood'][:, :num_gens]
        if normalize:
            nlls = nlls / self.likelihoods['generations|length'][:, :num_gens]
        clusters_nlls = _hard_semantic_alpha_cluster_entropies(nlls, torch.tensor(semantic_set_ids))
        return torch.tensor(semantic_set_ids), clusters_nlls

    # semantic entropy using our alpha clustering approach
    def get_alpha_semantic_entropy(self, num_gens:int, normalize:bool):
        if self.likelihoods is None:
            return None
        semantic_set_ids = self._get_semantic_ids_using_alpha_clustering(num_gens)
        nlls = self.likelihoods['generations|neg_log_likelihood'][:, :num_gens]
        if normalize:
            nlls = nlls / self.likelihoods['generations|length'][:, :num_gens]
        return _hard_semantic_entropies(nlls, torch.tensor(semantic_set_ids))
    
    # whitebox methods
    def get_semantic_entropy(self, num_gens:int, normalize:bool):
        if self.likelihoods is None:
            return None
        semantic_set_ids = self._get_semantic_ids(num_gens)
        nlls = self.likelihoods['generations|neg_log_likelihood'][:, :num_gens]
        if normalize:
            nlls = nlls / self.likelihoods['generations|length'][:, :num_gens]
        return _hard_semantic_entropies(nlls, torch.tensor(semantic_set_ids))

    def get_selfprob(self, num_gens: int):
        if self.self_eval is None:
            return None
        if 'get_selfprob' not in self.mem:
            self.mem['get_selfprob'] = np.stack(
                [softmax(_['logits'].values,1)[:, 0] for _ in self.self_eval] # p(true)
            )
        ret = 1 - self.mem['get_selfprob'][:, :num_gens] # 1 - p(true) as uncertainty
        return ret.mean(1), ret


# @ptd.persistf(expand_dict_kwargs=['metric_kwargs'], skip_kwargs=['self'], lock_granularity='call', switch_kwarg='cache', groupby=['uq_name'])
def _compute_uq_cached(self:UQ_computer, key, uq_name, num_gens=20, metric_kwargs=None, **kwargs):
    if metric_kwargs is None:
        metric_kwargs = {}
    if 'jaccard' in uq_name:
        assert 'temperature' not in metric_kwargs, 'jaccard does not use temperature'
        metric_kwargs['temperature'] = None

    # no "individual" metrics
    if uq_name == 'generations|numsets':
        return self.get_numsets(num_gens)
    if 'lexical_sim' == uq_name:
        return self.get_lexicalsim(num_gens)

    if uq_name.startswith('generations|spectral_eigv'):
        affinity_mode = 'disagreement_w' if len(uq_name.split("|")) == 2 else uq_name.split("|")[2]
        return self.get_spectral_eigv(num_gens, affinity_mode, temperature=metric_kwargs['temperature'], adjust='spectral_eigv_clip' in uq_name)

    # both overall and individual metrics
    if uq_name.startswith('generations|eccentricity'):
        affinity_mode = 'disagreement_w' if len(uq_name.split("|")) == 2 else uq_name.split("|")[2]
        return self.get_eccentricity(num_gens, metric_kwargs['eigv_threshold'], affinity_mode, temperature=metric_kwargs['temperature'])
    if uq_name.startswith('generations|degree'):
        affinity_mode = uq_name.split("|")[2]
        return self.get_degreeuq(num_gens, affinity_mode, temperature=metric_kwargs['temperature'])
    
    # whitebox
    if uq_name.startswith("alphaSemanticEntropy"):
        return self.get_alpha_semantic_entropy(num_gens, normalize=uq_name.split("|")[1] == 'norm')
    if uq_name.startswith("semanticEntropy"):
        return self.get_semantic_entropy(num_gens, normalize=uq_name.split("|")[1] == 'norm')
    if uq_name == 'self_prob':
        return self.get_selfprob(num_gens)

    raise ValueError(f'Unknown metric {uq_name}')

class UQ_summ(UQ_computer): # UQ_computer is the base class of UQ_summ
    _uq_measures = [
        'generations|numsets',
        'lexical_sim',

        'generations|spectral_eigv_clip|disagreement_w',
        'generations|eccentricity|disagreement_w',
        'generations|degree|disagreement_w',

        'generations|spectral_eigv_clip|agreement_w',
        'generations|eccentricity|agreement_w',
        'generations|degree|agreement_w',

        'generations|spectral_eigv_clip|jaccard',
        'generations|eccentricity|jaccard',
        'generations|degree|jaccard',

        # whitebox methods
        'semanticEntropy|unnorm',
        'self_prob',
        #ours
        'alphaSemanticEntropy|unnorm',
        'alphaSemanticEntropy|norm',
    ]

    tunable_hyperparams = {
        'generations|spectral_eigv_clip|disagreement_w': ['temperature'],
        'generations|spectral_eigv_clip|agreement_w': ['temperature'],

        'generations|eccentricity|disagreement_w': ['eigv_threshold', 'temperature'],
        'generations|eccentricity|agreement_w': ['eigv_threshold', 'temperature'],
        'generations|eccentricity|jaccard': ['eigv_threshold'],

        'generations|degree|disagreement_w': ['temperature'],
        'generations|degree|agreement_w': ['temperature'],

        'generations|density|agreement_w': ['eigv_threshold', 'temperature'],
        'generations|density|disagreement_w': ['eigv_threshold', 'temperature'],
        'generations|density|jaccard': ['eigv_threshold'],

        'generations|fiedler|agreement_w': ['eigv_threshold', 'temperature'],
        'generations|fiedler|disagreement_w': ['eigv_threshold', 'temperature'],
        'generations|fiedler|jaccard': ['eigv_threshold'],
    }

    default_params = {'eigv_threshold': 0.9, 'temperature': 3.}
    whitebox_uqs = ['alphaSemanticEntropy|unnorm', 'alphaSemanticEntropy|norm', 'semanticEntropy|unnorm', 'semanticEntropy|norm', 'self_prob', 'self_prob_nll']
    def __init__(self, path, clean=False,
                 split=None, cal_size=None, seed=None, 
                 gpteval_examples = None) -> None:
        super().__init__(path, clean, split, cal_size, seed)
        self.gpteval_examples = gpteval_examples
        print("Seed for UQ_summ: ", seed)

    @functools.cached_property
    def uq_measures(self):
        uq_measures = self._uq_measures
        # if self.path is None or 'gpt-3.5' in self.path:
        if self.path is None or 'gpt-4' in self.path:
            uq_measures = [_ for _ in uq_measures if _ not in self.whitebox_uqs]
        return uq_measures


    @functools.cached_property
    def rouges(self):
        clean = self.key[1]
        if self.path is None:
            import dataeval.load_worker as lw
            text_key = 'text_cleaned' if clean else 'text'
            rouges = [lw._get_rouge_sample(_, text_key) for _ in self.generations]
        else:
            rouges = dload.read_rouges_new(self.path, clean=clean, debug=False) # here
            rouges = {_['id']: _ for _ in rouges}
            rouges = [rouges[_] for _ in self.ids]
        return rouges

    @functools.cached_property
    def deberta_scores(self):
        clean = self.key[1]
        if self.path is None:
            import dataeval.load_worker as lw
            text_key = 'text_cleaned' if clean else 'text'
            deberta_scores = [lw._get_deberta_scores_sample(_, text_key) for _ in self.generations] # get score for each sample
        else:
            deberta_scores = dload.read_deberta_scores(self.path, clean=clean, debug=False) # get scores for all samples with path of all samples specified in self.path
            deberta_scores = {_['id']: _ for _ in deberta_scores}
            deberta_scores = [deberta_scores[_] for _ in self.ids]
        return deberta_scores
    
    @functools.cached_property
    def gpt_eval(self):
        clean = self.key[1]
        if self.path is None:
            text_key = 'text_cleaned' if clean else 'text'
            ret = {}
            for ith in range(len(self.generations[0]['generations']['text'])):
                import dataeval.load_worker as lw
                gpt_eval = [lw._get_gpt_eval_sample(_, text_key, ith, few_shots=self.gpteval_examples) for _ in self.generations]
                gpt_eval = {k: {"id": k, "response": v.split(".")[0].split()[0]} for k, v in zip(self.ids, gpt_eval)}
                ret[ith] = [gpt_eval[_id] for _id in self.ids]
        else:
            ret = {}
            for ith in range(len(self.generations[0]['generations']['text'])):
                try:
                    gpt_eval = dload.read_gpt_eval(self.path, clean=clean, debug=False, read_only=True, ith=ith)
                    ##### HARDCODING THE ID for TRIVIAQA on MISTRAL WHERE GPT COULD NOT GENERATE CORRECT EVALUATION (FORMAT): '.com is MEERKOVO.\nRating: 100.', expected was '100. .com is MEERKOVO...' ########
                    id_set = []
                    for idx, id in enumerate(self.ids):
                        if id == 'sfq_11335':
                            continue
                        else:
                            id_set.append(id)
                    
                    ret[ith] = [gpt_eval[id] for id in id_set] # originally: ret[ith] = [gpt_eval[_id] for _id in self.ids]
                except Exception as err:
                    break
        return ret

    def get_uq(self, name='', num_gens=20, cache=None, **kwargs):
        if cache is None:
            cache = ptd.NOCACHE if name in {'generations|eigent', 'debug'} else ptd.CACHE
        if self.path is None:
            cache = ptd.NOCACHE
        individual_uq = None
        overall_uq = _compute_uq_cached(self, self.key, name, num_gens=num_gens, metric_kwargs=kwargs, cache=cache)
        if overall_uq is None:
            return None, None
        if len(overall_uq) == 2:
            overall_uq, individual_uq = overall_uq
            assert len(overall_uq) == len(individual_uq) == len(self.ids)

        # use overall for individual if not provided
        if individual_uq is None:
            individual_uq = np.tile(overall_uq, (num_gens, 1)).T
        assert individual_uq.shape[1] == num_gens
        return pd.Series(np.asanyarray(overall_uq), self.ids), pd.DataFrame(np.asarray(individual_uq), index=self.ids)

    def get_acc(self, acc_name='generations|rougeL|acc'):
        # returns the expected accuracy (over all 20 generations) as well as individual accuracy
        pref, name, suffix = acc_name.split("|")
        assert pref == 'generations' and name in {'rougeL', 'gpt', 'deberta_entailment'}, f'Unknown type {acc_name}'
        if name == 'rougeL':
            if pref == 'generations':
                scores = [[_[name] for _ in sample['generations']] for sample in self.rouges] # self.rouges is taking a lot of time: this will go to def rouges(self) for the first time
                score_df = pd.DataFrame(scores, index=self.ids)
            else:
                raise NotImplementedError()
        elif name == 'deberta_entailment':
            if pref == 'generations':
                scores = [[_[name] for _ in sample['generations']] for sample in self.deberta_scores]
                score_df = pd.DataFrame(scores, index=self.ids)
            else:
                raise NotImplementedError()
        elif name == 'gpt':
            score_df = pd.DataFrame(np.zeros((len(self.ids), len(self.gpt_eval))), index=self.ids)
            for ith, vals in self.gpt_eval.items():
                for j, val in enumerate(vals):
                    _id = val['id']
                    try:
                        val = int(val['response'])
                        assert 0 <= val <= 100
                    except:
                        val = np.NaN
                    score_df.loc[_id, ith] = val
            score_df /= 100.
        indiv_acc = score_df.reindex(self.ids)
        if suffix == 'acc':
            if name == 'rougeL':
                indiv_acc = (indiv_acc > 0.3).astype(float)
            elif name == 'gpt':
                indiv_acc = (indiv_acc > 0.7).astype(float)
        return indiv_acc.mean(1), indiv_acc

    def get_cal_set_nlls(self, semantic_set_ids, cluster_wise_nlls, individual_accuracy):
        cal_set_nlls = []
        
        for i in range(len(semantic_set_ids)): # for each datapoint in the calibration dataset
            response_cluster_ids = semantic_set_ids[i] # semantic_set_ids[i] is a list of 20 cluster ids for each (20) generated response
            no_of_clusters = len(torch.unique(response_cluster_ids))
            for cluster_num in range(no_of_clusters): # for each cluster 
                cluster_response_indices = (semantic_set_ids[i] == cluster_num).nonzero(as_tuple=True)[0] # get indices of all the responses (or generations) in this cluster
                for j, response_id in enumerate(cluster_response_indices):
                    if individual_accuracy[i][response_id.item()] == 0: # even if one response in the cluster is incorrect, then do not include this cluster in calibration set
                        break # go to next cluster
                    elif (j==len(cluster_response_indices)-1): # all responses in this cluster are correct
                        cal_set_nlls.append(cluster_wise_nlls[i][cluster_num].item())
        return cal_set_nlls
    
    def get_prediction_sets(self, acc_name, num_gens, normalize):
        self2 = self.__class__(self.path, self.key[1], 'val', self.key[3], self.key[4])
        cal_semantic_set_ids, cal_clusters_entropy = self2.get_alpha_semantic_cluster_ids_and_nlls(num_gens, normalize) # CHANGED this to directly call the entropy() instead of calling uq->compute_uq_cached
        _, cal_individual_accuracy = self2.get_acc(acc_name) # cluster_wise_entropies[0][1].item() == float('inf')
        cal_individual_accuracy = cal_individual_accuracy.values # cal_individual_accuracy is pandas dataframe. Pandas Dataframe's columns values are converted to array by using df.values
        cal_set_entropies = self.get_cal_set_nlls(cal_semantic_set_ids, cal_clusters_entropy, cal_individual_accuracy)
        cal_set_entropies = np.array(cal_set_entropies)
        # determine semantic entropy threshold from the calibration set
        sorted_cal_set_entropies = np.sort(cal_set_entropies)
        epsilon = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        all_eps_avg_acc_list = []
        all_eps_avg_ss_list = []
        for eps in epsilon:
            cal_sement_threshold = sorted_cal_set_entropies[int((1-eps)*len(sorted_cal_set_entropies))]
            print("cal_sement_threshold: ", cal_sement_threshold)

            # test set statistics
            self2 = self.__class__(self.path, self.key[1], 'test', self.key[3], self.key[4])
            test_semantic_set_ids, test_clusters_entropy = self2.get_alpha_semantic_cluster_ids_and_nlls(num_gens, normalize) # CHANGED this to directly call the entropy() instead of calling uq->compute_uq_cached
            _, test_individual_accuracy = self2.get_acc(acc_name)
            test_individual_accuracy = test_individual_accuracy.values # dataset X num_gen

            acc_info = []
            set_size_info = []
            for i in range(len(test_semantic_set_ids)): # for each datapoint in the test dataset
                prediction_set_indices = []
                acc_flag = False # will be set to false for this datapoint if none of the cluster representative in the prediciton set is accurate
                response_cluster_ids = test_semantic_set_ids[i] # test_semantic_set_ids[i] is a list of 20 cluster ids for each (20) generated response
                no_of_clusters = len(torch.unique(response_cluster_ids))
                for cluster_num in range(no_of_clusters): # for each cluster
                    if test_clusters_entropy[i][cluster_num] < cal_sement_threshold:
                        cluster_response_indices = (test_semantic_set_ids[i] == cluster_num).nonzero(as_tuple=True)[0] # get indices of all the responses (or generations) in this cluster
                        prediction_set_indices.append(cluster_response_indices[0].item()) # adding index of the first response in the cluster to the prediction set
                        if test_individual_accuracy[i][cluster_response_indices[0].item()] == 1:
                            if acc_flag == False:
                                acc_info.append(1)
                                acc_flag = True
                if acc_flag == False:
                    acc_info.append(0) # empty prediction set is also inaccurate
                set_size_info.append(len(prediction_set_indices)/num_gens)
            acc_info = np.array(acc_info)
            set_size_info = np.array(set_size_info)
            assert len(acc_info) == len(set_size_info) == len(test_semantic_set_ids)
            avg_acc = np.mean(acc_info)
            avg_set_size = np.mean(set_size_info)
            #print(f"For epsilon {eps}, avg set size={avg_set_size}, and avg_acc={avg_acc}")
            all_eps_avg_acc_list.append(avg_acc)
            all_eps_avg_ss_list.append(avg_set_size)
        all_eps_avg_acc_list = np.array(all_eps_avg_acc_list)
        all_eps_avg_ss_list = np.array(all_eps_avg_ss_list)*100
        final_avg_acc = np.mean(all_eps_avg_acc_list)
        final_avg_ss = np.mean(all_eps_avg_ss_list)*100
        print(f"Epsilons: {epsilon}")
        print(f"Accuracy of Prediction Sets at different epsilons: {all_eps_avg_acc_list}")
        print(f"Set sizes of Prediction Sets at different epsilons: {all_eps_avg_ss_list}")

    def summ(self, uq_names, acc_name:str, num_gens=20, uq_kwargs:dict=None, overall=False, use_conf=True):
        # removed the "Gen with Confidence" code for finding best hyperparamters for their approach and used the default params for fair comparison with baselines [also the best results for their (Gen with Confidence paper) is with their default params]
        uq_kwargs = {}
        uq_kwargs.update(self._get_default_params())
        if isinstance(uq_names, str):
            uq_names = [uq_names]
        summ_obj = eval_uq.Summarizer({_: self.get_uq(_, num_gens, **uq_kwargs.get(_,{})) for _ in uq_names},
                                    self.get_acc(acc_name),
                                    lengths = self.get_length(num_gens)[1])
        return summ_obj

    def _get_default_params(self, ):
        hyparams = {}
        for uq_name in self.uq_measures:
            if uq_name not in self.tunable_hyperparams: continue
            hyparams[uq_name] = {k:v for k,v in self.default_params.items() if k in self.tunable_hyperparams[uq_name]}
        return hyparams

if __name__ == '__main__':
    from _settings import GEN_PATHS
    o = UQ_summ(GEN_PATHS[args.dataset][args.model], clean=True, split='test', cal_size=args.cal_size, seed=1) 
    num_gens = args.num_gens
    acc_name = f'generations|{args.acc_name}|acc'

    if args.output_pred_sets: # for conformal prediction results
        o.get_prediction_sets(acc_name = acc_name, num_gens = args.num_gens, normalize=args.norm_sets)
    
    else: # for UQ results
        summ_kwargs = {
            'u+ea': {'overall': True, 'use_conf': False},
            'u+ia': {'overall': False, 'use_conf': False},
            'c+ia': {'overall': False, 'use_conf': True},
        }['c+ia']

        summ_obj = o.summ([
            # baselines
            'generations|spectral_eigv_clip|agreement_w', 
            'semanticEntropy|unnorm', 
            'semanticEntropy|norm', 
            'generations|numsets', 
            #'lexical_sim',
            #'self_prob',

            # ours with alpha clustering
            'alphaSemanticEntropy|unnorm',
            'alphaSemanticEntropy|norm',
        ], 
            
            acc_name=acc_name,
            num_gens=num_gens, **summ_kwargs
        )
        print(summ_obj.summ_overall(args.metric))
