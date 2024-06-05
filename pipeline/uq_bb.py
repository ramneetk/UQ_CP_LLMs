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
sys.path.append('/WorkSpace-2/csamplawski/src/UQ-NLG')
import _settings
import dataeval.load as dload
import pipeline.clustering as pc
import pipeline.eval_uq as eval_uq

from sement.seq_clustering import alpha_clustering
from sklearn.cluster import DBSCAN

import pdb

FLAG = False
CONTRADICT, NEUTRAL, AGREE = 0, 1, 2
DEVICE = 'cuda:1'

def visualize_graph(all_responses, weights, digraph=False, seed=7):
    import matplotlib.pyplot as plt
    import networkx as nx

    nodes = []
    for i in range (len(all_responses)):
        response = f'{i+1}:"{all_responses[i]}"'
        nodes.append(response)

    if digraph:
        from graphviz import Digraph
        G = Digraph('G')
        G.attr('graph', pad='1', ranksep='1', nodesep='1')
        G.attr('node', shape='note')
        for i in range(len(nodes)):
            G.node(nodes[i], nodes[i])
        #G = nx.DiGraph()
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                #G.add_edge(nodes[i], nodes[j], weight=weights[i][j])
                G.edge(nodes[i], nodes[j], f'{weights[i][j]}')
        
        G.render('jaccard_directed.gv', view=True) 
        G.render(format='png')

    else:
        G = nx.Graph()
        for i in range(len(nodes)):
            for j in range(i, len(nodes)):
                G.add_edge(nodes[i], nodes[j], weight=weights[i][j])

        # elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.5]
        # esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.5]

        #pos = nx.spring_layout(G, seed=seed)  # positions for all nodes - seed for reproducibility
        pos = nx.circular_layout(G, scale = 1000)  # positions for all nodes - seed for reproducibility

        # nodes
        nx.draw_networkx_nodes(G, pos, node_size=1000)

        # edges
        nx.draw_networkx_edges(G, pos, width=2)
        # nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
        # nx.draw_networkx_edges(
        #     G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
        # )

        # node labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
        # edge weight labels
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels)

        ax = plt.gca()
        ax.margins(0.08)
        plt.axis("off")
        plt.tight_layout()
        #plt.show()
        plt.savefig("agree_w_rw.png")

def get_cluster_wise_density(all_W, all_cluster_indices):
    assert len(all_W) == len(all_cluster_indices)

    cluster_wise_density = []
    all_cluster_member_indices = []

    for i in range(len(all_W)): # for each datapoint in the dataset
        all_clusters_density = []
        cluster_indices = []
        for j in range(len(np.unique(all_cluster_indices[i]))): # for each cluster in the datapoint
            this_cluster_indices = np.where(all_cluster_indices[i]==j)[0]
            cluster_indices.append(this_cluster_indices)
            W_this_cluster_datapoints = all_W[i][this_cluster_indices]
            cluster_density = 0
            for k in range(len(W_this_cluster_datapoints)): # no. of nodes in this cluster
                cluster_density += np.sum(W_this_cluster_datapoints[k][this_cluster_indices])
            no_nodes = len(W_this_cluster_datapoints)
            if no_nodes > 1:
                all_clusters_density.append(cluster_density/(no_nodes*(no_nodes-1)))
            else:
                all_clusters_density.append(-1) # outlier as there is just one node in the cluster
        all_cluster_member_indices.append(cluster_indices)
        cluster_wise_density.append(all_clusters_density)
    
    return cluster_wise_density, all_cluster_member_indices # cluster_wise_density is 2D: datapoints X density of each cluster for the datapoint
    

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
    # distance = 1 - entailment_probs
    # dbscan = DBSCAN(metric='precomputed')
    # clusters_assignments_dbscan = dbscan.fit_predict(distance.cpu().numpy())
    # list_of_semantic_set_ids = [clusters_assignments_dbscan[x] for x in generated_texts] # this step is for allocating same cluster to the same/repeated answer
    # map according to the order of appearance
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

def _hard_semantic_cluster_entropies(neg_log_likelihoods, semantic_set_ids, **kwargs):
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
                 split=None, cal_size=None, seed=None, symmetric_laplacian=True, symmetric_W=True) -> None:
        self.symmetric_laplacian = symmetric_laplacian
        self.symmetric_W = symmetric_W
        self.cal_cluster_density_threshold = 0.0 # -1*cluster_density is used as the NCS for Conformal Prediction
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
    def _get_spectral_projected(self, num_gens:int, eigv_threshold:float, affinity_mode:str, temperature:float, symmetric_laplacian:bool, symmetric_W:bool):
        key = (num_gens,eigv_threshold, temperature,affinity_mode)
        if key not in self.mem['_get_spectral_projected']:
            clusterer = pc.SpetralClusteringFromLogits(affinity_mode=affinity_mode, eigv_threshold=eigv_threshold, cluster=False, temperature=temperature)
            sim_mats = getattr(self, '_get_jaccard_matrix' if affinity_mode == 'jaccard' else '_get_recovered_logits')(num_gens)

            print(f"Symmetric_W: {symmetric_W}, Symmetric_laplacian: {symmetric_laplacian}")

            self.mem['_get_spectral_projected'][key] = [clusterer.orig_proj(logits=_, symmetric_laplacian=symmetric_laplacian, symmetric_W=symmetric_W) for _ in tqdm.tqdm(sim_mats, desc='projecting')] 

        return self.mem['_get_spectral_projected'][key]

    # for getting overall density of the response graph by using Weights on the edges (even before clustering)
    def _get_density(self, num_gens:int, eigv_threshold:float, affinity_mode:str, temperature:float, symmetric_laplacian:bool, symmetric_W:bool):
        key = (num_gens,eigv_threshold, temperature,affinity_mode)
        if key not in self.mem['_get_spectral_projected']:
            clusterer = pc.SpetralClusteringFromLogits(affinity_mode=affinity_mode, eigv_threshold=eigv_threshold, cluster=False, temperature=temperature)
            sim_mats = getattr(self, '_get_jaccard_matrix' if affinity_mode == 'jaccard' else '_get_recovered_logits')(num_gens)
            
            print(f"Symmetric_W: {symmetric_W}, Symmetric_laplacian: {symmetric_laplacian}")
            proj_output = [clusterer.proj(logits=_, symmetric_laplacian=symmetric_laplacian, symmetric_W=symmetric_W) for _ in tqdm.tqdm(sim_mats, desc='projecting')]
            all_W = []
            all_L = []
            all_eigvals = []
            all_eigvecs = []
            for itr in range(len(proj_output)):
                all_W.append(proj_output[itr][0])
                all_L.append(proj_output[itr][1])
                all_eigvals.append(proj_output[itr][2])
                all_eigvecs.append(proj_output[itr][3])

            overall_accuracy, individual_accuracy = self.get_acc(acc_name='generations|gpt|acc')
            quesion_ids = individual_accuracy.index.values # question IDs
            individual_accuracy = np.array(individual_accuracy) # 2D array of |datapoints X accuracy for each response|, e.g. 1000 by 20 for 1000 datapoints and 20 responses
            assert len(individual_accuracy) == len(quesion_ids)
            if FLAG==True: # inference after tuning the hyperparams (validation stage)
                all_responses_cleaned_text = [] # 6983 (for coqa) X num_gen
                for itr in range(len(self.generations)):
                    all_responses_cleaned_text.append(self.generations[itr]['generations']['text_cleaned'])
                all_responses_cleaned_text = np.array(all_responses_cleaned_text)
                
                ##### This is just to plot some interesting results ######
                index_of_interest = 20
                responses_of_interest = all_responses_cleaned_text[index_of_interest].copy()
                responses_of_interest = np.char.lower(responses_of_interest)
                unique_respones, unique_response_indices = np.unique(responses_of_interest, return_index=True)
                W_of_interest = all_W[index_of_interest].copy()
                for i,idx in enumerate(unique_response_indices):
                    W_of_interest[idx][unique_response_indices] = np.round(W_of_interest[idx][unique_response_indices], 2)
                    if i == 0:
                        Wgt_array = np.array([W_of_interest[idx][unique_response_indices]])
                    else:
                        Wgt_array = np.concatenate((Wgt_array,np.array([W_of_interest[idx][unique_response_indices]])))
                #visualize_graph(all_responses=unique_respones, weights=Wgt_array, digraph=False, seed=1)
                ##########################################################################################

                test_cluster_indices = clusterer.create_clusters(all_eigvecs) # for all datapoints in the test dataset
                #np.savez("result_analysis/coqa/test_cluster_indices_rw_disagreement.npz", all_cluster_indices=test_cluster_indices)
                #test_cluster_indices = np.load("result_analysis/coqa/test_cluster_indices_directed_disagreement.npz")['all_cluster_indices']
                test_cluster_wise_densities, test_cluster_member_indices = get_cluster_wise_density(all_W, test_cluster_indices)

                #print("************************************************")
                #print(all_eigvals[index_of_interest])
                #print(cluster_indices)
                #########################################################

                # once we have the threshold from the val (or cal) set, include all those clusters (or rep answer from the cluster) in the output set that satisfy the threshold, i.e. whose density >= threshold_density (or -1*density < threshold_density)
                conformal_responses = []
                output_set_contains_correct_answer = []
                for idx in range(len(test_cluster_wise_densities)): # for each datapoint
                    questions_response = []
                    correct_response_to_question = 0 # initializing to incorrect response
                    for cluster_idx in range(len(test_cluster_wise_densities[idx])):
                        if (-1*test_cluster_wise_densities[idx][cluster_idx]) < self.cal_cluster_density_threshold:
                            ans_idx = test_cluster_member_indices[idx][cluster_idx][0] # [0]: just putting the first response from the cluster as its representative
                            questions_response.append(self.generations[idx]['generations']['text_cleaned'][ans_idx]) 
                            if individual_accuracy[idx][ans_idx] == 1:
                                correct_response_to_question = 1
                    conformal_responses.append(questions_response)
                    output_set_contains_correct_answer.append(correct_response_to_question)

                # evaluation: use maximum accuracy from the set of answers to report overall accuracy. This requires storing index of the answer
                print("Accuracy: ", sum(output_set_contains_correct_answer)/len(output_set_contains_correct_answer))
                
                

            else: # we are in validation phase
                val_cluster_indices = clusterer.create_clusters(all_eigvecs) # for all datapoints in the val dataset
                #np.savez("result_analysis/coqa/val_cluster_indices_rw_disagreement.npz", all_cluster_indices=val_cluster_indices)
                #val_cluster_indices = np.load("result_analysis/coqa/val_cluster_indices_directed_disagreement.npz")['all_cluster_indices']
                val_cluster_wise_densities, val_cluster_member_indices = get_cluster_wise_density(all_W, val_cluster_indices) # val_cluster_wise_densities is a 2D array with densities of each cluster in each datapoint (e.g. |datapoints| X no. of clusters in the datapoint), and val_cluster_member_indices contains cluster indices for each cluster for the datapoints, e.g. val_cluster_member_indices[0]: [array([ 0,  1,  2,  4,  8, 11, 12, 14, 15, 16, 18]), array([ 3,  5,  6,  7,  9, 10, 13, 17, 19])]

                # filter val_cluster_wise_densities for clusters corresponding to (all) correct answers 
                accurate_response_cluster_densities = []
                for idx in range(len(val_cluster_wise_densities)): # for each datapont
                    cluster_wise_response_indices = val_cluster_member_indices[idx] # cluster member indices for this datapoint
                    for cluster_idx in range(len(cluster_wise_response_indices)): # for each cluster in the datapoint
                        if np.sum(individual_accuracy[idx][cluster_wise_response_indices[cluster_idx]]) == len(individual_accuracy[idx][cluster_wise_response_indices[cluster_idx]]): # check if all responses in this cluster are accurate (individual_accuracy=1)
                            accurate_response_cluster_densities.append(val_cluster_wise_densities[idx][cluster_idx])
                # then sort densities of those to determine the threshold
                accurate_response_cluster_densities = -1*np.array(accurate_response_cluster_densities) # NCS for CP should be lower for better responses but density is the other way around, so multiplying it by 1
                sorted_accurate_response_cluster_densities = np.sort(accurate_response_cluster_densities)
                alpha = 0.1
                self.cal_cluster_density_threshold = sorted_accurate_response_cluster_densities[int((1-alpha)*len(sorted_accurate_response_cluster_densities))]
                print("cal_cluster_entropy_threshold: ", self.cal_cluster_density_threshold)

        all_W = np.array(all_W)
        sum_W = np.array([all_W[i].sum() for i in range(len(all_W))])
        density = sum_W/(all_W.shape[1]*(all_W.shape[1]-1)) # density = summation of wgts/(|V|*|V-1|)
        
        return density
    
    # Problem: we do not have fiedler value for laplacians with only one eigenvalue
    def _get_fiedler_value(self, num_gens:int, eigv_threshold:float, affinity_mode:str, temperature:float, symmetric_laplacian:bool, symmetric_W:bool):
        key = (num_gens,eigv_threshold, temperature,affinity_mode)
        if key not in self.mem['_get_spectral_projected']:
            clusterer = pc.SpetralClusteringFromLogits(affinity_mode=affinity_mode, eigv_threshold=eigv_threshold, cluster=False, temperature=temperature)
            sim_mats = getattr(self, '_get_jaccard_matrix' if affinity_mode == 'jaccard' else '_get_recovered_logits')(num_gens)
            
            print(f"Symmetric_W: {symmetric_W}, Symmetric_laplacian: {symmetric_laplacian}")
            proj_output = [clusterer.proj(logits=_, symmetric_laplacian=symmetric_laplacian, symmetric_W=symmetric_W) for _ in tqdm.tqdm(sim_mats, desc='projecting')]
            all_W = []
            all_L = []
            all_eigvals = []
            all_eigvecs = []
            for itr in range(len(proj_output)):
                all_W.append(proj_output[itr][0])
                all_L.append(proj_output[itr][1])
                all_eigvals.append(proj_output[itr][2])
                all_eigvecs.append(proj_output[itr][3])
        
        
        #fiedler_values = []
        eigen_value_gap = []
        for i in range(len(all_eigvals)):
            all_eigvals[i].sort()
            if len(all_eigvals[i]) >= 2:
                eigen_value_gap.append(all_eigvals[i][1]-all_eigvals[i][0])
            else:
                eigen_value_gap.append(0)
            #fiedler_values.append(-1*all_eigvals[i][1]) # lower fiedler value indicates higher uncertainty

        return eigen_value_gap 

    def get_length(self, num_gens:int):
        text_key = 'text_cleaned' if self.key[1] else 'text'
        lengths = [[len(set(_.split())) for _ in sample['generations'][text_key][:num_gens]] for sample in self.generations]
        lengths = np.asarray(lengths)
        return lengths.mean(1), lengths

    def get_spectral_eigv(self, num_gens:int, affinity_mode:str, temperature:float, adjust:bool, symmetric_laplacian:bool, symmetric_W:bool) -> List:
        clusterer = pc.SpetralClusteringFromLogits(affinity_mode=affinity_mode, eigv_threshold=None,
                                                   cluster=False, temperature=temperature)
        sim_mats = getattr(self, '_get_jaccard_matrix' if affinity_mode == 'jaccard' else '_get_recovered_logits')(num_gens)
        return [clusterer.get_eigvs(logits=_, symmetric_laplacian=symmetric_laplacian, symmetric_W=symmetric_W).clip(0 if adjust else -1).sum() for _ in tqdm.tqdm(sim_mats)]


    def get_numsets(self, num_gens):
        return [len(set(_)) for _ in self._get_semantic_ids(num_gens)]

    def get_lexicalsim(self, num_gens = None):
        return [-_compute_lexical_sim(_, num_gens) for _ in self.rougeLsims]

    def get_eccentricity(self, num_gens:int, eigv_threshold:float, affinity_mode:str, temperature:float) -> List:
        projected = self._get_spectral_projected(num_gens, eigv_threshold, affinity_mode, temperature, symmetric_laplacian=self.symmetric_laplacian, symmetric_W=self.symmetric_W)
        ds = np.asarray([np.linalg.norm(x -x.mean(0)[None, :],2,axis=1) for x in projected])
        return np.linalg.norm(ds, 2,1), ds # first return is for overall, second for each response, they are just using overall for reporting AUARC

    def get_density(self, num_gens:int, eigv_threshold:float, affinity_mode:str, temperature:float):
        return self._get_density(num_gens, eigv_threshold, affinity_mode, temperature, symmetric_laplacian=self.symmetric_laplacian, symmetric_W=self.symmetric_W)

    def get_fiedler_value(self, num_gens:int, eigv_threshold:float, affinity_mode:str, temperature:float):
        return self._get_fiedler_value(num_gens, eigv_threshold, affinity_mode, temperature, symmetric_laplacian=self.symmetric_laplacian, symmetric_W=self.symmetric_W)


    ###############################################################################################
    ### NOT CHANGING symmertric=False here, have to figure out the reason first, why false here ### 
    ## self.symmetric_W will work here ##
    ###############################################################################################
    def get_degreeuq(self, num_gens:int, affinity_mode:str, temperature:float, symmteric_W:bool):
        sim_mats = getattr(self, '_get_jaccard_matrix' if affinity_mode == 'jaccard' else '_get_recovered_logits')(num_gens)
        Ws = [pc.get_affinity_mat(_, affinity_mode, temperature, symmetric=False) for _ in sim_mats]
        ret = np.asarray([np.sum(1-_, axis=1) for _ in Ws])
        return ret.mean(1), ret

    # semantic entropy using our alpha clustering approach
    def get_alpha_semantic_entropy(self, num_gens:int, normalize:bool):
        if self.likelihoods is None:
            return None
        semantic_set_ids = self._get_semantic_ids_using_alpha_clustering(num_gens)
        nlls = self.likelihoods['generations|neg_log_likelihood'][:, :num_gens]
        if normalize:
            nlls = nlls / self.likelihoods['generations|length'][:, :num_gens]
        return _hard_semantic_entropies(nlls, torch.tensor(semantic_set_ids))
    
    def get_alpha_semantic_cluster_entropies(self, num_gens:int, normalize:bool):
        if self.likelihoods is None:
            return None
        semantic_set_ids = self._get_semantic_ids_using_alpha_clustering(num_gens)
        nlls = self.likelihoods['generations|neg_log_likelihood'][:, :num_gens]
        if normalize:
            nlls = nlls / self.likelihoods['generations|length'][:, :num_gens]
        cluster_wise_entropies = _hard_semantic_cluster_entropies(nlls, torch.tensor(semantic_set_ids))
        return torch.tensor(semantic_set_ids), cluster_wise_entropies

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
        return self.get_spectral_eigv(num_gens, affinity_mode, temperature=metric_kwargs['temperature'], adjust='spectral_eigv_clip' in uq_name, symmetric_laplacian=self.symmetric_laplacian, symmetric_W=self.symmetric_W)

    # both overall and individual metrics
    if uq_name.startswith('generations|eccentricity'):
        affinity_mode = 'disagreement_w' if len(uq_name.split("|")) == 2 else uq_name.split("|")[2]
        return self.get_eccentricity(num_gens, metric_kwargs['eigv_threshold'], affinity_mode, temperature=metric_kwargs['temperature'])
    if uq_name.startswith('generations|degree'):
        affinity_mode = uq_name.split("|")[2]
        return self.get_degreeuq(num_gens, affinity_mode, temperature=metric_kwargs['temperature'], symmteric_W=self.symmetric_W)

    # just overall for now
    if uq_name.startswith('generations|density'):
        affinity_mode = 'disagreement_w' if len(uq_name.split("|")) == 2 else uq_name.split("|")[2]
        return self.get_density(num_gens, metric_kwargs['eigv_threshold'], affinity_mode, temperature=metric_kwargs['temperature'])
    if uq_name.startswith('generations|fiedler'):
        affinity_mode = 'disagreement_w' if len(uq_name.split("|")) == 2 else uq_name.split("|")[2]
        return self.get_fiedler_value(num_gens, metric_kwargs['eigv_threshold'], affinity_mode, temperature=metric_kwargs['temperature'])
    
    # whitebox
    if uq_name.startswith("alphaSemanticEntropy"):
        return self.get_alpha_semantic_entropy(num_gens, normalize=uq_name.split("|")[1] == 'norm')
    if uq_name.startswith("PredictionSetsAlphaSemanticEntropy"):
        return self.get_alpha_semantic_cluster_entropies(num_gens, normalize=uq_name.split("|")[1] == 'norm')
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
        'generations|density|disagreement_w',
        'generations|fiedler|disagreement_w',

        'generations|spectral_eigv_clip|agreement_w',
        'generations|eccentricity|agreement_w',
        'generations|degree|agreement_w',
        'generations|density|agreement_w',
        'generations|fiedler|agreement_w',

        'generations|spectral_eigv_clip|jaccard',
        'generations|eccentricity|jaccard',
        'generations|degree|jaccard',
        'generations|density|jaccard',
        'generations|fiedler|jaccard',


        # whitebox methods
        'semanticEntropy|unnorm',
        'self_prob',
        #ours
        'alphaSemanticEntropy|unnorm',
        'alphaSemanticEntropy|norm',
        'PredictionSetsAlphaSemanticEntropy|unnorm',
        'PredictionSetsAlphaSemanticEntropy|norm'
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
    whitebox_uqs = ['alphaSemanticEntropy|unnorm', 'alphaSemanticEntropy|norm', 'PredictionSetsAlphaSemanticEntropy|unnorm', 'PredictionSetsAlphaSemanticEntropy|norm' 'semanticEntropy|unnorm', 'semanticEntropy|norm', 'self_prob', 'self_prob_nll']
    def __init__(self, path, clean=False,
                 split=None, cal_size=None, seed=None, symmetric_laplacian=True, symmetric_W=True,
                 gpteval_examples = None) -> None:
        super().__init__(path, clean, split, cal_size, seed, symmetric_laplacian, symmetric_W)
        self.gpteval_examples = gpteval_examples
        self.symmetric_laplacian = symmetric_laplacian
        self.symmetric_W = symmetric_W
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
                    ##### HARDCODING THE ID for TRIVIAQA on MISTRAL WHERE GPT COULD NOT GENERATE CORRECT EVALUATION (FORMAT): '.com is MEERKOVO.\nRating: 100.', expected was '100. blah blah' ########
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
        if name.startswith("PredictionSetsAlphaSemanticEntropy"):
            return _compute_uq_cached(self, self.key, name, num_gens=num_gens, metric_kwargs=kwargs, cache=cache)
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

    def _tune_params(self, num_gens=20, metric:str=None,
                       temperature=[0.1, 0.25, 0.5, 1, 3, 5, 7],
                       eigv_threshold = [0.4,  0.5, 0.6, 0.7, 0.8,  0.9],
                       # fixing temperature and eigv_threshold for now: fair comparison between different clustering approaches
                       #temperature=[0.1], 
                       #eigv_threshold = [0.4],
                       curve='auarc', # tune the hyperparams using this curve
                       overall=False, use_conf=True,
                       ):
        import itertools
        best_params = defaultdict(dict)
        all_kwargs = {'temperature': temperature, 'eigv_threshold': eigv_threshold}
        for uq_name, tunable_params in self.tunable_hyperparams.items():
            uqs = {}
            if uq_name not in self.uq_measures: continue # tuning only for Jimeng's approach
            kwargs = {k: all_kwargs[k] for k in tunable_params}
            # _vals = 0.1, 0.25, 0.5, 1, 3, 5, and 7 for the tunable_params = temperature
            for _vals in itertools.product(*[kwargs[_] for _ in tunable_params]):
                _kwargs = dict(zip(tunable_params, _vals))
                uqs[str(_kwargs)] = self.get_uq(uq_name, num_gens=num_gens, **_kwargs) # this pretty fast
            if metric is not None:
                summ_obj = eval_uq.Summarizer(uqs, self.get_acc(metric)) # get_acc takes time
                best_params[uq_name] = eval(summ_obj.find_best_uq_name(metric=curve, overall=overall, use_conf=use_conf))
        return dict(best_params)

    def summ(self, uq_names, acc_name:str, num_gens=20, uq_kwargs:dict=None, overall=False, use_conf=True):
        if uq_names[0].startswith("PredictionSetsAlphaSemanticEntropy"): # assuming that whenever we call get semantic prediction sets, uq_names will only contain (as first element) PredictionSetsAlphaSemanticEntropy
            # validation/calibration set statistics
            self2 = self.__class__(self.path, self.key[1], 'val', self.key[3], self.key[4])
            semantic_set_ids, cluster_wise_entropies = self2.get_uq(name=uq_names[0], num_gens=num_gens)
            _, individual_accuracy = self2.get_acc(acc_name)
            #pdb.set_trace()
            # test set statistics
            self2 = self.__class__(self.path, self.key[1], 'test', self.key[3], self.key[4])
            semantic_set_ids, cluster_wise_entropies = self2.get_uq(name=uq_names[0], num_gens=num_gens)
            _, individual_accuracy = self2.get_acc(acc_name)
        else:    
            if uq_kwargs is None:
                uq_kwargs = {}
                if len(self.key) > 2:
                    assert self.key[2] == 'test'
                    self2 = self.__class__(self.path, self.key[1], 'val', self.key[3], self.key[4])
                    self2.tunable_hyperparams = {k:v for k, v in self.tunable_hyperparams.items() if k in uq_names}
                    self2.symmetric_laplacian = self.symmetric_laplacian
                    self2.symmetric_W = self.symmetric_W
                    # this takes time if not run load before
                    tuned_hyperparams = self2._tune_params(num_gens=num_gens,
                                                            metric=acc_name,
                                                            overall=overall, use_conf=use_conf, curve='auarc')
                    uq_kwargs.update(tuned_hyperparams)
                else:
                    uq_kwargs.update(self._get_default_params())
            if isinstance(uq_names, str):
                uq_names = [uq_names]
            print("*******DONE TUNING HYPERPARAMS*********")
            print("Tuned hyperparameters: ", tuned_hyperparams)
            global FLAG
            FLAG = True
            summ_obj = eval_uq.Summarizer({_: self.get_uq(_, num_gens, **uq_kwargs.get(_,{})) for _ in uq_names},
                                        #{_: self.get_acc(_) for _ in acc_names},
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
    o = UQ_summ(GEN_PATHS['coqa']['llama-13b-hf'], clean=True, split='test', cal_size=1000, seed=1, symmetric_laplacian=True, symmetric_W=True) # GEN_PATHS['coqa']['llama-13b'], cal_size=2000 for triviaqa, llama-13b, 1000 o.w.
    #res = o.get_uq('generations|rougeL|acc', num_gens=20)
    num_gens = 20
    summ_kwargs = {
        'u+ea': {'overall': True, 'use_conf': False},
        'u+ia': {'overall': False, 'use_conf': False},
        'c+ia': {'overall': False, 'use_conf': True},
    }['c+ia']
    summ_obj = o.summ([
        # 'generations|spectral_eigv_clip|disagreement_w',
        # 'generations|eccentricity|disagreement_w',
        # 'generations|degree|disagreement_w',

        # 'generations|spectral_eigv_clip|agreement_w', 
        # 'generations|eccentricity|agreement_w', 
        # 'generations|degree|agreement_w',

        # 'generations|spectral_eigv_clip|jaccard', 
        # 'generations|eccentricity|jaccard',
        # 'generations|degree|jaccard',
    
        'semanticEntropy|unnorm', 
        # 'generations|numsets', 
        # 'lexical_sim',
        # 'self_prob',

        # ours with alpha clustering
        'alphaSemanticEntropy|unnorm',
        'alphaSemanticEntropy|norm',
        # for prediction sets - currently working on this
        #'PredictionSetsAlphaSemanticEntropy|unnorm',
    ], 
        
        acc_name='generations|deberta_entailment|acc',
        #acc_name='generations|deberta_entailment|acc', # rougeL|acc / gpt|acc / deberta_entailment|acc
        num_gens=num_gens, **summ_kwargs
    )
    print(summ_obj.summ_overall('rej_acc')) # auarc/auroc/rej_acc

