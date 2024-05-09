import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import networkx as nx
import scipy as sp
import pdb

def get_affinity_mat(logits, mode='disagreement', temp=None, symmetric=True):
    if mode == 'jaccard':
        return logits
    # can be weigheted
    if mode == 'disagreement':
        logits = (logits + logits.permute(1,0,2))/2
        W = logits.argmax(-1) != 0
    if mode == 'disagreement_w':
        W = torch.softmax(logits/temp, dim=-1)[:, :, 0]
        if symmetric:
            W = (W + W.permute(1,0))/2
        W = 1 - W
    if mode == 'agreement':
        logits = (logits + logits.permute(1,0,2))/2
        W = logits.argmax(-1) == 2
    if mode == 'agreement_w':
        W = torch.softmax(logits/temp, dim=-1)[:, :, 2]
        if symmetric:
            W = (W + W.permute(1,0))/2
    if mode == 'gal':
        W = logits.argmax(-1)
        _map = {i:i for i in range(len(W))}
        for i in range(len(W)):
            for j in range(i+1, len(W)):
                if min(W[i,j], W[j,i]) > 0:
                    _map[j] = _map[i]
        W = torch.zeros_like(W)
        for i in range(len(W)):
            W[i, _map[i]] = W[_map[i], i] = 1
    W = W.cpu().numpy()
    W[np.arange(len(W)), np.arange(len(W))] = 1
    W = W.astype(np.float32)
    return W

def get_D_mat(W):
    # compute the degreee matrix from the weighted adjacency matrix
    D = np.diag(np.sum(W, axis=1))
    return D

def get_L_mat(W, symmetric_laplacian=True, symmetric_W=True):
    # compute the degreee matrix from the weighted adjacency matrix
    D = np.diag(np.sum(W, axis=1))
    # compute the normalized laplacian matrix from the degree matrix and weighted adjacency matrix
    if symmetric_laplacian and symmetric_W: # symmetric normalized for undirected (symmetric_W) graph: same as the gen with confidence paper
        L = np.linalg.inv(np.sqrt(D)) @ (D - W) @ np.linalg.inv(np.sqrt(D))
    elif not(symmetric_laplacian) and symmetric_W: # Random Walk (RW) for undirected (symmetric_W) graph
        #raise NotImplementedError() ## COMMENTING THIS FOR Random Walk LAPLACIAN ###
        # compute the normalized laplacian matrix from the degree matrix and weighted adjacency matrix
        L = np.linalg.inv(D) @ (D - W)
    elif not(symmetric_laplacian) and not(symmetric_W): # laplacian for directed graph (!symmetric_W)
        dg = nx.DiGraph()
        #creating weighted edges required to call Laplacian for directed graphs using nx
        no_of_nodes = W.shape[0]
        weighted_edges = []
        for i in range(no_of_nodes):
            for j in range(no_of_nodes):
                edge_ij = (i+1, j+1, W[i][j])
                weighted_edges.append(edge_ij)
        dg.add_weighted_edges_from(weighted_edges) 
        L = nx.directed_laplacian_matrix(G=dg, walk_type='pagerank') # we can also set the param 'walk_type' from {random, lazy, pagerank}

    return L.copy()

def get_eig(L, symmetric, thres=None, eps=None):
    # This function assumes L is symmetric #### FIXED THIS IN ORIGINAL CODE
    # compute the eigenvalues and eigenvectors of the laplacian matrix
    if eps is not None:
        L = (1-eps) * L + eps * np.eye(len(L))
    
    if symmetric:
        eigvals, eigvecs = np.linalg.eigh(L)
    
    else:
        #pdb.set_trace()
        eigvals, eigvecs = np.linalg.eig(L)

    #eigvals, eigvecs = np.linalg.eig(L)
    #assert np.max(np.abs(eigvals.imag)) < 1e-5
    #eigvals = eigvals.real
    #idx = eigvals.argsort()
    #eigvals = eigvals[idx]
    #eigvecs = eigvecs[:,idx]

    if thres is not None:
        keep_mask = eigvals < thres
        eigvals, eigvecs = eigvals[keep_mask], eigvecs[:, keep_mask]
    return eigvals, eigvecs

def find_equidist(P, eps=1e-4):
    from scipy.linalg import eig
    P = P / P.sum(1)[:, None]
    P = (1-eps) * P + eps * np.eye(len(P))
    assert np.abs(P.sum(1)-1).max() < 1e-3
    w, vl, _ = eig(P, left=True)
    #assert np.max(np.abs(w.imag)) < 1e-5
    w = w.real
    idx = w.argsort()
    w = w[idx]
    vl = vl[:, idx]
    assert np.max(vl[:, -1].imag) < 1e-5
    return vl[:, -1].real / vl[:, -1].real.sum()

class SpetralClusteringFromLogits:
    def __init__(self,
                 affinity_mode='disagreement_w',
                 eigv_threshold=0.9,
                 cluster=True,
                 temperature=3., adjust=False) -> None:
        self.affinity_mode = affinity_mode
        self.eigv_threshold = eigv_threshold
        self.rs = 0
        self.cluster = cluster
        self.temperature = temperature
        self.adjust = adjust
        if affinity_mode == 'jaccard':
            assert self.temperature is None

    def get_laplacian(self, logits, symmetric_laplacian=True, symmetric_W=True):
        W = get_affinity_mat(logits, mode=self.affinity_mode, temp=self.temperature, symmetric=symmetric_W) # default is True for symmetry of W and Laplacian 
        L = get_L_mat(W, symmetric_laplacian=symmetric_laplacian, symmetric_W=symmetric_W) # originally true
        return L

    def get_eigvs(self, logits, symmetric_laplacian=True, symmetric_W=True):
        L = self.get_laplacian(logits, symmetric_laplacian=symmetric_laplacian, symmetric_W=symmetric_W)
        return (1-get_eig(L, symmetric=symmetric_laplacian)[0])

    def __call__(self, logits, cluster=None, symmetric_laplacian=True, symmetric_W=True):
        if cluster is None: cluster = self.cluster
        L = self.get_laplacian(logits, symmetric_laplacian=symmetric_laplacian, symmetric_W=symmetric_W)
        if not cluster:
            return (1-get_eig(L, symmetric=symmetric_laplacian)[0]).clip(0 if self.adjust else -1).sum()
        eigvals, eigvecs = get_eig(L, symmetric=symmetric_laplacian, thres=self.eigv_threshold)
        k = eigvecs.shape[1]
        self.rs += 1
        kmeans = KMeans(n_clusters=k, random_state=self.rs, n_init='auto').fit(eigvecs)
        return kmeans.labels_

    def clustered_entropy(self, logits):
        from scipy.stats import entropy
        labels = self(logits, cluster=True)
        P = torch.softmax(logits, dim=-1)[:, :, 2].cpu().numpy()
        pi = find_equidist(P)
        clustered_pi = pd.Series(pi).groupby(labels).sum().values
        return entropy(clustered_pi)

    def eig_entropy(self, logits, symmetric_laplacian=True, symmetric_W=True):
        W = get_affinity_mat(logits, mode=self.affinity_mode, temp=self.temperature, symmetric=symmetric_W)
        L = get_L_mat(W, symmetric_laplacian=symmetric_laplacian, symmetric_W=symmetric_W)
        eigs = get_eig(L, symmetric=symmetric_laplacian, eps=1e-4)[0] / W.shape[0]
        return np.exp(- (eigs * np.nan_to_num(np.log(eigs))).sum())

    def proj(self, logits, symmetric_laplacian=True, symmetric_W=True): # originally True for symmetry of both L and W
        W = get_affinity_mat(logits, mode=self.affinity_mode, temp=self.temperature, symmetric=symmetric_W)
        L = get_L_mat(W, symmetric_laplacian=symmetric_laplacian, symmetric_W=symmetric_W)
        eigvals, eigvecs = get_eig(L, symmetric=symmetric_laplacian, thres=self.eigv_threshold)
        return W, L, eigvals, eigvecs # orig: return eigvecs

    def orig_proj(self, logits, symmetric_laplacian=True, symmetric_W=True): # originally True for symmetry of both L and W
        W = get_affinity_mat(logits, mode=self.affinity_mode, temp=self.temperature, symmetric=symmetric_W)
        L = get_L_mat(W, symmetric_laplacian=symmetric_laplacian, symmetric_W=symmetric_W)
        eigvals, eigvecs = get_eig(L, symmetric=symmetric_laplacian, thres=self.eigv_threshold)
        return eigvecs # orig: return eigvecs

    def kmeans(self, eigvals, eigvecs, k_threshold, orig):
        if not(orig):
            # https://arxiv.org/pdf/0711.0189.pdf: (k+1)th eigen value has a huge gap from kth eigen value
            eigvals = np.sort(eigvals)
            k=1
            for i in range(len(eigvals)-1):
                if (eigvals[i+1]-eigvals[i]) >= k_threshold:
                    break
                else:
                    k += 1
        #k = sum(eigvecs < 0)
        else: # originally
            k = eigvecs.shape[1] # orig
        self.rs += 1
        kmeans = KMeans(n_clusters=k, random_state=self.rs, n_init='auto').fit(eigvecs)
        return kmeans.labels_

    def create_clusters(self, eigvecs): # eigvals and eigvecs are arrays over entire dataset
        cluster_labels_all_datapoints = []
        # range_n_clusters = [j+1 for j in range(eigvecs[i].shape[1])] # this might be problematic as some datapoints have only 1 eigvector 
        range_n_clusters = [2,3,4,5,6] # from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py

        for i in range(len(eigvecs)): # iterating over each datapoint in the dataset
            all_cluster_labels = []
            all_silhouette_avg = []
            # check on the number of eigen values: if there is only 1 eigen value, then return only 1 cluster
            # if len(eigvecs[i][0]) == 1:
            #     cluster_labels_all_datapoints.append(np.array([0]*len(eigvecs[i]))) # put all responses in one cluster: cluster 0
            # else:
            for n_clusters in range_n_clusters: # iterating over the number of clusters
                clusterer = KMeans(n_clusters=n_clusters, random_state=self.rs+1) # random_state = 1
                cluster_labels = clusterer.fit_predict(eigvecs[i].real)
                all_cluster_labels.append(cluster_labels)
                # The silhouette_score gives the average value for all the samples. This gives a perspective into the density and separation of the formed clusters
                silhouette_avg = silhouette_score(eigvecs[i].real, cluster_labels)
                all_silhouette_avg.append(silhouette_avg)
            all_silhouette_avg = np.array(all_silhouette_avg)
            cluster_labels_all_datapoints.append(all_cluster_labels[np.argmax(all_silhouette_avg)])
        
        return cluster_labels_all_datapoints


def umap_visualization(eigvecs, labels):
    # perform umap visualization on the eigenvectors
    import umap
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(eigvecs)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels)
    return embedding