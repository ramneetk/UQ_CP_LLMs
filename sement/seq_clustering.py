import torch

def alpha_clustering(probs, alpha=0.5):
    num_points = len(probs)
    clusters = [[0]] #first response in cluster by itself
    for i in range(1, num_points):
        cluster_scores = torch.zeros(len(clusters) + 1)
        for j, C in enumerate(clusters):
            score_pairs = [(probs[i, c_i], probs[c_i, i]) for c_i in C]
            scores = [max(p) for p in score_pairs]
            avg_score = sum(scores) / len(scores)
            cluster_scores[j] = avg_score
        new_cluster_score = alpha / (alpha + len(clusters))
        cluster_scores[-1] = new_cluster_score
        
        cluster_probs = torch.softmax(cluster_scores, dim=0)
        assigment = torch.argmax(cluster_probs)
        if assigment == len(clusters): #new cluster
            clusters.append([i])
        else:
            clusters[assigment].append(i)
    return clusters
