import heapq
import numpy as np
import torch

def find_rel_dist (x, y, measure="Euclidean"):
    if (measure == "Euclidean"):
        return torch.linalg.norm(x - y).item()
    elif (measure == "cosine-sim"):
        return (torch.dot(x, y)/(torch.linalg.norm(x)*torch.linalg.norm(y))).item()

def all_knn (embeddings, k, measure="Euclidean"):
    knn_arr = np.zeros(shape=(embeddings.shape[0], k))
    for i in range(embeddings.shape[0]):
        knn_arr[i, :] = heapq.nsmallest(k+1, np.arange(embeddings.shape[0]), 
                lambda j: find_rel_dist(embeddings[j, :], embeddings[i, :], measure=measure))[1:]
    return knn_arr

def knn_embs (embs, n, k=100, measure="Euclidean"):
    knn_arr = np.zeros(shape=k)
    knn_arr = heapq.nsmallest(k+1, torch.arange(embs.shape[0]), 
                lambda j: find_rel_dist(embs[j, :], embs[n, :], measure=measure))[1:]
    return knn_arr

def get_neighborhood (embs, node, neighborhood):
    if (neighborhood.endswith("-NN")):
        k = int(neighborhood[:neighborhood.find("-NN")])
        return knn_embs(embs, node, k=k)

def dist_score (ti, zp, z, only_do=False, neighborhood="50-NN", dist_metric="lp", p=2):
    N_t = get_neighborhood(z, ti, neighborhood)
    if (dist_metric == "lp"):
        if (only_do):
            return (torch.sum(torch.linalg.norm(zp[0,:] - zp[1:,:], ord=p, dim=1))/len(N_t))
        else:
            Np_t = get_neighborhood(zp, ti, neighborhood)
            do_t = torch.sum(torch.linalg.norm(zp[ti,:] - zp[N_t,:], ord=p, dim=1))/len(N_t)
            dp_t = torch.sum(torch.linalg.norm(zp[ti,:] - zp[Np_t,:], ord=p, dim=1))/len(Np_t)
            return (do_t - dp_t).item()