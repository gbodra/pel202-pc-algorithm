from __future__ import print_function
from itertools import combinations, permutations
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from gsq.ci_tests import ci_test_bin, ci_test_dis


def estimate_skeleton(indep_test_func, data_matrix, alpha):
    node_ids = range(data_matrix.shape[1])
    node_size = data_matrix.shape[1]
    sep_set = [[set() for i in range(node_size)] for j in range(node_size)]

    g = nx.Graph()
    g.add_nodes_from(node_ids)
    for (i, j) in combinations(node_ids, 2):
        g.add_edge(i, j)

    l = 0
    while True:
        cont = False
        remove_edges = []
        for (i, j) in permutations(node_ids, 2):
            adj_i = list(g.neighbors(i))

            if j in adj_i:
                adj_i.remove(j)

            if len(adj_i) >= l:
                if len(adj_i) < l:
                    continue
                for k in combinations(adj_i, l):
                    p_val = indep_test_func(data_matrix, i, j, set(k))
                    if p_val > alpha:
                        if g.has_edge(i, j):
                            g.remove_edge(i, j)
                        sep_set[i][j] |= set(k)
                        sep_set[j][i] |= set(k)
                        break
                cont = True
        l += 1
        if cont is False:
            break

    return g, sep_set


def estimate_cpdag(skel_graph, sep_set):
    dag = skel_graph.to_directed()
    node_ids = skel_graph.nodes()

    for (i, j) in combinations(node_ids, 2):
        adj_i = set(dag.successors(i))

        if j in adj_i:
            continue

        adj_j = set(dag.successors(j))

        if i in adj_j:
            continue

        if sep_set[i][j] is None:
            continue

        common_k = adj_i & adj_j

        for k in common_k:
            if k not in sep_set[i][j]:

                if dag.has_edge(k, i):
                    dag.remove_edge(k, i)

                if dag.has_edge(k, j):
                    dag.remove_edge(k, j)

    def _has_both_edges(dag, i, j):
        return dag.has_edge(i, j) and dag.has_edge(j, i)

    def _has_any_edge(dag, i, j):
        return dag.has_edge(i, j) or dag.has_edge(j, i)

    old_dag = dag.copy()
    while True:
        for (i, j) in combinations(node_ids, 2):
            if _has_both_edges(dag, i, j):

                for k in dag.predecessors(i):

                    if dag.has_edge(i, k):
                        continue

                    if _has_any_edge(dag, k, j):
                        continue

                    dag.remove_edge(j, i)
                    break

            if _has_both_edges(dag, i, j):

                succs_i = set()
                for k in dag.successors(i):
                    if not dag.has_edge(k, i):
                        succs_i.add(k)

                preds_j = set()
                for k in dag.predecessors(j):
                    if not dag.has_edge(j, k):
                        preds_j.add(k)

                if len(succs_i & preds_j) > 0:
                    dag.remove_edge(j, i)

            if _has_both_edges(dag, i, j):

                adj_i = set()
                for k in dag.successors(i):
                    if dag.has_edge(k, i):
                        adj_i.add(k)

                for (k, l) in combinations(adj_i, 2):

                    if _has_any_edge(dag, k, l):
                        continue

                    if dag.has_edge(j, k) or (not dag.has_edge(k, j)):
                        continue

                    if dag.has_edge(j, l) or (not dag.has_edge(l, j)):
                        continue

                    dag.remove_edge(j, i)
                    break

        if nx.is_isomorphic(dag, old_dag):
            break
        old_dag = dag.copy()

    return dag


def mapping(data, feature):
    featureMap = dict()
    count = 0
    for i in sorted(data[feature].unique(), reverse=True):
        featureMap[i] = count
        count = count + 1
    data[feature] = data[feature].map(featureMap)
    return data


columns = ["Sex", "Lenght", "Diameter", "Height", "WholeWeight", "ShuckedWeight", "VisceraWeight", "ShellWeight",
           "Rings"]
columns_dict = dict(enumerate(columns))
data = pd.read_csv("./data/abalone.data", header=0)
data = mapping(data, "Sex")

data = data.to_numpy(dtype=int, na_value=0)

(g, sep_set) = estimate_skeleton(indep_test_func=ci_test_dis,
                                 data_matrix=data,
                                 alpha=0.01)
g = estimate_cpdag(skel_graph=g, sep_set=sep_set)

pos = nx.planar_layout(g)
nx.draw(g, pos, node_size=1000)
nx.draw_networkx_labels(g, pos, labels=columns_dict, font_size=6, font_weight="bold")
plt.show()

columns = ["Smoking", "YellowFingers", "Anxiety", "PeerPressure", "Genetics", "AttentionDisorder",
           "BornEvenDay", "CarAccident", "Fatigue", "Allergy", "Coughing", "LungCancer"]

columns_dict = dict(enumerate(columns))
data_trn = pd.read_csv("./data/lucas0_text/lucas0_train.data", sep=" ", header=None)
data_trn.drop(columns=11, inplace=True)
targets = pd.read_csv("./data/lucas0_text/lucas0_train.targets", sep=" ", header=None)
data_trn[11] = targets[0]
data_trn = mapping(data_trn, 11)

data_trn = data_trn.to_numpy(dtype=int, na_value=0)

(g, sep_set) = estimate_skeleton(indep_test_func=ci_test_bin,
                                 data_matrix=data_trn,
                                 alpha=0.01)
g = estimate_cpdag(skel_graph=g, sep_set=sep_set)

pos = nx.planar_layout(g)
nx.draw(g, pos, node_size=1000)
nx.draw_networkx_labels(g, pos, labels=columns_dict, font_size=6, font_weight="bold")
plt.show()
