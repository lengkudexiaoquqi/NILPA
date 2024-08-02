# 真？？？


# 真
import copy
import math
import random
import time

from networkx.algorithms import community
import numpy as np
from numpy.linalg import norm
from NetworkLoader import NetworkLoader
from sklearn import metrics
import community
import pandas as pd
import networkx as nx
import pylab
import optuna

from read_BlogCatalog import read_BlogCatalog
from read_DBLP import read_DBLP
from read_actor import read_actor
from read_citeseer_4230 import read_Citeseer4230
from read_photo import get_Amazon_photo
from read_pubmed import read_pubmed


class XJQV0724():
    def __init__(self, G, content_matrix, alpha,  n):
        self.G = G
        self.content_matrix = content_matrix

        self.alpha = alpha
        # self.beita = beita
        self.n = n

    def make_frag(self):
        for node in self.G.nodes():
            self.G.nodes[node]['frag'] = 0

    def if_node_neighbor(self, node1, node2):
        node1_nei = nx.neighbors(self.G, node1)
        if node2 in node1_nei:
            return True
        else:
            return False

    def content_similarity(self, node1, node2):

        x = np.array(self.content_matrix[node1, :])
        y = np.array(self.content_matrix[node2, :])
        result1 = x.dot(y)
        x_sum = np.sum(x)
        y_sum = np.sum(y)
        result2 = x_sum + y_sum - result1
        if result2 == 0:
            result2 = 1
        return result1 / result2
# 相似度指标
    def jaccard_Similarity(self, node1, node2):

        node1_neighbors = set(nx.neighbors(self.G, node1))
        node1_neighbors.add(node1)
        node2_neighbors = set(nx.neighbors(self.G, node2))
        node2_neighbors.add(node2)
        intersection = len(list(node1_neighbors.intersection(node2_neighbors)))  # 交集
        union = len(list(node1_neighbors.union(node2_neighbors)))  # 并集
        if union == 0:
            union = 1
        return intersection / union

    def Salton_Similarity(self, node1, node2):

        k_node1 = self.G.degree(node1)
        k_node2 = self.G.degree(node2)
        node1_neighbors = set(nx.neighbors(self.G, node1))
        node2_neighbors = set(nx.neighbors(self.G, node2))
        intersection = len(list(node1_neighbors.intersection(node2_neighbors)))
        return intersection / math.sqrt(k_node1 * k_node2)

    def HDI_Similarity(self, node1, node2):

        k_node1 = self.G.degree(node1)
        k_node2 = self.G.degree(node2)
        node1_neighbors = set(nx.neighbors(self.G, node1))
        node2_neighbors = set(nx.neighbors(self.G, node2))
        intersection = len(list(node1_neighbors.intersection(node2_neighbors)))
        max_num = max(k_node1, k_node2)
        return intersection / max_num

    def HPI_Similarity(self, node1, node2):

        k_node1 = self.G.degree(node1)
        k_node2 = self.G.degree(node2)
        node1_neighbors = set(nx.neighbors(self.G, node1))
        node2_neighbors = set(nx.neighbors(self.G, node2))
        intersection = len(list(node1_neighbors.intersection(node2_neighbors)))
        min_num = min(k_node1, k_node2)
        return intersection / min_num


    def LHN_Similarity1(self, node1, node2):

        k_node1 = self.G.degree(node1)
        k_node2 = self.G.degree(node2)
        node1_neighbors = set(nx.neighbors(self.G, node1))
        node2_neighbors = set(nx.neighbors(self.G, node2))
        intersection = len(list(node1_neighbors.intersection(node2_neighbors)))
        return intersection / (k_node1 * k_node2)

    def AA_Similarity(self, node1, node2):

        node1_neighbors = list(nx.neighbors(self.G, node1))
        node2_neighbors = list(nx.neighbors(self.G, node2))
        total_neighbors = copy.deepcopy(node1_neighbors)
        total_neighbors.extend(node2_neighbors)
        sum = 0
        for node in total_neighbors:
            if math.log(self.G.degree(node)) == 0:
                continue
            else:
                sum = sum + 1/math.log(self.G.degree(node))
        return sum

    def LHN_Similarity2(self, node1, node2):
        k_node1 = self.G.degree(node1)
        k_node2 = self.G.degree(node2)
        sum = k_node1 + k_node2
        node1_neighbors = set(nx.neighbors(self.G, node1))
        node2_neighbors = set(nx.neighbors(self.G, node2))
        intersection = len(list(node1_neighbors.intersection(node2_neighbors)))
        return intersection/sum






# 相似度指标

    def get_Sim_Matrix(self):
        # print("len(G)", len(self.G.nodes))
        simMatrix = np.zeros((len(self.G.nodes), len(self.G.nodes)))
        for node1 in self.G.nodes:
            for node2 in self.G.nodes:
                if self.if_node_neighbor(node1, node2):
                    simMatrix[node1][node2] = self.content_similarity(node1,node2) + self.alpha * self.jaccard_Similarity(
                        node1, node2)
        return simMatrix

    def get_Sim(self, node1, node2):
        return self.content_similarity(node1, node2) + self.alpha * self.jaccard_Similarity(node1, node2)

    def get_centrality(self, W):  # 找到所有的矩阵的行和
        sum_list = list(np.sum(W, axis=0))
        node_labels = list(range(0, len(W)))
        return dict(zip(node_labels, sum_list))

    def find_error(self, node_label_dic):
        _dict = {}
        for key, value in node_label_dic.items():
            if value not in _dict.keys():
                _dict[value] = []
            _dict[value].append(key)
        for key, value in _dict.items():
            print(key, value)
        return _dict

    def find_error_1(self, node_label_dic):
        _dict = {}
        for key, value in node_label_dic.items():
            if value not in _dict.keys():
                _dict[value] = []
            _dict[value].append(key)

        return _dict

    def convertCommunities2Dic(self, community_list):
        node_label_dic = {}
        nodeLabelASCII = 1
        for community in community_list:
            for node in community:
                node_label_dic[node] = nodeLabelASCII
            nodeLabelASCII = nodeLabelASCII + 1
        return node_label_dic

    def get_node_second_rank_dict(self):
        node_second_rank_dict = {}
        for node in self.G.nodes:
            di = 0
            node_nei = list(nx.neighbors(self.G, node))
            for node1 in node_nei:
                for node2 in nx.neighbors(self.G, node1):
                    if node2 != node and node2 not in node_nei:
                        node_nei.append(node2)
            node_second_list = list(set(node_nei))
            for true_node in node_second_list:
                di = di + self.G.degree(true_node)
            node_second_rank_dict[node] = di
        return node_second_rank_dict

    def get_node_beta(self,node,node_Sim_dict):
        node_di = node_Sim_dict[node]
        return node_di

    def find_max_label(self, node, lpa_dict, node_vote_ability, node_Sim_dict):
        node_nei = nx.neighbors(self.G, node)

        dict = {}
        for fnode in node_nei:
            if lpa_dict[fnode] not in dict:
                dict[lpa_dict[fnode]] = node_vote_ability[fnode] * self.get_Sim(node, fnode) / self.G.degree(fnode)
            else:
                dict[lpa_dict[fnode]] = dict[lpa_dict[fnode]] + node_vote_ability[fnode] * self.get_Sim(node,
                                                                                                        fnode) / self.G.degree(
                    fnode)
        node_second_list = []
        for node1 in nx.neighbors(self.G, node):
            for node2 in nx.neighbors(self.G, node1):
                if node2 != node and node2 not in nx.neighbors(self.G, node):
                    node_second_list.append(node2)
        # node_second_list = list(set(node_second_list))
        for snode in node_second_list:
            beta = self.get_node_beta(snode,node_Sim_dict)
            if lpa_dict[snode] not in dict:
                dict[lpa_dict[snode]] = beta * node_vote_ability[snode] * self.get_Sim(node,
                                                                                             snode) / self.G.degree(
                    snode)
            else:
                dict[lpa_dict[snode]] = dict[lpa_dict[snode]] + beta * node_vote_ability[snode] * self.get_Sim(
                    node, snode) / self.G.degree(snode)
        if len(dict) == 0:
            return node
        else:
            return max(dict, key=dict.get)

    def find_vote_ability(self, node_centrality_dic):
        kMax = max(node_centrality_dic.values())
        dict_result = {}
        for node in self.G.nodes:
            kv = node_centrality_dic[node]
            dict_result[node] = math.log(kv / kMax + 1)
        return dict_result

    def lpa_wight(self, node_vote_ability, Max,node_Sim_dict):
        lpa_dict = {}
        for node in self.G.nodes:
            lpa_dict[node] = node
        lpa_dict_copy = {}
        sum = 0
        while lpa_dict != lpa_dict_copy or sum < Max:
            lpa_dict_copy = copy.deepcopy(lpa_dict)
            test = dict(sorted(node_vote_ability.items(), key=lambda x: x[1], reverse=True))
            node_list = list(test.keys())
            for node in node_list:
                lpa_dict[node] = self.find_max_label(node, lpa_dict, node_vote_ability,node_Sim_dict)
            # print("第", sum, "轮完毕")
            sum = sum + 1
        return lpa_dict

    def get_new_graph_list(self, node_label_dic):
        graph_list = []
        for value in list(node_label_dic.values()):
            graph = nx.Graph()

            for node in value:
                graph.add_node(node)
            graph_list.append(graph)
        return graph_list

    # 找到索引
    def find_community_index(self, community, community_list):
        index = 0
        for community_true in community_list:
            if list(community.nodes)[0] in community_true:
                return index
            index = index + 1


    def get_dubious_node_list(self, already_lpa_graph_list):
        dubious_node_list = []
        for community in already_lpa_graph_list:
            if len(community.nodes) <= self.n:
                for node in community.nodes:
                    dubious_node_list.append(node)

        return list(set(dubious_node_list))

    def get_surplus_commmunity_list(self, dubious_node_list, graph_list):
        del_index_list = []
        for node in dubious_node_list:
            for index in range(len(graph_list)):
                if node in graph_list[index]:
                    del_index_list.append(index)
        del_index_list = list(set(del_index_list))

        surplus_commmunity_list = []
        for i in range(len(graph_list)):
            if i not in del_index_list:
                surplus_commmunity_list.append(graph_list[i])

        return surplus_commmunity_list

    # 看看要不要加长度都试试
    def find_node_community_sim(self, community, node1):
        sum = 0
        for node2 in community:
            if node1 != node2:
                sum = sum + self.get_Sim(node1, node2)
        num = len(community.nodes)
        if num == 0:
            num = 1
        return sum / num

    def find_node_community_sim1(self, community, node1):
        sum = 0
        for fnode1 in nx.neighbors(self.G,node1):
            if fnode1 in community:
                sum = sum + 1
        return sum


    def find_node_community_sim2(self,community,node1):
        sum = 0
        for fnode1 in nx.neighbors(self.G,node1):
            if fnode1 in community:
                sum = sum + 1
        return sum

    def find_node_community_sim3(self,community, node1):
        sum = 0
        num = len(list(nx.neighbors(self.G,node1)))
        for fnode1 in nx.neighbors(self.G,node1):
            if fnode1 in community:
                sum = self.get_Sim(node1,fnode1) + sum
        if num == 0:
            num = 1
        return sum / num


    def find_community_boundary_node_list(self, community):
        community_boundary_node_list = []
        for node in community:
            node_neighbors = nx.neighbors(self.G, node)
            for node_neighbor in node_neighbors:
                if node_neighbor not in community:
                    community_boundary_node_list.append(node)
        return list(set(community_boundary_node_list))

    def find_all_community_boundary_node_list(self, community_list):
        dubious_node = []
        for community in community_list:
            community_boundary_node_list = self.find_community_boundary_node_list(community)
            dubious_node.extend(community_boundary_node_list)
        return list(set(dubious_node))

    def deal_dubious_node_list(self, dubious_node_list, surplus_commmunity_list, node_vote_ability):
        dubious_node_list_copy = copy.deepcopy(dubious_node_list)
        dubious_node_dict = self.find_check_dubious_node_dict(dubious_node_list_copy, node_vote_ability)
        while dubious_node_dict:
            node = max(dubious_node_dict, key=lambda x: dubious_node_dict[x])
            del dubious_node_dict[node]
            max_sim = -1
            max_index = -1
            for index in range(len(surplus_commmunity_list)):
                sim_current = self.find_node_community_sim(surplus_commmunity_list[index], node)
                if sim_current > max_sim:
                    max_index = index
                    max_sim = sim_current


            surplus_commmunity_list[max_index].add_node(node)
        return surplus_commmunity_list

    def find_node_community_index(self, node, community_list):
        index = 0
        for community in community_list:
            if node in community:
                return index
            index = index + 1

    def find_max_index_community(self, node, community_list):
        index_true = -1
        MsxSim = -10000
        for index, community in zip(range(len(community_list)), community_list):
            Sim_score = self.find_node_community_sim(community, node)
            if Sim_score > MsxSim:
                MsxSim = Sim_score
                index_true = index
        return index_true

    def boundary_recheck(self, cluster_graph_list):
        cluster_graph_list_copy = copy.deepcopy(cluster_graph_list)
        dubious_node_list = self.find_all_community_boundary_node_list(cluster_graph_list_copy)
        while dubious_node_list:
            node = random.choice(dubious_node_list)
            self.G.nodes[node]['frag'] = 1
            dubious_node_list.remove(node)
            index_current = self.find_node_community_index(node, cluster_graph_list_copy)
            index_true = self.find_max_index_community(node, cluster_graph_list_copy)
            if index_current != index_true:
                cluster_graph_list_copy[index_current].remove_node(node)
                cluster_graph_list_copy[index_true].add_node(node)
            for node1 in nx.neighbors(self.G, node):
                if self.G.nodes[node1]['frag'] == 0 and node1 not in dubious_node_list:
                    dubious_node_list.append(node1)

        return cluster_graph_list_copy

    def find_check_dubious_node_dict(self, dubious_node_list, node_vote_ability):
        dubious_node_dict = {}
        for node in dubious_node_list:
            dubious_node_dict[node] = node_vote_ability[node]
        return dubious_node_dict

    def boundary_recheck2(self, cluster_graph_list, node_vote_ability):
        cluster_graph_list_copy = copy.deepcopy(cluster_graph_list)
        dubious_node_list = self.find_all_community_boundary_node_list(cluster_graph_list_copy)
        while dubious_node_list:
            dubious_node_dict = self.find_check_dubious_node_dict(dubious_node_list, node_vote_ability)
            node = max(dubious_node_dict, key=lambda x: dubious_node_dict[x])
            self.G.nodes[node]['frag'] = 1
            dubious_node_list.remove(node)
            index_current = self.find_node_community_index(node, cluster_graph_list_copy)
            index_true = self.find_max_index_community(node, cluster_graph_list_copy)
            if index_current != index_true:
                cluster_graph_list_copy[index_current].remove_node(node)
                cluster_graph_list_copy[index_true].add_node(node)
                for node1 in nx.neighbors(self.G, node):
                    if self.G.nodes[node1]['frag'] == 0 and node1 not in dubious_node_list:
                        dubious_node_list.append(node1)

        return cluster_graph_list_copy

    def community_detection(self):
        self.make_frag()
        # print("alpha", self.alpha)
        # print("beita", self.beita)
        # print("n", self.n)
        Sim_Matrix = self.get_Sim_Matrix()

        node_centrality_dic = self.get_centrality(Sim_Matrix)
        # print(node_centrality_dic)
        node_vote_ability = self.find_vote_ability(node_centrality_dic)
        # print(node_vote_ability)
        # node_second_rank_dict = self.get_node_second_rank_dict()
        # print("节点中心性计算完毕")
        node_label_dic = self.lpa_wight(node_vote_ability, 100,node_vote_ability)
        # print("第一阶段")
        node_list_dict = self.find_error_1(node_label_dic)

        already_lpa_graph_list = self.get_new_graph_list(node_list_dict)

        dubious_node_list = self.get_dubious_node_list(already_lpa_graph_list)
        # print("已经找到模糊节点")
        surplus_commmunity_list = self.get_surplus_commmunity_list(dubious_node_list, already_lpa_graph_list)

        cluster_graph_list = self.deal_dubious_node_list(dubious_node_list, surplus_commmunity_list, node_vote_ability)
        # print("第二阶段")


        cluster_graph_list_checked1 = self.boundary_recheck2(cluster_graph_list, node_vote_ability)
        # cluster_graph_list_checked2 = self.boundary_recheck2(cluster_graph_list_checked1, node_vote_ability)
        # node_label_dic1 = self.convertCommunities2Dic(cluster_graph_list_checked1)
        # print("第三阶段")
        # self.find_error(node_label_dic1)


        # print(result)
        # for node in nx.neighbors(self.G,0):
        # 	print(self.jaccard_similarity(0,node))
        # 	print(self.G.degree(0))
        # self.find_error(node_list_dict_new)
        return cluster_graph_list_checked1


def evaluate(label_dict, node_community_label_list):
    predicted_label_list = []
    true_label_list = []
    for key, value in label_dict.items():
        predicted_label_list.append(label_dict[key])
        true_label_list.append(node_community_label_list[key])

    nmi = metrics.normalized_mutual_info_score(true_label_list, predicted_label_list)
    modularity = community.modularity(label_dict, graph)
    ari = metrics.adjusted_rand_score(true_label_list, predicted_label_list)
    return nmi, modularity, ari, predicted_label_list

def evaluate_nmi(label_dict, node_community_label_list):
    predicted_label_list = []
    true_label_list = []
    for key, value in label_dict.items():
        predicted_label_list.append(label_dict[key])
        true_label_list.append(node_community_label_list[key])

    nmi = metrics.normalized_mutual_info_score(true_label_list, predicted_label_list)

    return nmi
def evaluate_ari(label_dict, node_community_label_list):
    predicted_label_list = []
    true_label_list = []
    for key, value in label_dict.items():
        predicted_label_list.append(label_dict[key])
        true_label_list.append(node_community_label_list[key])

    ari = metrics.adjusted_rand_score(true_label_list, predicted_label_list)

    return ari

def draw_alpha_beita_nmi(data):
    # 将数据转换为 DataFrame
    df = pd.DataFrame(data, columns=['alpha', 'beita', 'nmi'])
    df = df.pivot(index='alpha', columns='beita', values='nmi')
    # 将 DataFrame 写入 CSV 文件
    df.to_csv('data.csv')

# def alpha_nmi(data):
#     data

def objective(trial):
    # nmi
    global node_community_label_list
    global graph
    global content_matrix
    # 定义三个超参数的搜索范围
    alpha = trial.suggest_float('alpha', 1, 10)
    # beta = trial.suggest_float('beta', 0, 1)
    n = trial.suggest_int('n', 1, 2)
    xjq0717 = XJQV0724(graph, content_matrix, alpha,  n)
    predicted_label_dict = xjq0717.community_detection()
    nmi = evaluate_nmi(predicted_label_dict, node_community_label_list)
    return nmi


def objective1(trial):
    # nmi
    global node_community_label_list
    global graph
    global content_matrix
    # 定义三个超参数的搜索范围
    alpha = trial.suggest_float('alpha', 0.05  , 0.1)
    # alpha = trial.suggest_float('alpha', 0.01, 0.23)
    # beta = trial.suggest_float('beta', 0, 1)
    n = trial.suggest_int('n', 3, 3)
    xjq0717 = XJQV0724(graph, content_matrix, alpha,  n)
    predicted_label_dict = xjq0717.community_detection()
    nmi = evaluate_nmi(predicted_label_dict, node_community_label_list)
    return nmi

# def objective2(trial):
#     # ari
#     global node_community_label_list
#     global graph
#     global content_matrix
#     # 定义三个超参数的搜索范围
#     alpha = trial.suggest_float('alpha', 0, 0.2)
#     # beta = trial.suggest_float('beta', 0, 0.5)
#     n = trial.suggest_int('n', 1, 7)
#     xjq0717 = XJQV0724(graph, content_matrix, alpha,  n)
#     predicted_label_dict = xjq0717.community_detection()
#     ari = evaluate_ari(predicted_label_dict, node_community_label_list)
#     return ari
# #
if __name__ == '__main__':

    # datasets = ['cornell', 'texas', 'wisconsin', 'washington']
    # datasets = ['art_network5','art_network6','art_network7','art_network8']
    # datasets = ['cornell', 'texas', 'wisconsin', 'washington','cora','citeseer']
    # datasets = ['art_network5','art_network6','art_network7','art_network8']
    datasets = ['wisconsin']
    for i in range(1):
        data = []
        for item in datasets:
            test_alpha = [0.00379108127408624]
            # test_alpha = list(np.arange(0, 1, 0.1))
            # test_beita = list(np.arange(0.01, 0.1, 0.01))
            # test_beita = [0.306171877950806]
            # n_list = list(np.arange(1, 33, 1))
            n_list = [1]

            for alpha in test_alpha:
                # for beita in test_beita:
                    for n in n_list:
                        print("alpha:",alpha,"n:",n)
                        print(item)
                        loader = NetworkLoader(item)
                        adjajency_matrix, content_matrix, node_community_label_list, edge_list \
                            = loader.network_parser('data/' + item)
                        # print(edge_list)
                        graph = nx.Graph(edge_list)

                        # print(edge_list)
                        # graph = nx.Graph(edge_list)

                        # print(graph.nodes)
                        start_time = time.time()
                        # print(start_time)
                        xjq0717 = XJQV0724(graph, content_matrix, alpha, n)
                        predicted_label_dict = xjq0717.community_detection()
                        # print(predicted_label_dict)
                        # for i in predicted_label_dict:
                        # # 	print(i)
                        # 	print(i.nodes)
                        #
                        # nmi, modularity, ari, predicted_label_list = evaluate(predicted_label_dict,
                        #                                                       node_community_label_list)
                        # print("modularity:", modularity, "nmi:", nmi, "ari:", ari)
                        # print("nmi:", nmi)

                        end_time = time.time()
                        # print(end_time)
                        execution_time = end_time - start_time
                        print("执行时间：", execution_time, "秒")



# if __name__ == '__main__':
#
#     # datasets = ['cornell', 'texas', 'washington', 'wisconsin']
#     # datasets = ['art_network1','art_network2','art_network3','art_network4']
#     # datasets = ['art_network5','art_network6','art_network7','art_network8']
#     # datasets = ['art_network51','art_network52','art_network53','art_network54', 'art_network55']
#     # datasets = ['art_network4','art_network5','art_network6','art_network7']
#     datasets = ['art_network6']
#
#     for item in datasets:
#         print(item)
#         loader = NetworkLoader(item)
#         adjajency_matrix, content_matrix, node_community_label_list, edge_list \
#             = loader.network_parser('data/' + item)
#         graph = nx.Graph(edge_list)
#         study = optuna.create_study(direction='maximize')
#         study.optimize(objective1, n_trials=1000)
#         print('Best trial:')
#         trial = study.best_trial
#         print('Value: ', trial.value)
#         print('Params: ')
#         for key, value in trial.params.items():
#             print(f'    {key}: {value}')
