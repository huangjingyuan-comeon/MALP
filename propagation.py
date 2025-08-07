import json
import sys
import numpy as np
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


# def find_top_similar_news(entities_list, target_index, top_n=10):
#     """
#     查找与目标新闻最相似的前top_n条新闻，并返回它们的相似度矩阵
#     :param entities_list: 二维实体列表
#     :param target_index: 目标新闻的索引
#     :param top_n: 需要返回的相似新闻数量
#     :return: (相似度矩阵numpy数组, 相关新闻索引列表)
#     """
#     # 构建二进制稀疏矩阵（与原始代码相同）
#     entity2idx = {}
#     idx_counter = 0
#     for entities in entities_list:
#         print(entities)
#         for entity in set(entities):
#             if entity not in entity2idx:
#                 entity2idx[entity] = idx_counter
#                 idx_counter += 1
#
#     rows, cols = [], []
#     for news_idx, entities in enumerate(entities_list):
#         unique_entities = list(set(entities))
#         indices = [entity2idx[e] for e in unique_entities]
#         rows.extend([news_idx] * len(indices))
#         cols.extend(indices)
#
#     binary_matrix = csr_matrix((np.ones(len(rows)), (rows, cols)),shape=(len(entities_list), len(entity2idx)))
#
#     # 计算目标新闻与其他新闻的共现实体数
#     target_vector = binary_matrix[target_index]
#     co_counts = binary_matrix.dot(target_vector.T).toarray().flatten()
#
#     # 排除自身并获取前top_n个索引
#     co_counts[target_index] = -1  # 屏蔽自身
#     sorted_indices = np.argsort(co_counts)[::-1]  # 降序排列
#     top_indices = sorted_indices[:top_n]
#
#     # 添加目标新闻到结果集
#     selected_indices = np.sort(np.append(top_indices, target_index))
#
#     # 提取子矩阵
#     sub_matrix = binary_matrix[selected_indices]
#
#
#
#     # 计算余弦相似度矩阵
#     similarity_matrix = cosine_similarity(sub_matrix)
#
#     # sub_matrix = torch.tensor(sub_matrix)
#     # emb_all = sub_matrix
#     # emb_all = emb_all   # N*d
#     # emb1    = torch.unsqueeze(emb_all,1) # N*1*d
#     # emb2    = torch.unsqueeze(emb_all,0) # 1*N*d
#     # W       = ((emb1-emb2)**2).mean(2)   # N*N*d -> N*N
#     # similarity_matrix   = torch.exp(-W/2)
#     return similarity_matrix, selected_indices


def build_co_occurrence_matrix(entities_list):
    """
    构建新闻实体共现矩阵
    :param entities_list: 二维列表，每个元素表示一条新闻的实体列表
    :return: 共现矩阵（scipy稀疏矩阵格式）
    """
    # 收集所有唯一实体并建立索引映射
    entity2idx = {}
    idx_counter = 0

    # 生成实体到索引的映射
    for entities in entities_list:
        for entity in set(entities):  # 先对实体去重
            if entity not in entity2idx:
                entity2idx[entity] = idx_counter
                idx_counter += 1

    # 初始化稀疏矩阵的构建参数
    num_news = len(entities_list)
    num_entities = len(entity2idx)

    # 创建行索引、列索引和数据
    rows, cols = [], []
    for news_idx, entities in enumerate(entities_list):
        unique_entities = list(set(entities))  # 去重处理
        indices = [entity2idx[e] for e in unique_entities]
        rows.extend([news_idx] * len(indices))
        cols.extend(indices)

    data = np.ones(len(rows), dtype=np.uint8)

    # 构建二进制稀疏矩阵
    binary_matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(num_news, num_entities),
        dtype=np.uint8
    )

    # 计算共现矩阵（矩阵乘法）
    co_occurrence = binary_matrix.dot(binary_matrix.T)

    max_val = co_occurrence.max()
    normalized = co_occurrence.astype(float) / max_val


    return normalized


# 示例用法

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity



def find_top_similar_news(entities_list, target_index, top_n=15):
    """
    查找与目标新闻最相似的前top_n条新闻，并返回它们的相似度矩阵
    :param entities_list: 二维实体列表
    :param target_index: 目标新闻的索引
    :param top_n: 需要返回的相似新闻数量
    :return: (相似度矩阵numpy数组, 相关新闻索引列表)
    """
    # 构建二进制稀疏矩阵（与原始代码相同）
    entity2idx = {}
    idx_counter = 0
    for entities in entities_list:
        # print(entities)
        for entity in entities:
            if entity not in entity2idx:
                entity2idx[entity] = idx_counter
                idx_counter += 1

    with open("entity.json","w",encoding="UTF-8") as w:
        json.dump(entity2idx,w,indent=4,ensure_ascii=False)

    rows, cols = [], []
    for news_idx, entities in enumerate(entities_list):
        unique_entities = list(set(entities))
        indices = [entity2idx[e] for e in unique_entities]
        rows.extend([news_idx] * len(indices))
        cols.extend(indices)

    binary_matrix = csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(len(entities_list), len(entity2idx)))

    # 计算目标新闻与其他新闻的共现实体数
    target_vector = binary_matrix[target_index]
    co_counts = binary_matrix.dot(target_vector.T).toarray().flatten()

    filtered = [x for x in co_counts if x != 0]
    # second_num =1
    if len(filtered) < 5:
        second_num=0
    else:
        second_num=1


    # 排除自身并获取前top_n个索引
    co_counts[target_index] = -1  # 屏蔽自身
    sorted_indices = np.argsort(co_counts)[::-1]  # 降序排列
    top_indices = sorted_indices[:top_n]

    # 添加目标新闻到结果集
    selected_indices = np.sort(np.append(top_indices, target_index))

    # 提取子矩阵
    sub_matrix = binary_matrix[selected_indices]

    # 计算余弦相似度矩阵
    similarity_matrix = cosine_similarity(sub_matrix)

    return similarity_matrix, selected_indices,second_num


def build_co_occurrence_matrix(entities_list):
    """
    构建新闻实体共现矩阵
    :param entities_list: 二维列表，每个元素表示一条新闻的实体列表
    :return: 共现矩阵（scipy稀疏矩阵格式）
    """
    # 收集所有唯一实体并建立索引映射
    entity2idx = {}
    idx_counter = 0

    # 生成实体到索引的映射
    for entities in entities_list:
        for entity in set(entities):  # 先对实体去重
            if entity not in entity2idx:
                entity2idx[entity] = idx_counter
                idx_counter += 1

    # 初始化稀疏矩阵的构建参数
    num_news = len(entities_list)
    num_entities = len(entity2idx)

    # 创建行索引、列索引和数据
    rows, cols = [], []
    for news_idx, entities in enumerate(entities_list):
        unique_entities = list(set(entities))  # 去重处理
        indices = [entity2idx[e] for e in unique_entities]
        rows.extend([news_idx] * len(indices))
        cols.extend(indices)

    data = np.ones(len(rows), dtype=np.uint8)

    # 构建二进制稀疏矩阵
    binary_matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(num_news, num_entities),
        dtype=np.uint8
    )

    # 计算共现矩阵（矩阵乘法）
    co_occurrence = binary_matrix.dot(binary_matrix.T)
    max_val = co_occurrence.max()
    normalized = co_occurrence.astype(float) / max_val

    return normalized


def label_propagation(entities,results,num):

    sim_matrix, indices,second_num = find_top_similar_news(entities, 6900, top_n=num)
    # print("相关新闻索引:", indices)
    # print("相似度矩阵:")
    # print(np.round(sim_matrix, 2))
    # print(second_num)
    if second_num == 0:
        return 0,0,0

    s_labels = []


    for id in indices[:num]:
        # print(entities[id])
        # print(results[id]['label'])
        if results[id]['label'] == "real":
            s_labels.append([1, 0])
        else:
            s_labels.append([0, 1])
    s_labels = torch.tensor(s_labels)

    # 计算相似度
    W = torch.tensor(np.round(sim_matrix, 2))

    for id in range(0, W.shape[0]):
        W[id][id] = 0
    # print(W)

    eps = np.finfo(float).eps

    query = torch.tensor([[0, 0], [0, 0]])
    N = num+1
    alpha = 0.9
    num_classes = s_labels.shape[1]
    num_support = int(s_labels.shape[0] / num_classes)
    # num_queries = int(query.shape[0] / num_classes)

    # 选择前k个
    # topk, indices = torch.topk(W, 2)
    # mask = torch.zeros_like(W)
    # mask = mask.scatter(1, indices, 1)
    # mask = ((mask + torch.t(mask)) > 0).type(torch.float32)  # union, kNN graph
    # # mask = ((mask>0)&(torch.t(mask)>0)).type(torch.float32)  # intersection, kNN graph
    # W = W * mask
    # print(W)

    # normalize
    D = W.sum(0)
    D_sqrt_inv = torch.sqrt(1.0 / (D + eps))
    D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, N)
    D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(N, 1)
    S = D1 * W * D2
    # print(S)

    # label
    ys = s_labels
    # print(f"ys shape:{ys.shape}")
    # yu = torch.zeros(num_classes * num_queries, num_classes)
    # yu = torch.zeros(num_queries, num_classes)
    yu = torch.tensor([[0,0]])
    # print(f"yu shape:{yu.shape}")
    y = torch.cat((ys, yu), 0)
    # 转换 S 和 y 为 Float 类型
    S = S.float()  # 或 S = S.type(torch.FloatTensor)
    y = y.float()
    # 确保 alpha 和 eps 是 float32 标量
    alpha = float(alpha)  # 如果 alpha 是 int 或 float64
    eps = float(eps)

    F = torch.matmul(torch.inverse(torch.eye(N) - alpha * S + eps), y)
    # print(F)
    Fq = F[num :, :]  # query predictions
    # print(Fq)
    # print(s_labels)
    return Fq,s_labels,second_num


# 示例用法
if __name__ == "__main__":
    entities = []
    with open(f"train_entity.json","r",encoding="UTF-8") as r:
        train_results = json.load(r)

    for result in train_results:
        entities.append(result['entity'])

    with open(f"test_entity_datasets.json","r",encoding="UTF-8") as r1:
        test_results = json.load(r1)

    with open(f"zh_test_result.json","r",encoding="UTF-8") as r2:
        test_predict_results=json.load(r2)


    test_entities = entities
    id = 1
    samples = []
    for id,sample in zip(range(1,len(test_results)+1),test_results):


        sample['entity'] = sample['entity']

        test_entities.append(sample['entity'])


        Fq,s_labels,second_num = label_propagation(test_entities,train_results,6)

        test_entities=test_entities[:-1]



        sample["re_predict"] = sample['predict']
        if second_num != 0:

            # print(float(Fq[0][0]))
            # print(float(Fq[0][1]))
            if sample['predict']==0:
                if float(Fq[0][0]) ==0 or float(Fq[0][1]/Fq[0][0])>=3:
                    sample["re_predict"] = 1
                    print(f"第{id}个样本")
                    print(f"新闻内容:{sample['content']}")
                    print(f"新闻实体：{sample['entity']}")
                    print(f"新闻真实标签：{sample['label']}")
                    print(f"新闻预测标签：{sample['predict']}")
                    print(Fq)
                    print(f"修改{id}条")
            if sample['predict']==1:
                if float(Fq[0][1])==0 or float(Fq[0][0]/Fq[0][1])>=3:
                    sample["re_predict"] = 0
                    print(f"第{id}个样本")
                    print(f"新闻内容:{sample['content']}")
                    print(f"新闻实体：{sample['entity']}")
                    print(f"新闻真实标签：{sample['label']}")
                    print(f"新闻预测标签：{sample['predict']}")
                    print(Fq)
                    print(f"修改{id}条")
        samples.append(sample)


with open(f"zh_test_prediction.json","w",encoding="UTF-8") as w:
    json.dump(samples,w,indent=4,ensure_ascii=False)








