#%%
import pandas as pd
import torch
import numpy as np
import random
import json
from collections import defaultdict
import scipy.sparse as sp

def load_data():
    file_path = './data/page8_movie_data'
    train_data = sp.load_npz(file_path+"/"+"val_coo_record.npz")
    test_data = sp.load_npz(file_path+"/"+"val_coo_record.npz")
    popularity_data = np.load(file_path+"/"+'popularity.npy')
    
    num_user = 37962
    num_item = 4819
    return train_data, test_data, popularity_data, num_user, num_item

def most_similar_row(
    mat: sp.coo_matrix,
    col_idx: list[int],
    metric: str = "dot",
    topk: int = 1,
):
    """여러 컬럼(col_idx)을 OR 연산으로 묶어 가장 비슷한 행 top-k 반환"""
    if sp.isspmatrix_coo(mat):        # 연산용 CSR 로 변환
        csr = mat.tocsr()
    else:
        csr = mat

    col_idx = np.asarray(col_idx, dtype=int)
    if col_idx.ndim != 1:
        raise ValueError("col_idx must be 1-D array of integers")

    # ── (1) 교집합 크기 ──────────────────────────────────────────
    # csr[:, col_idx] : rows × len(col_idx) 희소 행렬
    # getnnz(axis=1)  : 행마다 1의 개수 = |A ∩ B|
    intersection = csr[:, col_idx].getnnz(axis=1).astype(float)   # shape (rows,)

    if metric == "dot":          # 단순 교집합 개수
        scores = intersection

    else:
        row_nnz = csr.getnnz(axis=1).astype(float)
        col_nnz = float(len(col_idx))         # 선택 컬럼 수 (1 벡터 길이)

        if metric == "cosine":
            denom = np.sqrt(row_nnz) * np.sqrt(col_nnz)
            denom[denom == 0] = 1
            scores = intersection / denom

        elif metric == "jaccard":
            union = row_nnz + col_nnz - intersection
            union[union == 0] = 1
            scores = intersection / union
        else:
            raise ValueError("metric must be 'dot', 'jaccard', or 'cosine'")

    # ── (2) 상위 k개 행 추출 ────────────────────────────────────
    if topk == 1:
        best = int(scores.argmax())
        return best, np.array([scores[best]])

    topk_idx = np.argpartition(-scores, kth=topk - 1)[:topk]
    order    = np.argsort(-scores[topk_idx])
    return topk_idx[order][0], scores[topk_idx][order]
    
def get_items_for_user(data, user_id: int):
    """
    주어진 user_id에 대해 item 값이 1인 item index 목록을 반환합니다.
    """
    # coo_matrix의 row/col/data 사용
    row = data.row
    col = data.col
    data = data.data

    # 해당 user의 row 위치와 값이 1인 것만 추출
    mask = (row == user_id) & (data == 1)
    item_indices = col[mask]

    return item_indices.tolist()

def read_json(filepath):
    with open(filepath, 'r') as f:
        item_reindex = json.load(f)
        return {int(v): int(k) for k, v in item_reindex.items()}
def get_movie_mapping(filepath):
    movie_mapping = {}
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]
        for idx, line in enumerate(lines):
            parts = line.strip().split("::")
            movie_id = parts[0]
            movie_mapping[int(movie_id)] = parts[1]
    return movie_mapping
def get_movie_list(item_reindex_path, movie_path):
    item_reindex = read_json(item_reindex_path)
    
    movie_mapping = get_movie_mapping(movie_path)
    
    max_key = max(item_reindex.keys()) + 1
    item_index_lookup = {}
    for old, new in item_reindex.items():
        item_index_lookup[old] = {
            'title' : movie_mapping[new],
            'img' : f'{new}.jpg'
        }
    return item_index_lookup