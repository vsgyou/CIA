#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter

# import data
import pages.page8_model as model
import pages.page8_utils as utils
import pages.page8_candidate_generator as cg
# import config.const as const_util
from scipy.sparse import load_npz

import os
import pickle
import numpy as np
import dgl
import scipy.sparse as sp



class Recommender(object):

    def __init__(self, flags_obj):
        self.flags_obj = flags_obj
        self.set_device()
        self.set_model()
        self.topk = flags_obj['topk']

    def set_device(self):

        self.device  = 'cpu'

    def set_model(self):

        raise NotImplementedError

    def transfer_model(self):

        self.model = self.model.to(self.device)
        
    def filter_history(self, items, train_pos):

        return np.stack([items[i][np.isin(items[i], np.array(train_pos), invert=True)][:self.topk] for i in range(len(items))], axis=0)

    def load_ckpt(self, model_load_path):
        self.model.load_state_dict(torch.load(model_load_path, map_location=torch.device('cpu')))

    def inference(self, sample):

        raise NotImplementedError

    def make_cg(self):

        raise NotImplementedError

    def cg(self, users, topk):

        raise NotImplementedError


class MFRecommender(Recommender):

    def __init__(self, flags_obj):

        super(MFRecommender, self).__init__(flags_obj)

    def set_model(self):

        self.model = model.MF(self.n_user, self.n_item, self.embedding_size)

    def make_cg(self):

        self.item_embeddings = self.model.get_item_embeddings()
        self.generator = cg.FaissInnerProductMaximumSearchGenerator(self.flags_obj, self.item_embeddings)

        self.user_embeddings = self.model.get_user_embeddings()

    def cg(self, users):

        return self.generator.generate(self.user_embeddings[users], self.topk)

class PopularityRecommender(Recommender):

    def __init__(self, flags_obj):

        super(PopularityRecommender, self).__init__(flags_obj)

    def set_model(self):

        pass

    def transfer_model(self):

        pass

    def load_ckpt(self, epoch):

        pass

    def make_cg(self, popularity_data):
        self.generator = cg.PopularityGenerator(self.flags_obj, popularity_data, 500)

    def cg(self, users):

        return self.generator.generate(users, self.topk) 


class IPSRecommender(MFRecommender):

    def __init__(self, flags_obj):
        self.n_user = flags_obj['n_user']
        self.n_item = flags_obj['n_item']
        self.embedding_size = 128

        super(IPSRecommender, self).__init__(flags_obj)

    def make_faiss_db(self):
        self.generator = cg.FaissInnerProductMaximumSearchGenerator(self.flags_obj, self.item_embeddings, index_path)
    
    def cg(self, users, train_pos):
        items = self.generator.generate(self.user_embeddings[users], self.topk)
        return self.filter_history(items, train_pos)


class DICERecommender(MFRecommender):

    def __init__(self, flags_obj):

        self.n_user = flags_obj['n_user']
        self.n_item = flags_obj['n_item']
        self.embedding_size = flags_obj['embedding_size']
        super(DICERecommender, self).__init__(flags_obj)
        
    def set_model(self):
        self.model = model.DICE(self.n_user, self.n_item, self.embedding_size)
    
    def cg(self, users, train_pos):
        items = self.generator.generate(self.user_embeddings[users], self.topk)
        return self.filter_history(items, train_pos)


