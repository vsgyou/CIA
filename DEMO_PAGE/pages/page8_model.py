import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.nn.parameter import Parameter

class MF(nn.Module):

    def __init__(self, num_users, num_items, embedding_size):

        super(MF, self).__init__()

        self.users = Parameter(torch.FloatTensor(num_users, embedding_size))
        self.items = Parameter(torch.FloatTensor(num_items, embedding_size))

    def get_item_embeddings(self):

        return self.items.detach().cpu().numpy().astype('float32').copy()

    def get_user_embeddings(self):

        return self.users.detach().cpu().numpy().astype('float32').copy()


class DICE(nn.Module):

    def __init__(self, num_users, num_items, embedding_size):

        super(DICE, self).__init__()

        self.users_int = Parameter(torch.FloatTensor(num_users, embedding_size))
        self.users_pop = Parameter(torch.FloatTensor(num_users, embedding_size))
        self.items_int = Parameter(torch.FloatTensor(num_items, embedding_size))
        self.items_pop = Parameter(torch.FloatTensor(num_items, embedding_size))

    def forward(self, user, item):

        users_int = self.users_int[user]
        users_pop = self.users_pop[user]
        items = self.items_int[item]

        score_int = torch.sum(users_int*items_int, 2)
        score_pop = torch.sum(users_pop*items_pop, 2)

        score_total = score_int + score_pop
        
        return score_total

    def get_item_embeddings(self):

        item_embeddings = torch.cat((self.items_int, self.items_pop), 1)
        return item_embeddings.detach().cpu().numpy().astype('float32')

    def get_user_embeddings(self):

        user_embeddings = torch.cat((self.users_int, self.users_pop), 1)
        #user_embeddings = self.users_pop
        return user_embeddings.detach().cpu().numpy().astype('float32')
