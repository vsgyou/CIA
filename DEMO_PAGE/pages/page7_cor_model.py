
from huggingface_hub import hf_hub_download
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# huggingface repo
REPO_ID = "jihji/cor-g-yelp-model"
SUBFOLDER = "yelp"

def load_from_hub(filename):
    return hf_hub_download(repo_id=REPO_ID, filename=filename, subfolder=SUBFOLDER, repo_type="model")

class COR_G(nn.Module):
    """
    code for adding causal graph
    extension of item feature is not used in this model
    """
    def __init__(self, mlp_q_dims, mlp_p1_1_dims, mlp_p1_2_dims, mlp_p2_dims, mlp_p3_dims, \
                                                         item_feature, adj, E1_size, dropout=0.5, bn=0, sample_freq=1, regs=0, act_function='tanh'):
        super(COR_G, self).__init__()
        self.mlp_q_dims = mlp_q_dims
        self.mlp_p1_1_dims = mlp_p1_1_dims
        self.mlp_p1_2_dims = mlp_p1_2_dims
        self.mlp_p2_dims = mlp_p2_dims
        self.mlp_p3_dims = mlp_p3_dims
        self.adj = adj
        self.E1_size = E1_size
        self.Z1_size = adj.size(0)
        self.bn = bn
        self.sample_freq = sample_freq
        self.regs = regs

        if act_function == 'tanh':
            self.act_function = F.tanh
        elif act_function == 'sigmoid':
            self.act_function = F.sigmoid

        # not used in this model, extended for using item feature in future work
        self.item_feature = item_feature
        self.item_learnable_dim = self.mlp_p2_dims[-1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.item_learnable_feat = torch.randn([self.item_feature.size(0), self.item_learnable_dim], \
                                            requires_grad=True).to(device)

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.mlp_q_dims[:-1] + [self.mlp_q_dims[-1] * 2]
        temp_p1_1_dims = self.mlp_p1_1_dims
        temp_p1_2_dims = self.mlp_p1_2_dims[:-1] + [self.mlp_p1_2_dims[-1] * 2]
        temp_p2_dims = self.mlp_p2_dims[:-1] + [self.mlp_p2_dims[-1] * 2]
        temp_p3_dims = self.mlp_p3_dims

        self.mlp_q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.mlp_p1_1_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_p1_1_dims[:-1], temp_p1_1_dims[1:])])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mlp_p1_2_layers = [(torch.randn([self.Z1_size, d_in, d_out],requires_grad=True)).to(device) for
            d_in, d_out in zip(temp_p1_2_dims[:-1], temp_p1_2_dims[1:])]
       
        for i, matrix in enumerate(self.mlp_p1_2_layers):
            temp = torch.unsqueeze(matrix,0) if i==0 else torch.concat((temp,torch.unsqueeze(matrix,0)),0)
        self.mlp_p1_2_layers = nn.Parameter(temp)

        self.mlp_p2_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_p2_dims[:-1], temp_p2_dims[1:])])
        self.mlp_p3_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_p3_dims[:-1], temp_p3_dims[1:])])
        
        self.drop = nn.Dropout(dropout)
        if self.bn:
            self.batchnorm = nn.BatchNorm1d(E1_size)
        self.init_weights()

    def reuse_Z2(self,D,E1):
        if self.bn:
            E1 = self.batchnorm(E1)
        D = F.normalize(D)
        mu, _ = self.encode(torch.cat((D,E1),1))
        h_p2 = mu
        for i, layer in enumerate(self.mlp_p2_layers):
            h_p2 = layer(h_p2)
            if i != len(self.mlp_p2_layers) - 1:
                h_p2 = F.tanh(h_p2)
            else:
                h_p2 = h_p2[:, :self.mlp_p2_dims[-1]]
        return h_p2

    def forward(self, D, E1, Z2_reuse=None, CI=0):
        if self.bn:
            E1 = self.batchnorm(E1)
        encoder_input = torch.cat((D, E1), 1)  # D,E1
        mu, logvar = self.encode(encoder_input)    # E2 distribution
        E2 = self.reparameterize(mu, logvar)       # E2
        
        if CI == 1: # D=NULL
            encoder_input_0 = torch.cat((torch.zeros_like(D), E1), 1)
            mu_0, logvar_0 = self.encode(encoder_input_0)
            E2_0 = self.reparameterize(mu_0, logvar_0)
            scores = self.decode(E1, E2, E2_0, Z2_reuse)
        else:
            scores = self.decode(E1, E2, None, Z2_reuse)
        reg_loss = self.reg_loss()
        return scores, mu, logvar, reg_loss
    
    def encode(self, encoder_input):
        h = self.drop(encoder_input)
        
        for i, layer in enumerate(self.mlp_q_layers):
            h = layer(h)
            if i != len(self.mlp_q_layers) - 1:
                h = self.act_function(h)
            else:
                mu = h[:, :self.mlp_q_dims[-1]]
                logvar = h[:, self.mlp_q_dims[-1]:]
                
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def decode(self, E1, E2, E2_0=None, Z2_reuse=None):
        if E2_0 is None:
            h_p1 = torch.cat((E1, E2), dim=1)  # (B, D1 + D2)
        else:
            h_p1 = torch.cat((E1, E2_0), dim=1)

        h_p1 = h_p1.unsqueeze(-1)  # (B, D, 1)
        for i, layer in enumerate(self.mlp_p1_1_layers):
            h_p1 = layer(h_p1)
            if i != len(self.mlp_p1_1_layers) - 1:
                h_p1 = self.act_function(h_p1)

        h_p1 = torch.matmul(self.adj, h_p1).unsqueeze(2)  # (Z1_size, B, D, 1)
        for i, matrix in enumerate(self.mlp_p1_2_layers):
            h_p1 = torch.matmul(h_p1, matrix)
            if i != len(self.mlp_p1_2_layers) - 1:
                h_p1 = self.act_function(h_p1)

        # Z1 mean, logvar
        Z1_mu = h_p1[:, :, :, :self.mlp_p1_2_dims[-1]].squeeze(-1)      # (Z1_size, B, D)
        Z1_logvar = h_p1[:, :, :, self.mlp_p1_2_dims[-1]:].squeeze(-1)  # (Z1_size, B, D)

        # (B, Z1_size, D)
        Z1_mu = Z1_mu.permute(1, 0, 2)
        Z1_logvar = Z1_logvar.permute(1, 0, 2)

        # Z2
        h_p2 = E2
        for i, layer in enumerate(self.mlp_p2_layers):
            h_p2 = layer(h_p2)
            if i != len(self.mlp_p2_layers) - 1:
                h_p2 = self.act_function(h_p2)

        Z2_mu = h_p2[:, :self.mlp_p2_dims[-1]]
        Z2_logvar = h_p2[:, self.mlp_p2_dims[-1]:]

        # sampling
        Z1_list, Z2_list = [], []
        for _ in range(self.sample_freq):
            Z1_sample = self.reparameterize(Z1_mu, Z1_logvar)  # (B, Z1_size, D)
            Z2_sample = self.reparameterize(Z2_mu, Z2_logvar)  # (B, D)
            Z1_list.append(Z1_sample)
            Z2_list.append(Z2_sample)

        Z1 = torch.mean(torch.stack(Z1_list), dim=0)  # (Z1_size, B, D)
        Z1 = Z1.permute(1, 0, 2)                      # (B, Z1_size, D)
        Z1 = Z1.reshape(Z1.size(0), -1)               # (B, Z1_size * D)

        Z2 = torch.mean(torch.stack(Z2_list), dim=0)  # (B, D)
        user_preference = torch.cat((Z1, Z2), dim=1)  # (B, Z1_size * D + D)

        # MLP for prediction
        h_p3 = user_preference
        for i, layer in enumerate(self.mlp_p3_layers):
            h_p3 = layer(h_p3)
            if i != len(self.mlp_p3_layers) - 1:
                h_p3 = self.act_function(h_p3)

        return h_p3
    
    def init_weights(self):
        for layer in self.mlp_q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.mlp_p1_1_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
            
        for matrix in self.mlp_p1_2_layers:
            # Xavier Initialization for weights
            size = matrix.data.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            matrix.data.normal_(0.0, std)
            
        for layer in self.mlp_p2_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
            
        for layer in self.mlp_p3_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

    def reg_loss(self):
        r"""Calculate the L2 normalization loss of model parameters.
        Including embedding matrices and weight matrices of model.
        Returns:
            loss(torch.FloatTensor): The L2 Loss tensor. shape of [1,]
        """
        reg_loss = 0
        for name, parm in self.mlp_q_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1/2) * parm.norm(2).pow(2)
        for name, parm in self.mlp_p1_1_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1/2) * parm.norm(2).pow(2)
        for name, parm in self.mlp_p2_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1/2) * parm.norm(2).pow(2)
        for name, parm in self.mlp_p3_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1/2) * parm.norm(2).pow(2)
        return reg_loss