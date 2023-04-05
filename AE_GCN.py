from __future__ import print_function, division
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
from utils import normalize, sparse_mx_to_torch_sparse_tensor
from GNN import GNNLayer

import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z


class AE_GCN(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, 
                n_input, n_z, n_clusters, v=1):
        super(AE_GCN, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z = self.ae(x)
        
        sigma = 0.5

        # GCN Module
        h = self.gnn_1(x, adj)
        h = self.gnn_2((1-sigma)*h + sigma*tra1, adj)
        h = self.gnn_3((1-sigma)*h + sigma*tra2, adj)
        h = self.gnn_4((1-sigma)*h + sigma*tra3, adj)
        h = self.gnn_5((1-sigma)*h + sigma*z, adj, active=False)
        predict = F.softmax(h, dim=1)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


if __name__ == "__main__":

    n_z=10
    n_clusters=10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr=1e-3
    k=6

    adata = sc.read_visium('./Invasive Ductal Carcinoma Stained With Fluorescent CD3 Antibody/')
    adata.var_names_make_unique()
    
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor='seurat_v3')
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]

    model = AE_GCN(500, 500, 2000, 2000, 500, 500,
                   n_input=adata.n_vars,
                   n_z=n_z,
                   n_clusters=n_clusters,
                   v=1.0).to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=lr)

    x = torch.from_numpy(adata.X.toarray()).type(torch.float32).to(device)

    for epoch in range(1000):

        x_bar, _, _, _, _ = model.ae(x)
        loss = torch.norm(x - x_bar)

        print(epoch, 'loss='+str(loss.detach().cpu().numpy()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # KNN Graph
    coord = adata.obsm['spatial']
    neigh = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(coord)
    adj = neigh.kneighbors_graph(coord)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj = adj.cuda()

    # cluster parameter initiate
    with torch.no_grad():
        _, _, _, _, z = model.ae(x)

    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    for epoch in range(500):
        if epoch % 1 == 0:
        # update_interval
            _, tmp_q, pred, _ = model(x, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

        x_bar, q, pred, _ = model(x, adj)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, x)

        loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss

        print(epoch, 'loss='+str(loss.detach().cpu().numpy()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



    adata.obs['cluster'] = pred.data.cpu().numpy().argmax(1).astype('U')
    
    fig, axs = plt.subplots(figsize=(7, 7))
    
    sc.pl.spatial(
        adata,
        img_key=None,
        color='cluster',
        size=1.5,
        palette=sc.pl.palettes.default_20,
        legend_loc='right margin',
        frameon=False,
        title='', # method,
        show=False,
        ax=axs,
    )
    
    plt.tight_layout()



















