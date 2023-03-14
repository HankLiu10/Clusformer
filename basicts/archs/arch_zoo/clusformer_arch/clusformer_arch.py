import math

import torch
import torch.nn as nn

class NodeEncoder(nn.Module):
    def __init__(self, model_args):
        super(NodeEncoder, self).__init__()
        self.samples_per_hour = model_args["samples_per_hour"]
        self.channel = model_args["channel"]
        self.num_nodes = model_args["num_nodes"]
        self.dim = model_args["feature_embedding_dim"]
        self.input_len = model_args["input_len"]
        self.adaptive_embedding_dim = model_args["adaptive_embedding_dim"]

        # feature embedding
        self.feature_linear = nn.Linear(self.channel, self.dim)
        # spatial embedding
        self.spatial_embedding = nn.Parameter(torch.empty(self.num_nodes, self.dim))
        nn.init.kaiming_uniform_(self.spatial_embedding)
        # timestamp embedding
        self.timestamp_embedding = nn.Parameter(torch.empty(24 * self.samples_per_hour, self.dim))
        nn.init.kaiming_uniform_(self.timestamp_embedding)
        # dayinweek embedding
        self.dayinweek_embedding = nn.Parameter(torch.empty(7, self.dim))
        nn.init.kaiming_uniform_(self.dayinweek_embedding)
        # self-adaptive embedding
        self.adaptive_embedding = nn.Parameter(torch.empty(self.input_len , self.num_nodes, self.adaptive_embedding_dim))
        nn.init.kaiming_uniform_(self.adaptive_embedding)

    def forward(self, input):
        # generate node embedding
        #   inputs:  STFeature(B,T,N,F+EXTRA_FEATURES)
        batch_size = input.shape[0]
        if len(input.shape) == 3:
            input = torch.unsqueeze(input, -1)
        adaptive_embedding = torch.tile(self.adaptive_embedding, (batch_size, 1, 1, 1))
        feature_embedding = self.feature_linear(input[:, :, :, :self.channel])
        timestamp_embedding = self.timestamp_embedding[input[:, :, :, self.channel].type(torch.LongTensor)]
        dayinweek_embedding = self.dayinweek_embedding[input[:, :, :, self.channel + 1].type(torch.LongTensor)]
        # spatial_embedding = self.spatial_embedding[input[:, :, :, self.channel + 2].type(torch.LongTensor)]
        # node_embedding = torch.cat((feature_embedding, timestamp_embedding, dayinweek_embedding, spatial_embedding, adaptive_embedding), axis=-1)  # (B,T,N,6*node_embedding_dim)
        node_embedding = torch.cat((feature_embedding, timestamp_embedding, dayinweek_embedding,  adaptive_embedding), axis=-1)  # (B,T,N,5*node_embedding_dim)
        return node_embedding

class AddBatchNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(AddBatchNorm, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x, y):
        z = x + y
        z = self.bn(z.transpose(-2,-1)).transpose(-2,-1)
        z = self.alpha * z + self.bias
        return z


class FeedForward(nn.Module):
    def __init__(self, input_size):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_size, input_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, Num_of_attn_heads = 4):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = Num_of_attn_heads
        self.head_size = input_size // Num_of_attn_heads

        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)

        self.output = nn.Linear(input_size, input_size)

    def forward(self, inputs):
        # inputs: [batch_size, seq_len, input_size]
        batch_size, seq_len, _ = inputs.size()
        Q = self.query(inputs).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_size]
        K = self.key(inputs).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_size]
        V = self.value(inputs).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_size]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_size ** 0.5)  # [batch_size, num_heads, seq_len, seq_len]

        attn_weights = nn.functional.softmax(scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
        attn_output = torch.matmul(attn_weights, V)  # [batch_size, num_heads, seq_len, head_size]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)  # [batch_size, seq_len, num_heads * head_size]
        output = self.output(attn_output)  # [batch_size, seq_len, input_size]

        return output

class CentroidDiscoverBlock(nn.Module):
    def __init__(self, node_embedding_dim, centroid_dim):
        super(CentroidDiscoverBlock, self).__init__()
        self.centroid_dim = centroid_dim
        self.node_embedding_dim = node_embedding_dim

        self.CentroidsQLayer = nn.Linear(self.centroid_dim, node_embedding_dim)
        self.NodeKLayer = nn.Linear(node_embedding_dim, node_embedding_dim)
        self.NodeVLayer = nn.Linear(node_embedding_dim, node_embedding_dim)
        # self.restart = nn.Parameter(torch.rand(self.centroid_dim)-0.5)
        # self.restart = (torch.rand(node_embedding_dim)/10 - 0.05)

        self.AlignLayer = nn.Linear(self.node_embedding_dim,self.centroid_dim)
        self.MHSA = MultiHeadAttention(centroid_dim)
        self.AddAndNorm = AddBatchNorm(centroid_dim)
        self.FeedForward = FeedForward(centroid_dim)

    def forward(self, STFeature, centroidsTemp):
        # Performed to calculate the new centroids according to the old centroids and STFeatures.
        # Args:
        #   inputs:  STFeature(B,T,N,node_embedding_dim)  centroids(B,L,centroid_dim)
        #   returns: scaled attention among STFeatures and old centroids (B,T,N,L), along with the new centroids (B,L,centroid_dim).

        centroids = centroidsTemp.clone()

        self.CentroidNum = centroidsTemp.size()[1]  # L
        Q_Cent = self.CentroidsQLayer(centroidsTemp)  # (B,L,node_embedding_dim)
        K_Node = self.NodeKLayer(STFeature)  # (B,T,N,node_embedding_dim)
        V_Node = self.NodeVLayer(STFeature)  # (B,T,N,node_embedding_dim)

        Scaled_Cross_Attn = torch.einsum("blc, btnc -> btnl", Q_Cent, K_Node) / math.sqrt(self.node_embedding_dim)  # (B,T,N,L)
        with torch.no_grad():
            Belongs = nn.functional.one_hot(torch.argmax(Scaled_Cross_Attn, dim=3), self.CentroidNum).float()  # (B,T,N,L)
            # Belongs = torch.nn.functional.gumbel_softmax(Scaled_Cross_Attn, dim=3,tau=1e-200, hard=True)  # (B,T,N,L)
            cluster_result = torch.einsum('btnl, btnc -> blc', Belongs, V_Node)  # (B,L,node_embedding_dim)
            BelongsMeans = torch.sum(Belongs.reshape(Belongs.shape[0],-1,self.CentroidNum).permute(0,2,1),dim=-1) **2  # (B,L)
            BelongsMeans = torch.unsqueeze(BelongsMeans,2).repeat(1,1,self.node_embedding_dim) + 1   # eps
            cluster_result = cluster_result/BelongsMeans

        centroids = centroids + self.AlignLayer(cluster_result)
        centroids_ = self.MHSA(centroids)
        centroids = self.AddAndNorm(centroids,centroids_)
        centroids = self.FeedForward(centroids)

        return centroids

class CentroidEstimator(nn.Module):
    def __init__(self, node_embedding_dim, num_blocks, num_of_centroid, centroid_dim):
        super(CentroidEstimator, self).__init__()
        self.num_blocks = num_blocks
        self.CentroidDiscoverBlocks = nn.ModuleList()
        for i in range(num_blocks):
            self.CentroidDiscoverBlocks.append(CentroidDiscoverBlock(node_embedding_dim, centroid_dim))

        self.centroids = nn.Parameter(torch.empty(num_of_centroid, centroid_dim))  # (L,centroid_dim)
        nn.init.kaiming_uniform_(self.centroids)

    def forward(self, node_embedding):
        # (B,T,N,node_embedding_dim)
        batch_size = node_embedding.shape[0]
        centroids = torch.tile(self.centroids, (batch_size, 1, 1))
        for i in range(self.num_blocks):
            centroids = self.CentroidDiscoverBlocks[i](node_embedding, centroids)  # (B,T,N,L) (B,L,centroid_dim) (B,T,N,L)

        return centroids


class NodeDecoder(nn.Module):
    def __init__(self, model_args, num_of_blocks, centroid_dim):
        super(NodeDecoder, self).__init__()
        self.node_embedding_dim = model_args["feature_embedding_dim"]+model_args["diw_embedding_dim"]+model_args["tid_embedding_dim"]+model_args["adaptive_embedding_dim"]
        self.num_nodes = model_args["num_nodes"]

        self.centroid_dim = centroid_dim

        self.node_queries = nn.ModuleList()
        self.centroids_keys = nn.ModuleList()
        self.centroids_values = nn.ModuleList()

        self.attn_output = nn.ModuleList()
        self.AddAndNorm = nn.ModuleList()
        self.FeedForward = nn.ModuleList()
        self.Output = nn.ModuleList()

        self.num_of_blocks = num_of_blocks
        self.alpha = nn.Parameter(torch.tensor([1.0]))

        for i in range(self.num_of_blocks):
            self.node_queries.append(nn.Linear(self.node_embedding_dim, self.node_embedding_dim))
            self.centroids_keys.append(nn.Linear(centroid_dim, self.node_embedding_dim))
            self.centroids_values.append(nn.Linear(centroid_dim, self.node_embedding_dim))
            self.attn_output.append(nn.Linear(self.node_embedding_dim, self.node_embedding_dim))
            self.AddAndNorm.append(AddBatchNorm(self.node_embedding_dim))
            self.FeedForward.append(FeedForward(self.node_embedding_dim))
            self.Output.append(nn.Linear(self.node_embedding_dim, self.node_embedding_dim))

    def forward(self, node_embed, centroids):
        # (B,T,N,node_embedding_dim) (B,sumL,output_centroid_dim)
        batch_size = node_embed.shape[0]
        node_embed = torch.reshape(node_embed, (batch_size, -1, node_embed.shape[-1]))  # (B,T*N,node_embedding_dim)
        for i in range(self.num_of_blocks):
            Q_node = self.node_queries[i](node_embed)
            K_centroid = self.centroids_keys[i](centroids)
            V_centroid = self.centroids_values[i](centroids)

            P2CAttn = torch.matmul(Q_node, K_centroid.transpose(-2, -1)) / (self.node_embedding_dim ** 0.5)

            attn_weights = nn.functional.softmax(P2CAttn, dim=-1)
            attn_output = torch.matmul(attn_weights, V_centroid)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.node_embedding_dim)
            attn_output = self.Output[i](attn_output)

            node_embed = self.AddAndNorm[i](node_embed,attn_output)
            node_embed = self.FeedForward[i](node_embed)

        output = torch.reshape(node_embed, (batch_size, -1 , self.num_nodes, self.node_embedding_dim))  # (B,T,N,node_embedding_dim)

        return output  # (B,T,N,node_embedding_dim)

class OutputBlock(nn.Module):
    def __init__(self, model_args):
        super(OutputBlock, self).__init__()

        self.input_dim = model_args["feature_embedding_dim"]+model_args["diw_embedding_dim"]+model_args["tid_embedding_dim"]+model_args["adaptive_embedding_dim"]
        self.output_dim = model_args["channel"]
        self.input_len = model_args["input_len"]
        self.output_len = model_args["output_len"]
        self.channel = model_args["channel"]
        self.num_nodes = model_args["num_nodes"]

        self.FeedForward1 = FeedForward(self.input_dim)
        self.AddAndNorm1 = AddBatchNorm(self.input_dim)
        self.FeedForward2 = FeedForward(self.input_dim)

        self.output = nn.Linear(self.input_len *self.input_dim,self.output_len *self.channel)

    def forward(self, input):
        # (B,T,N,node_embedding_dim) (B,T,N,node_embedding_dim)
        batch_size = input.shape[0]

        outputs = self.AddAndNorm1(
            input.view(batch_size,self.input_len *self.num_nodes,-1),
            self.FeedForward1(input).view(batch_size,self.input_len *self.num_nodes,-1))\
            .view(batch_size,self.input_len ,self.num_nodes,-1)

        outputs = self.FeedForward2(outputs)

        outputs = outputs.permute(0,2,1,3)           # (B,N,T,node_embedding_dim)
        outputs = torch.reshape(outputs,(batch_size,self.num_nodes,-1))
        outputs = self.output(outputs)
        outputs = torch.reshape(outputs, (batch_size, self.num_nodes, self.output_len ,self.channel))
        outputs = outputs.permute(0, 2, 1, 3)  # (B,Tout,N,F)

        return outputs

class Clusformer(nn.Module):
    def __init__(self, **model_args):
        super(Clusformer, self).__init__()

        # attributes
        self.num_nodes = model_args["num_nodes"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.output_len = model_args["output_len"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]
        self.node_embedding_dim = model_args["feature_embedding_dim"]+model_args["diw_embedding_dim"]+model_args["tid_embedding_dim"]+model_args["adaptive_embedding_dim"]
        self.channel = model_args["channel"]
        self.num_of_centroids = model_args["num_of_centroids"]
        self.centroid_embed_dims = model_args["centroid_embed_dims"]
        self.samples_per_hour = model_args["samples_per_hour"]

        self.nodeEncoder = NodeEncoder(model_args)

        self.EDBs_in_ED = 2

        self.CentroidEstimators = nn.ModuleList()
        self.NodeDecoders = nn.ModuleList()

        self.attn_dim = 128
        self.num_of_nodeDecoder_block = 1

        for (num_of_centroid, centroid_embed_dim) in zip(self.num_of_centroids, self.centroid_embed_dims):
            self.CentroidEstimators.append(CentroidEstimator(self.node_embedding_dim, self.EDBs_in_ED, num_of_centroid, centroid_embed_dim))  # (B,L,centroid_dim)
            self.NodeDecoders.append(NodeDecoder(model_args,self.num_of_nodeDecoder_block,centroid_embed_dim))  # (B,L,centroid_dim)

        self.alpha = nn.Parameter(torch.tensor([1e-2]))
        self.beta = nn.Parameter(torch.tensor([1e-2]))

        # self.outputs = nn.Parameter(torch.ones(BATCHSIZE,self.input_len ,self.num_nodes,6*self.node_embedding_dim))

        self.CentroidAlignBlocks = nn.ModuleList()
        for (num_of_centroid, centroid_embed_dim) in zip(self.num_of_centroids, self.centroid_embed_dims):
            self.CentroidAlignBlocks.append(nn.Linear(centroid_embed_dim, self.node_embedding_dim))

        self.OutputLayer = OutputBlock(model_args)

    def forward(self, history_data: torch.Tensor,future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs):
        input = history_data
        node_embedding = self.nodeEncoder(input)  # (B,T,N,6 * node_embedding_dim)
        for i in range(len(self.num_of_centroids)):
            for j in range(self.EDBs_in_ED):
                centroid = self.CentroidEstimators[i](node_embedding)
            outputs = self.NodeDecoders[i](node_embedding, centroid)  # (B,T,N,node_embedding_dim)

        outputs = self.OutputLayer(outputs)  # (B,T,N,F)

        return outputs   # (B,T,N,F) (num_of_centroidDecoders,B,T,N,L)