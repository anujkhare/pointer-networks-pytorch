import numpy as np
import torch
import torch.nn.functional as F


class ScaledDotAttention(torch.nn.Module):
    def __init__(self, d_k=94, d_q=94, model_dim=32):
        super().__init__()
        self.proj_k = torch.nn.Linear(d_k, model_dim)
        self.proj_q = torch.nn.Linear(d_q, model_dim)
#         self.proj_v = torch.nn.Linear(d_v, model_dim)

        self.model_dim = model_dim

    def forward(self,K,Q=None,V=None):
        # Input is assumed to be bsz * n_k * d_k
        N,d = K.shape[-2:]

        if Q is None and V is None:
            Q = K
            V = K

        # The number of keys and values should be the same
        assert N == V.shape[-2]

        # linearly project
        proj_k = self.proj_k(K) # -> bsz * n_k * model_dim
        proj_q = self.proj_q(Q) # -> bsz * n_q * model_dim
#         proj_v = self.proj_v(V) # -> bsz * n_k * model_dim

        # dot product
        dot1 = torch.matmul(proj_q, proj_k.transpose(1,2)) # -> bsz * n_q * n_k

        # scale
        scaled = dot1 / np.sqrt(self.model_dim)

        # softmax over axis 1, 
        sft = F.softmax(scaled,dim=-1)  # -> bsz * n_q * n_k

        # dot product with attn signal
#         rslt = torch.matmul(sft, proj_v) # -> Nxd
        rslt = torch.matmul(sft, V) # -> Nxd

        return rslt


def test_sda_batched():
    bsz = 4

    d_q = 300
    d_k = 400
    d_v = 500
    Q = torch.from_numpy(np.random.rand(bsz, 20, d_q).astype(np.float32))
    K = torch.from_numpy(np.random.rand(bsz, 30, d_k).astype(np.float32))
    V = torch.from_numpy(np.random.rand(bsz, 30, d_v).astype(np.float32))

    scaled_attention = ScaledDotAttention(d_q=d_q, d_k=d_k, model_dim=32)
#     scaled_attention = ScaledDotAttention(d_q=d_q, d_k=d_k, d_v=d_v, model_dim=32)

    values = scaled_attention(K=K, Q=Q, V=V)
    assert values.shape[1] == 20 and values.shape[2] == d_v

    

def test_sda_common_queries():
    bsz = 4

    d_q = 300
    d_k = 400
    d_v = 500
    Q = torch.from_numpy(np.random.rand(20, d_q).astype(np.float32))  # Just 20 queries to be run against everything
    K = torch.from_numpy(np.random.rand(bsz, 30, d_k).astype(np.float32))
    V = torch.from_numpy(np.random.rand(bsz, 30, d_v).astype(np.float32))

    scaled_attention = ScaledDotAttention(d_q=d_q, d_k=d_k, model_dim=32)
#     scaled_attention = ScaledDotAttention(d_q=d_q, d_k=d_k, d_v=d_v, model_dim=32)

    values = scaled_attention(K=K, Q=Q, V=V)
    assert values.shape[1] == 20 and values.shape[2] == d_v

    
# Call the tests
test_sda_batched()
test_sda_common_queries()