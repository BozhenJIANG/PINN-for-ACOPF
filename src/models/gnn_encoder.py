"""
Graph Neural Network Encoder for ACOPF with topology-aware message passing.

The GNN encoder uses the admittance matrix (Ybus) as the graph structure
to propagate information across the power network.

Architecture:
  - State: [B, state_dim] = [B, num_buses * 2] (Pd, Qd per bus)
  - Reshape to [B, num_buses, 2] before GNN
  - GNN layers output [B, num_buses, gnn_hidden_per_bus]
  - Flatten + FC layers → latent space → decoded action
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.activations import BetaSiLU, MinMaxSigmoid

class GNNLayer(nn.Module):
    """
    Graph Neural Network layer with message passing.
    Uses Ybus magnitude as adjacency matrix.
    """
    
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.msg_weight = nn.Linear(in_features, out_features)
        self.self_weight = nn.Linear(in_features, out_features)

        
    def forward(self, x, adj_matrix):
        """
        Args:
            x:          [batch, num_buses, in_features]
            adj_matrix: [num_buses, num_buses]  (Ybus magnitude)
        Returns:
            out:        [batch, num_buses, out_features]
        """
        # Degree-normalized adjacency
        degree = torch.sum(torch.abs(adj_matrix), dim=1, keepdim=True) + 1e-6
        norm_adj = adj_matrix / degree                      # [N, N]

        # Aggregate neighbour features:
        # out[b, n, f] = sum_m  norm_adj[n, m] * x[b, m, f]
        nbr = torch.einsum('nm,bmf->bnf', norm_adj, x)     # [B, N, out]
        return F.silu(self.self_weight(x) + self.msg_weight(nbr))


class GNNEncoder(nn.Module):
    """
    Deterministic GNN encoder: state -> action.

    Architecture:
      state [B, state_dim]
        -> reshape [B, num_buses, state_feat]
        -> GNN1, GNN2  ->  [B, num_buses, H]
        -> flatten     ->  [B, num_buses * H]
        -> enc_fc      ->  [B, 4*intermediate_dim]   (no bottleneck)
        -> cat(h, state) -> [B, 4*intermediate_dim + state_dim]
        -> decoder     ->  [B, act_dim]
    """
    
    def __init__(self, state_dim, act_dim, intermediate_dim, latent_dim,
                 num_buses, ybus_matrix, limits, model_type="GNN",
                 gnn_hidden_per_bus=4):
        super(GNNEncoder, self).__init__()
        
        self.state_dim   = state_dim
        self.act_dim     = act_dim
        self.latent_dim  = latent_dim
        self.num_buses   = num_buses
        self.limits      = limits

        assert state_dim % num_buses == 0, \
            f"state_dim ({state_dim}) must be divisible by num_buses ({num_buses})"
        self.state_feat = state_dim // num_buses   # features per bus (typically 2: Pd, Qd)

        # Binary adjacency matrix: 1 if transmission line exists, 0 otherwise
        # Take abs() on numpy side to avoid "casting complex to real" warning
        if isinstance(ybus_matrix, np.ndarray):
            ybus_matrix = torch.tensor(np.abs(ybus_matrix), dtype=torch.float32)
        elif ybus_matrix.is_complex():
            ybus_matrix = torch.abs(ybus_matrix).float()
        self.register_buffer('ybus', ybus_matrix)
        adj = (ybus_matrix > 0).float()               # [N, N]  binary {0,1}
        self.register_buffer('adj_matrix', adj)

        # GNN layers (operate on per-bus features)
        H = gnn_hidden_per_bus
        self.gnn1 = GNNLayer(self.state_feat, H)
        self.gnn2 = GNNLayer(H, H)
        gnn_out_dim = num_buses * H                # e.g. 1354 * 4 = 5416

        # Encoder FC layers
        self.enc_fc = nn.Sequential(
            nn.Linear(gnn_out_dim,        intermediate_dim * 4),  nn.SiLU(),
            nn.Linear(intermediate_dim*4, intermediate_dim * 8),  nn.SiLU(),
            nn.Linear(intermediate_dim*8, intermediate_dim * 4),  nn.SiLU(),
        )
        enc_out = intermediate_dim * 4   # no bottleneck, feed directly to decoder

        # Decoder FC layers  (input = enc_fc output concatenated with raw state)
        self.decoder = nn.Sequential(
            nn.Linear(enc_out + state_dim, intermediate_dim),  nn.SiLU(),
            nn.Linear(intermediate_dim, intermediate_dim * 2),  nn.SiLU(),
            nn.Linear(intermediate_dim * 2, intermediate_dim * 4),  nn.SiLU(),
            nn.Linear(intermediate_dim * 4,   intermediate_dim * 8),  nn.SiLU(),
            nn.Linear(intermediate_dim * 8,   intermediate_dim * 16), nn.SiLU(),
            nn.Linear(intermediate_dim * 16,  intermediate_dim * 8),  nn.SiLU(),
            nn.Linear(intermediate_dim * 8,   intermediate_dim * 4),  nn.SiLU(),
            nn.Linear(intermediate_dim * 4,   intermediate_dim * 2),  nn.SiLU(),
            nn.Linear(intermediate_dim * 2,   intermediate_dim),
            nn.Linear(intermediate_dim,   act_dim),
        )

        # Action range buffers
        self.register_buffer('act_min', torch.tensor([lim[0] for lim in limits[:act_dim]], dtype=torch.float32))
        self.register_buffer('act_max', torch.tensor([lim[1] for lim in limits[:act_dim]], dtype=torch.float32))

    def encode(self, state):
        """
        state: [B, state_dim]
        returns: action [B, act_dim]
        """
        B = state.shape[0]

        # Reshape state into per-bus features: [B, num_buses, state_feat]
        x = state.view(B, self.num_buses, self.state_feat)

        # GNN message passing (uses binary adjacency)
        x = self.gnn1(x, self.adj_matrix)   # [B, num_buses, H]
        x = self.gnn2(x, self.adj_matrix)   # [B, num_buses, H]
        x = x.reshape(B, -1)              # [B, num_buses * H]

        # Encoder FC (full width, no bottleneck)
        h = self.enc_fc(x)                # [B, 4*intermediate_dim]

        # Decode: concatenate enc_fc output with raw state
        action = self.decoder(torch.cat([h, state], dim=1))
        return action
        # action_ = self.act_min + torch.sigmoid(action) * (self.act_max - self.act_min)
        # return action_


    def forward(self, state):
        return self.encode(state)


class GNNPINNPF(nn.Module):
    """
    Physics-informed power flow model with GNN for state encoding.

    Action (generator setpoints) is processed by an FC branch.
    State (load demands) is processed by a GNN branch using Ybus.
    Outputs: concatenated [q (260), u (1094), delta (1354)] = 2708 dims.
    """

    def __init__(self, act_dim, state_dim, intermediate_dim, limits_q,
                 num_buses, ybus_matrix, gnn_hidden_per_bus=4):
        super(GNNPINNPF, self).__init__()

        self.act_dim   = act_dim
        self.state_dim = state_dim
        self.num_buses = num_buses
        self.limits_q  = limits_q
        self.beta_silu = BetaSiLU(beta=0.3)
        self.min_max_sigmoid = MinMaxSigmoid()

        # Binary adjacency matrix: 1 if transmission line exists, 0 otherwise
        # Take abs() on numpy side to avoid "casting complex to real" warning
        if isinstance(ybus_matrix, np.ndarray):
            ybus_matrix = torch.tensor(np.abs(ybus_matrix), dtype=torch.float32)
        elif ybus_matrix.is_complex():
            ybus_matrix = torch.abs(ybus_matrix).float()
        self.register_buffer('ybus',     ybus_matrix)
        adj = (ybus_matrix > 0).float()               # [N, N]  binary {0,1}
        self.register_buffer('adj_matrix', adj)

        # State features per bus
        assert state_dim % num_buses == 0
        self.state_feat = state_dim // num_buses

        # GNN branch for state  (GNN×2 + 7-layer FC, mirrors main.py y_branch)
        H = gnn_hidden_per_bus
        self.gnn1 = GNNLayer(self.state_feat, H)
        self.gnn2 = GNNLayer(H, H)
        gnn_out_dim = num_buses * H

        self.state_fc = nn.Sequential(
            nn.Linear(gnn_out_dim+2708,       intermediate_dim),      nn.SiLU(),
            nn.Linear(intermediate_dim,     2*intermediate_dim),    nn.SiLU(),
            nn.Linear(2*intermediate_dim,   4*intermediate_dim),    nn.SiLU(),
            nn.Linear(4*intermediate_dim,   8*intermediate_dim),    nn.SiLU(),
            nn.Linear(8*intermediate_dim,  16*intermediate_dim),    nn.SiLU(),
            nn.Linear(16*intermediate_dim, 32*intermediate_dim),    nn.SiLU(),
            nn.Linear(32*intermediate_dim,  8*intermediate_dim),    nn.SiLU(),
        )
        state_branch_out = 8 * intermediate_dim   # 1024

        # FC branch for action  (7-layer, mirrors main.py x_branch)
        self.action_fc = nn.Sequential(
            nn.Linear(act_dim,              intermediate_dim),      nn.SiLU(),
            nn.Linear(intermediate_dim,     2*intermediate_dim),    nn.SiLU(),
            nn.Linear(2*intermediate_dim,   4*intermediate_dim),    nn.SiLU(),
            nn.Linear(4*intermediate_dim,   8*intermediate_dim),    nn.SiLU(),
            nn.Linear(8*intermediate_dim,  16*intermediate_dim),    nn.SiLU(),
            nn.Linear(16*intermediate_dim, 32*intermediate_dim),    nn.SiLU(),
            nn.Linear(32*intermediate_dim,  8*intermediate_dim),    nn.SiLU(),
        )
        action_branch_out = 8 * intermediate_dim  # 1024

        # Combined layers  (5-layer, mirrors main.py combined_net, peak 64×D)
        combined_in = state_branch_out + action_branch_out   # 2048
        self.combined_fc = nn.Sequential(
            nn.Linear(combined_in,          16*intermediate_dim),   nn.SiLU(),
            nn.Linear(16*intermediate_dim,  32*intermediate_dim),   nn.SiLU(),
            nn.Linear(32*intermediate_dim,  64*intermediate_dim),   nn.SiLU(),
            nn.Linear(64*intermediate_dim,  32*intermediate_dim),   nn.SiLU(),
            nn.Linear(32*intermediate_dim,  2708),
        )

        self.beta = nn.Parameter(torch.tensor(0.3))

    def forward(self, action, state):
        """
        action: [B, act_dim]
        state:  [B, state_dim]
        returns:[B, 2708]  (q | u | delta)
        """
        B = state.shape[0]

        # GNN branch: state -> [B, state_branch_out]  (binary adjacency)
        xs = state.view(B, self.num_buses, self.state_feat)
        xs = self.gnn1(xs, self.adj_matrix)
        xs = self.gnn2(xs, self.adj_matrix)
        
        xs = self.state_fc(torch.cat([xs.reshape(B, -1), state], dim=1))

        # FC branch: action -> [B, action_branch_out]
        xa = self.action_fc(action)

        # Combine and output
        out = self.combined_fc(torch.cat([xs, xa], dim=1))  # [B, 2708]
        return out

        # q0 = torch.sigmoid(out[:, :260])
        # q = torch.stack([a + (b - a) * q0[:, i] for i, (a, b) in enumerate(self.limits_q)], dim=1)
        
        # u = self.min_max_sigmoid(out[:, 260:1354])
        # delta = self.beta_silu(out[:, 1354:])
        
        # return torch.cat([q, u, delta], dim=1)



class GNNACOPFModel(nn.Module):
    """
    Wrapper combining GNNEncoder + GNNPINNPF with the same interface as ACOPFM.

    This allows the existing loss.py / trainer.py code to work unchanged:
      - model.encode(state)  -> action tensor  (no tuple)
      - model.pinn_pf(action, state) -> q_u_delta tensor
    """

    def __init__(self, state_dim, act_dim, intermediate_dim, latent_dim,
                 num_buses, ybus_matrix, limits, limits_q,
                 gnn_hidden_per_bus=4):
        super(GNNACOPFModel, self).__init__()

        self.state_dim = state_dim
        self.act_dim   = act_dim
        self.limits    = limits
        self.limits_q  = limits_q
        self.num_buses = num_buses

        self.encoder = GNNEncoder(
            state_dim, act_dim, intermediate_dim, latent_dim,
            num_buses, ybus_matrix, limits, "GNN",
            gnn_hidden_per_bus=gnn_hidden_per_bus
        )
        self.pinn_pf_ = GNNPINNPF(
            act_dim, state_dim, intermediate_dim, limits_q,
            num_buses, ybus_matrix,
            gnn_hidden_per_bus=gnn_hidden_per_bus
        )

    def encode(self, state):
        """Returns action tensor (compatible with ACOPFM interface)."""
        return self.encoder.encode(state)

    def pinn_pf(self, action, state):
        """Delegate to GNNPINNPF."""
        return self.pinn_pf_(action, state)
