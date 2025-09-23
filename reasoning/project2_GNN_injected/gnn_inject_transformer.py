import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from typing import Optional

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GNNInjectedTransformerLayer(nn.Module):
    """
    Transformer layer with optional GNN injection after attention or FFN.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "gelu",
                 gnn_type: str = "gcn", use_gnn: bool = True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.gelu if activation == "gelu" else F.relu

        # Optional GNN
        self.use_gnn = use_gnn
        if use_gnn:
            if gnn_type == "gcn":
                self.gnn = GCNConv(d_model, d_model)
            elif gnn_type == "gat":
                self.gnn = GATConv(d_model, d_model // 8, heads=8, concat=True)
            else:
                raise ValueError("gnn_type must be 'gcn' or 'gat'")
            self.gnn_norm = nn.LayerNorm(d_model)
            self.gnn_dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, edge_index: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src: (batch_size, seq_len, d_model)
            edge_index: (2, num_edges) - graph connectivity for this batch
        """
        # Self-attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Optional GNN after attention
        if self.use_gnn:
            batch_size, seq_len, d_model = src.shape
            # Reshape for GNN: (batch_size * seq_len, d_model)
            src_gnn = src.view(-1, d_model)
            
            # Apply GNN
            src_gnn_out = self.gnn(src_gnn, edge_index)
            src_gnn_out = src_gnn_out.view(batch_size, seq_len, -1)
            
            # Residual + Norm
            src = src + self.gnn_dropout(src_gnn_out)
            src = self.gnn_norm(src)

        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class GNNInjectedTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: torch.Tensor, edge_index: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, edge_index, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class GNNInjectedTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "gelu",
                 gnn_type: str = "gcn", use_gnn: bool = True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.gelu if activation == "gelu" else F.relu

        # Optional GNN
        self.use_gnn = use_gnn
        if use_gnn:
            if gnn_type == "gcn":
                self.gnn = GCNConv(d_model, d_model)
            elif gnn_type == "gat":
                self.gnn = GATConv(d_model, d_model // 8, heads=8, concat=True)
            else:
                raise ValueError("gnn_type must be 'gcn' or 'gat'")
            self.gnn_norm = nn.LayerNorm(d_model)
            self.gnn_dropout = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                edge_index: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Optional GNN after self-attention
        if self.use_gnn:
            batch_size, seq_len, d_model = tgt.shape
            tgt_gnn = tgt.view(-1, d_model)
            tgt_gnn_out = self.gnn(tgt_gnn, edge_index)
            tgt_gnn_out = tgt_gnn_out.view(batch_size, seq_len, -1)
            tgt = tgt + self.gnn_dropout(tgt_gnn_out)
            tgt = self.gnn_norm(tgt)

        # Cross-attention
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class GNNInjectedTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, edge_index: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, edge_index,
                           tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class GNNInjectedTransformer(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "gelu",
                 gnn_type: str = "gcn", use_gnn_in_encoder: bool = True,
                 use_gnn_in_decoder: bool = True):
        super().__init__()

        # Encoder
        encoder_layer = GNNInjectedTransformerLayer(
            d_model, nhead, dim_feedforward, dropout, activation,
            gnn_type, use_gnn_in_encoder
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = GNNInjectedTransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # Decoder
        decoder_layer = GNNInjectedTransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation,
            gnn_type, use_gnn_in_decoder
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = GNNInjectedTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_edge_index: torch.Tensor, tgt_edge_index: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src: (batch_size, src_len, d_model)
            tgt: (batch_size, tgt_len, d_model)
            src_edge_index: (2, E_src) - graph for source sequence
            tgt_edge_index: (2, E_tgt) - graph for target sequence
        """
        # Encode
        memory = self.encoder(src, src_edge_index, mask=src_mask,
                              src_key_padding_mask=src_key_padding_mask)

        # Decode
        output = self.decoder(tgt, memory, tgt_edge_index,
                              tgt_mask=tgt_mask,
                              memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)

        return output


# Example usage
def create_sample_graph_edge_index(seq_len, batch_size):
    """Create fully connected graph for demonstration"""
    edge_index_list = []
    offset = 0
    for b in range(batch_size):
        # Create fully connected graph for this sequence
        nodes = torch.arange(seq_len)
        row, col = torch.meshgrid(nodes, nodes, indexing='ij')
        edge_index = torch.stack([row.flatten(), col.flatten()], dim=0)
        edge_index += offset
        edge_index_list.append(edge_index)
        offset += seq_len
    return torch.cat(edge_index_list, dim=1).to(device)


if __name__ == "__main__":
    # Create model
    model = GNNInjectedTransformer(
        d_model=128,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        gnn_type="gat",  # or "gcn"
        use_gnn_in_encoder=True,
        use_gnn_in_decoder=True
    ).to(device)

    # Dummy input
    batch_size = 2
    src_len = 10
    tgt_len = 8
    d_model = 128

    src = torch.rand(batch_size, src_len, d_model).to(device)
    tgt = torch.rand(batch_size, tgt_len, d_model).to(device)

    # Create graph structures
    src_edge_index = create_sample_graph_edge_index(src_len, batch_size)
    tgt_edge_index = create_sample_graph_edge_index(tgt_len, batch_size)

    # Forward pass
    output = model(src, tgt, src_edge_index, tgt_edge_index)
    print("Model output shape:", output.shape)  # (2, 8, 128)
    print("Model parameters:", sum(p.numel() for p in model.parameters()))