import torch
import torch.nn as nn
import numpy as np
import json
import re
from collections import defaultdict
import os
import sys

# Add current directory to path to allow imports from gnto_models
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from torch_geometric.data import Data, Batch
except ImportError:
    print("Warning: torch_geometric not installed. GNTO models will not work.")
    Data = object
    Batch = object

from gnto_models.NodeEncoder import NodeEncoder_V2
from gnto_models.TreeEncoder import GATv2TreeEncoder_V3
from gnto_models.PredictionHead import PredictionHead_V2

class GNTOModel(nn.Module):
    def __init__(self, num_node_types, num_cols, num_ops, 
                 type_dim=16, hidden_dim=64, out_dim=64, heads=8):
        super().__init__()
        self.node_enc = NodeEncoder_V2(
            num_node_types=num_node_types,
            num_cols=num_cols,
            num_ops=num_ops,
            type_dim=type_dim,
            out_dim=hidden_dim,
            hidden_dim=hidden_dim
        )
        # GAT Tree Encoder
        self.tree_enc = GATv2TreeEncoder_V3(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            heads1=heads,
            heads2=4,
            pooling="mean"
        )
        # Prediction Head
        # Ensure the input dimension matches tree_enc output
        self.head = PredictionHead_V2(
            in_dim=out_dim,  # This should be 64
            out_dim=1,
            hidden_dims=(32, 32) # Reduce hidden dims to avoid shape mismatch if any
        )

    def forward(self, data):
        # print("GNTOModel forward called! Processing batch size:", data.num_graphs if hasattr(data, 'num_graphs') else 1)
        if isinstance(data, list):
            # If passed a list of Data objects, batch them
            data = Batch.from_data_list(data)
            
        # Check device of the model's parameters
        device = next(self.parameters()).device
        
        # Move data to the same device
        if device.type == 'cuda':
             data = data.to(device)
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Move extracted tensors to the same device as the model
        if device.type == 'cuda':
             x = x.to(device)
             edge_index = edge_index.to(device)
             if batch is not None:
                 batch = batch.to(device)

        # 1. Node Embedding
        # Ensure x is on the same device as node_enc parameters
        enc_device = next(self.node_enc.parameters()).device
        if x.device != enc_device:
             x = x.to(enc_device)
        h = self.node_enc(x)
        
        # DEBUG: Ensure h stayed on device
        if h.device != enc_device:
            # print(f"WARNING: h fell back to {h.device}, moving back to {enc_device}")
            h = h.to(enc_device)

        # 2. Tree Embedding
        # Ensure edge_index and batch are on the same device as tree_enc parameters
        # Note: GATv2Conv usually expects input node features x (h here) and edge_index to be on the same device
        # We already ensured x/h is on the correct device in step 1.
        # Now we double check edge_index and batch.
        tree_device = next(self.tree_enc.parameters()).device
        
        # Explicitly sync everything to tree_device (which should be same as enc_device)
        if h.device != tree_device: h = h.to(tree_device)
        if edge_index.device != tree_device: edge_index = edge_index.to(tree_device)
        if batch is not None and batch.device != tree_device: batch = batch.to(tree_device)
             
        g = self.tree_enc(h, edge_index, batch)
        
        # 3. Prediction
        out = self.head(g)
        return out

class GNTOFeaturizer:
    def __init__(self, max_preds=3):
        self.max_preds = max_preds
        
        # Vocabularies with some defaults
        self.node_type_vocab = dict()
        self.col_vocab = dict()
        self.op_vocab = dict()
        
        # Initialize common ops
        ops = ['=', '!=', '<', '>', '<=', '>=', '~~', '~~*', '!~~', '!~~*']
        for op in ops:
            if op not in self.op_vocab:
                self.op_vocab[op] = len(self.op_vocab)
            
        # Initialize common node types (optional, can learn online)
        
    def num_node_types(self):
        return max(100, len(self.node_type_vocab) + 1)
        
    def num_cols(self):
        return max(1000, len(self.col_vocab) + 1)
        
    def num_ops(self):
        return max(50, len(self.op_vocab) + 1)

    def _parse_val(self, val):
        try:
            return float(val)
        except:
            pass
        s = str(val).strip()
        if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
            return s[1:-1]
        return s

    def _parse_predicates(self, node):
        preds = []
        # Regex to match simple predicates: col op val
        # This is a heuristic. Real parsing is harder.
        # GNTO DataPreprocessor logic is better but requires db_info.
        # Here we try to extract useful info.
        keys = ["Filter", "Index Cond", "Recheck Cond", "Join Filter", "Hash Cond", "Merge Cond"]
        for key in keys:
            if key in node:
                try:
                    cond = str(node[key])
                    # Remove parens
                    clean = cond.replace('(', '').replace(')', '')
                    parts = re.split(r'\s+AND\s+', clean)
                    for p in parts:
                        # Try to find op
                        found = False
                        # Iterating over keys() of a dict is pickle-safe
                        for op in self.op_vocab.keys():
                            if op in p:
                                # split by op
                                split_p = p.split(op)
                                if len(split_p) == 2:
                                    lhs = split_p[0].strip()
                                    rhs = split_p[1].strip()
                                    preds.append((lhs, op, rhs))
                                    found = True
                                    break
                        if not found:
                            # Fallback or simple split by space
                            pass
                except Exception:
                    # If predicate parsing fails (e.g. malformed string), ignore it
                    continue
        return preds

    def _process_node(self, node, update_vocab=False):
        nt = node.get("Node Type", "Unknown")
        if update_vocab:
            if nt not in self.node_type_vocab:
                self.node_type_vocab[nt] = len(self.node_type_vocab)
            nt_id = self.node_type_vocab[nt]
        else:
            nt_id = self.node_type_vocab.get(nt, 0) # 0 as Unknown
        
        rows = float(node.get("Plan Rows", 0))
        width = float(node.get("Plan Width", 0))
        
        rows = np.log1p(rows)
        width = np.log1p(width)
        
        # Try to parse predicates if they exist
        raw_preds = self._parse_predicates(node)
        pred_vecs = []
        
        # If raw_preds is empty (e.g. original Bao PG extension), this loop is skipped
        # and we pad with zeros below.
        for lhs, op, rhs in raw_preds[:self.max_preds]:
            if update_vocab:
                if lhs not in self.col_vocab:
                    self.col_vocab[lhs] = len(self.col_vocab)
                lhs_id = self.col_vocab[lhs]
                
                if op not in self.op_vocab:
                    self.op_vocab[op] = len(self.op_vocab)
                op_id = self.op_vocab[op]
            else:
                lhs_id = self.col_vocab.get(lhs, 0)
                op_id = self.op_vocab.get(op, 0)
                
            rhs_val = self._parse_val(rhs)
            is_join = False
            rhs_feat = 0.0
            
            if isinstance(rhs_val, str):
                try:
                    rhs_feat = float(rhs_val)
                except:
                    if update_vocab:
                        if rhs_val not in self.col_vocab:
                            self.col_vocab[rhs_val] = len(self.col_vocab)
                        rhs_feat = float(self.col_vocab[rhs_val])
                    else:
                        rhs_feat = float(self.col_vocab.get(rhs_val, 0))
                    is_join = True
            else:
                rhs_feat = float(rhs_val)
                
            pred_vecs.append([float(lhs_id), float(op_id), rhs_feat, 1.0 if is_join else 0.0])
            
        # Pad with zeros to ensure fixed feature size
        while len(pred_vecs) < self.max_preds:
            pred_vecs.append([0.0, 0.0, 0.0, 0.0])
            
        pred_flat = [x for p in pred_vecs for x in p]
        return [float(nt_id), rows, width] + pred_flat

    def plan_to_graph(self, plan_json, update_vocab=False):
        node_feats = []
        edges = []
        
        curr_idx = 0
        
        def recurse(node, parent_idx):
            nonlocal curr_idx
            my_idx = curr_idx
            curr_idx += 1
            
            feat = self._process_node(node, update_vocab=update_vocab)
            node_feats.append(feat)
            
            if parent_idx != -1:
                edges.append([parent_idx, my_idx])
                edges.append([my_idx, parent_idx])
                
            if "Plans" in node:
                for child in node["Plans"]:
                    recurse(child, my_idx)
                    
        recurse(plan_json, -1)
        
        x = torch.tensor(node_feats, dtype=torch.float32)
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            
        return Data(x=x, edge_index=edge_index)

    def fit(self, plans):
        # Online learning of vocab
        for p in plans:
            if isinstance(p, str):
                p = json.loads(p)
            root = p["Plan"] if "Plan" in p else p
            # Just running transformation updates the vocab as a side effect
            self.plan_to_graph(root, update_vocab=True)

    def transform(self, plans):
        print(f"GNTOFeaturizer.transform called with {len(plans)} plans.")
        data_list = []
        for p in plans:
            if isinstance(p, str):
                p = json.loads(p)
            root = p["Plan"] if "Plan" in p else p
            data_list.append(self.plan_to_graph(root, update_vocab=False))
        return data_list


