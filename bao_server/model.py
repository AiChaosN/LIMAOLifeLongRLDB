import json
import numpy as np
import torch
import torch.optim
import joblib
import os
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
import random

# Try to import PyG loader
try:
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Data, Batch
except ImportError:
    # Fallback or just let it fail later if used
    from torch.utils.data import DataLoader
    
from gnto_adapter import GNTOModel, GNTOFeaturizer

CUDA = torch.cuda.is_available()

def _nn_path(base):
    return os.path.join(base, "nn_weights")

def _x_transform_path(base):
    return os.path.join(base, "x_transform")

def _y_transform_path(base):
    return os.path.join(base, "y_transform")

def _channels_path(base):
    return os.path.join(base, "channels")

def _n_path(base):
    return os.path.join(base, "n")

def _inv_log1p(x):
    return np.exp(x) - 1

class BaoRegression:
    def __init__(self, verbose=False, have_cache_data=False):
        self.__net = None
        self.__verbose = verbose

        log_transformer = preprocessing.FunctionTransformer(
            np.log1p, _inv_log1p,
            validate=True)
        scale_transformer = preprocessing.MinMaxScaler()

        self.__pipeline = Pipeline([("log", log_transformer),
                                    ("scale", scale_transformer)])
        
        # Use GNTO Featurizer instead of TreeFeaturizer
        self.__tree_transform = GNTOFeaturizer()
        self.__have_cache_data = have_cache_data
        self.__n = 0
        
    def __log(self, *args):
        if self.__verbose:
            print(*args)

    def num_items_trained_on(self):
        return self.__n
            
    def load(self, path):
        with open(_n_path(path), "rb") as f:
            self.__n = joblib.load(f)
        
        # Load the featurizer (vocabularies)
        with open(_x_transform_path(path), "rb") as f:
            self.__tree_transform = joblib.load(f)
            
        # Reconstruct model with correct dimensions
        self.__net = GNTOModel(
            num_node_types=self.__tree_transform.num_node_types(),
            num_cols=self.__tree_transform.num_cols(),
            num_ops=self.__tree_transform.num_ops()
        )
        self.__net.load_state_dict(torch.load(_nn_path(path)))
        self.__net.eval()
        
        with open(_y_transform_path(path), "rb") as f:
            self.__pipeline = joblib.load(f)

    def save(self, path):
        # try to create a directory here
        os.makedirs(path, exist_ok=True)
        
        torch.save(self.__net.state_dict(), _nn_path(path))
        with open(_y_transform_path(path), "wb") as f:
            joblib.dump(self.__pipeline, f)
        with open(_x_transform_path(path), "wb") as f:
            joblib.dump(self.__tree_transform, f)
        # We don't save channels anymore as they are dynamic/fixed in GNTO
        # with open(_channels_path(path), "wb") as f:
        #     joblib.dump(self.__in_channels, f)
        with open(_n_path(path), "wb") as f:
            joblib.dump(self.__n, f)

    def fit(self, X, y, epochs=100):
        print("Passed to model fit - X count:", len(X))
        if isinstance(y, list):
            y = np.array(y)

        X = [json.loads(x) if isinstance(x, str) else x for x in X]
        self.__n = len(X)
            
        y = self.__pipeline.fit_transform(y.reshape(-1, 1)).astype(np.float32)
        
        # Featurize (update vocab)
        self.__tree_transform.fit(X)
        graphs = self.__tree_transform.transform(X)
        
        # Attach targets
        data_list = []
        for g, target in zip(graphs, y):
            g.y = torch.tensor([target], dtype=torch.float)
            data_list.append(g)

        # Init model if needed
        if self.__net is None:
            self.__net = GNTOModel(
                num_node_types=self.__tree_transform.num_node_types(),
                num_cols=self.__tree_transform.num_cols(),
                num_ops=self.__tree_transform.num_ops()
            )
        
        if CUDA:
            self.__net = self.__net.cuda()

        optimizer = torch.optim.Adam(self.__net.parameters())
        loss_fn = torch.nn.MSELoss()
        
        # Use PyG DataLoader which handles batching of graphs
        dataset = DataLoader(data_list, batch_size=16, shuffle=True)
        
        losses = []
        for epoch in range(epochs):
            loss_accum = 0
            num_batches = 0
            self.__net.train()
            for batch in dataset:
                if CUDA:
                    batch = batch.cuda()
                    
                pred = self.__net(batch)
                # pred shape [B, 1], batch.y shape [B, 1]
                loss = loss_fn(pred.view(-1), batch.y.view(-1))
                loss_accum += loss.item()
                num_batches += 1
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_accum /= max(1, num_batches)
            losses.append(loss_accum)
            if epoch % 15 == 0:
                self.__log("Epoch", epoch, "training loss:", loss_accum)

            # stopping condition
            if len(losses) > 10 and losses[-1] < 0.1:
                last_two = np.min(losses[-2:])
                if last_two > losses[-10] or (losses[-10] - last_two < 0.0001):
                    self.__log("Stopped training from convergence condition at epoch", epoch)
                    break
        else:
            self.__log("Stopped training after max epochs")

    # Argument module_assigner is kept for compatibility with main.py but ignored
    def predict(self, X, module_assigner=None):
        if not isinstance(X, list):
            X = [X]
        X = [json.loads(x) if isinstance(x, str) else x for x in X]

        graphs = self.__tree_transform.transform(X)
        
        # Batch prediction
        loader = DataLoader(graphs, batch_size=len(graphs), shuffle=False)
        
        self.__net.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                if CUDA:
                    batch = batch.cuda()
                out = self.__net(batch)
                preds.append(out.cpu().numpy())
        
        if len(preds) > 0:
            pred_raw = np.concatenate(preds, axis=0)
        else:
            pred_raw = np.array([])
            
        return self.__pipeline.inverse_transform(pred_raw)
