import torch
import sys
import os

# 将 bao_server 添加到路径，以便可以导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
bao_server_path = os.path.join(current_dir, 'bao_server')
sys.path.append(bao_server_path)

# 导入两个模型定义
from bao_server.net import BaoNet
from bao_server.gnto_adapter import GNTOModel

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    print("="*50)
    print("Model Parameter Counter")
    print("="*50)

    # 1. 统计 BaoNet (TCNN) 参数
    # BaoNet 的输入通道数通常取决于操作符数量。
    # 根据 bao_server/featurize.py，基础操作符约 5-7 个，加上 Cost/Rows 等特征
    # 我们假设一个典型的输入通道数，例如 15 (这只是个近似值，用于实例化)
    # 实际运行时，这个值由 TreeFeaturizer 决定
    dummy_in_channels = 15
    try:
        bao_model = BaoNet(in_channels=dummy_in_channels)
        bao_params = count_parameters(bao_model)
        print(f"[BaoNet / TCNN]")
        print(f"Input Channels (Approx): {dummy_in_channels}")
        print(f"Total Parameters: {bao_params:,}")
    except Exception as e:
        print(f"[BaoNet] Error initializing: {e}")

    print("-" * 30)

    # 2. 统计 GNTOModel (GAT) 参数
    # GNTOModel 需要三个主要参数：Node Types, Columns, Ops 的数量
    # 我们使用一些合理的默认值（与 GNTOFeaturizer 中的默认值类似）
    num_node_types = 100
    num_cols = 1000
    num_ops = 50
    
    try:
        gnto_model = GNTOModel(
            num_node_types=num_node_types,
            num_cols=num_cols,
            num_ops=num_ops,
            # 以下是默认超参数
            type_dim=16,
            hidden_dim=64,
            out_dim=64,
            heads=8
        )
        gnto_params = count_parameters(gnto_model)
        print(f"[GNTOModel / GAT]")
        print(f"Config: NodeTypes={num_node_types}, Cols={num_cols}, Ops={num_ops}")
        print(f"Total Parameters: {gnto_params:,}")
    except Exception as e:
        print(f"[GNTOModel] Error initializing: {e}")

    print("="*50)

if __name__ == "__main__":
    main()

