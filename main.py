import argparse
import numpy as np
import torch
import json
import os

def main():
    parser = argparse.ArgumentParser(description='脑网络分类训练')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'extract_features', 'diagnostic'],
                        help='运行模式: train, test, extract_features, diagnostic')
    parser.add_argument('--config', type=str, default='config.json',
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='检查点路径（用于恢复训练或测试）')
    args = parser.parse_args()

    # 加载配置
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # 默认配置（使用更保守的参数）
        config = {
            "data": {
                "h5_path": "data/useful_wavelet_features.h5",
                "label_path": "data/labels.csv"
            },
            "model": {
                "num_rois": 116,
                "num_scales": 64,
                "num_bands": 4,
                "feat_dim": 28,
                "hidden_dim": 64,
                "num_classes": 2
            },
            "training": {
                "batch_size": 8,
                "num_epochs": 100,
                "learning_rate": 1e-5,  # 更小的学习率
                "weight_decay": 1e-4,
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "save_interval": 10
            },
            "paths": {
                "best_model": "result/best_model.pth",
                "checkpoint_dir": "result/checkpoints",
                "train_history": "result/train_history.json",
                "test_metrics": "result/test_metrics.json",
                "confusion_matrix": "result/confusion_matrix.png"
            }
        }

        # 保存默认配置
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        with open(args.config, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"已创建默认配置文件: {args.config}")

    # 确保result目录存在
    result_dirs = [
        os.path.dirname(config['paths']['best_model']),
        config['paths']['checkpoint_dir'],
        os.path.dirname(config['paths']['train_history']),
        os.path.dirname(config['paths']['test_metrics'])
    ]
    for dir_path in result_dirs:
        os.makedirs(dir_path, exist_ok=True)

    if args.mode == 'train':
        from trainer import train_complete_model

        if not os.path.exists(config['data']['h5_path']):
            print(f"错误: 特征文件不存在: {config['data']['h5_path']}")
            print("请先运行: python main.py --mode extract_features")
            return

        trainer = train_complete_model(config)

    elif args.mode == 'diagnostic':
        from diagnostic_train import diagnostic_train
        print("开始诊断训练...")
        trainer = diagnostic_train(config)

    elif args.mode == 'extract_features':
        from cwt_processor import ABIDEWaveletFeatureExtractor, Config
        print("开始提取小波特征...")
        cfg = Config()
        os.makedirs(os.path.dirname(cfg.OUTPUT_DIR), exist_ok=True)
        extractor = ABIDEWaveletFeatureExtractor(cfg)
        extractor.run()

    elif args.mode == 'test':
        # 测试代码保持不变...
        pass

if __name__ == "__main__":
    main()