import argparse
from platform import processor

import torch
import json
import os


def main():
    parser = argparse.ArgumentParser(description='脑网络分类训练')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'extract_features'],
                        help='运行模式: train, test, extract_features')
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
        # 默认配置
        config = {
            "data": {
                "h5_path": "data/useful_wavelet_features.h5",
                "label_path": "data/labels.csv"
            },
            "model": {
                "num_rois": 116,
                "num_scales": 64,
                "num_bands": 4,
                "feat_dim": 72,
                "hidden_dim": 64,
                "num_classes": 2
            },
            "training": {
                "batch_size": 8,
                "num_epochs": 100,
                "learning_rate": 1e-4,
                "weight_decay": 1e-4,
                "train_ratio": 0.8
            },
            "paths": {
                "best_model": "best_model.pth",
                "checkpoint_dir": "checkpoints"
            }
        }

        # 保存默认配置
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        with open(args.config, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"已创建默认配置文件: {args.config}")

    if args.mode == 'train':
        from trainer import train_complete_model

        # 合并配置
        train_config = {
            'h5_path': config['data']['h5_path'],
            'label_path': config['data']['label_path'],
            'num_rois': config['model']['num_rois'],
            'num_scales': config['model']['num_scales'],
            'num_bands': config['model']['num_bands'],
            'feat_dim': config['model']['feat_dim'],
            'hidden_dim': config['model']['hidden_dim'],
            'num_classes': config['model']['num_classes'],
            'batch_size': config['training']['batch_size'],
            'num_epochs': config['training']['num_epochs'],
            'learning_rate': config['training']['learning_rate'],
            'weight_decay': config['training']['weight_decay'],
            'train_ratio': config['training']['train_ratio'],
            'best_model_path': config['paths']['best_model']
        }

        # 检查数据文件
        if not os.path.exists(train_config['h5_path']):
            print(f"错误: 特征文件不存在: {train_config['h5_path']}")
            print("请先运行: python -m utils.cwt-processor")
            return

        # 开始训练
        trainer = train_complete_model(train_config)

    elif args.mode == 'extract_features':
        from cwt_processor import ABIDEWaveletFeatureExtractor, Config
        print("开始提取小波特征...")
        cfg = Config()
        extractor = ABIDEWaveletFeatureExtractor(cfg)
        extractor.run()

    elif args.mode == 'test':
        print("测试模式尚未实现")
        # 这里可以添加模型测试代码


if __name__ == "__main__":
    main()