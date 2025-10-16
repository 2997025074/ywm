import argparse

import numpy as np
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
                "train_ratio": 0.8,
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
            'best_model_path': config['paths']['best_model'],
            'checkpoint_dir': config['paths']['checkpoint_dir'],
            'train_history': config['paths']['train_history'],
            'save_interval': config['training']['save_interval']
        }

        # 检查数据文件
        if not os.path.exists(train_config['h5_path']):
            print(f"错误: 特征文件不存在: {train_config['h5_path']}")
            print("请先运行: python main.py --mode extract_features")
            return

        # 开始训练
        trainer = train_complete_model(train_config)

    elif args.mode == 'extract_features':
        from cwt_processor import ABIDEWaveletFeatureExtractor, Config
        print("开始提取小波特征...")
        cfg = Config()
        # 确保数据目录存在
        os.makedirs(os.path.dirname(cfg.OUTPUT_DIR), exist_ok=True)
        extractor = ABIDEWaveletFeatureExtractor(cfg)
        extractor.run()


    elif args.mode == 'test':

        print("开始测试模型...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 确保result文件夹存在

        result_dirs = [
            os.path.dirname(config['paths']['test_metrics']),
            os.path.dirname(config['paths']['confusion_matrix'])
        ]

        for dir_path in result_dirs:
            os.makedirs(dir_path, exist_ok=True)

        # 加载模型

        from model.classifier import BrainNetworkClassifier
        model = BrainNetworkClassifier(

            num_rois=config['model']['num_rois'],
            num_scales=config['model']['num_scales'],
            num_bands=config['model']['num_bands'],
            feat_dim=config['model']['feat_dim'],
            hidden_dim=config['model']['hidden_dim'],
            num_classes=config['model']['num_classes']
        )

        from trainer import BrainNetworkTrainer

        trainer = BrainNetworkTrainer(model, device)
        # 加载检查点（默认使用最佳模型）
        checkpoint_path = args.checkpoint or config['paths']['best_model']
        if not os.path.exists(checkpoint_path):
            print(f"错误：检查点文件不存在 - {checkpoint_path}")
            return

        trainer.load_checkpoint(checkpoint_path)

        # 创建测试数据加载器

        from utils.dataloader import create_data_loaders
        # 修正：接收四个返回值，提取第三个作为test_loader
        _, _, test_loader, _ = create_data_loaders(
            config['data']['h5_path'],
            config['data']['label_path'],
            batch_size=config['training']['batch_size'],
            train_ratio=config['training']['train_ratio']
        )

        # 执行测试

        from utils.metrics import ClassificationMetrics
        metrics_calc = ClassificationMetrics(num_classes=config['model']['num_classes'])
        model.eval()
        with torch.no_grad():

            for features, labels in test_loader:
                features = features.to(device)
                labels = labels.to(device)
                predictions, _ = model(features)
                metrics_calc.update(predictions, labels)

        # 计算并保存测试指标

        test_metrics = metrics_calc.compute()
        with open(config['paths']['test_metrics'], 'w') as f:
            # 转换numpy数组为列表以支持JSON序列化
            serializable_metrics = {}
            for k, v in test_metrics.items():
                if isinstance(v, np.ndarray):
                    serializable_metrics[k] = v.tolist()
                else:
                    serializable_metrics[k] = v
            json.dump(serializable_metrics, f, indent=2)

        print(f"测试指标已保存到: {config['paths']['test_metrics']}")


if __name__ == "__main__":
    main()