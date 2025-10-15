import argparse
import os
import json
import logging
import random
import numpy as np
import torch
from datetime import datetime


def set_seed(seed=42):
    """设置随机种子，保证实验可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(mode, log_dir="logs"):
    """配置日志系统，同时输出到控制台和文件"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/{mode}_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def validate_config(config):
    """验证配置文件的完整性，确保必要参数存在"""
    required_keys = {
        "data": ["h5_path", "label_path"],
        "model": ["num_rois", "num_scales", "num_bands", "feat_dim", "hidden_dim", "num_classes"],
        "training": ["batch_size", "num_epochs", "learning_rate", "weight_decay", "train_ratio"],
        "paths": ["best_model", "checkpoint_dir"]
    }

    for section, keys in required_keys.items():
        if section not in config:
            raise ValueError(f"配置文件缺少必要部分: {section}")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"配置文件 '{section}' 部分缺少必要参数: {key}")
    return True


def main():
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description='脑网络分类训练与评估')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'extract_features'],
                        help='运行模式: train, test, extract_features')
    parser.add_argument('--config', type=str, default='config.json',
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='检查点路径（用于恢复训练或测试）')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备 (例如: cuda, cpu)，默认自动选择')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子，保证实验可复现性')
    args = parser.parse_args()

    # 2. 初始化基础设置
    set_seed(args.seed)
    logger = setup_logging(args.mode)
    logger.info(f"启动模式: {args.mode}")
    logger.info(f"使用随机种子: {args.seed}")

    # 3. 设备配置
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用计算设备: {device}")

    # 4. 加载并验证配置
    try:
        if os.path.exists(args.config):
            with open(args.config, 'r') as f:
                config = json.load(f)
            logger.info(f"成功加载配置文件: {args.config}")
        else:
            # 生成默认配置
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
                    "save_interval": 10  # 新增：检查点保存间隔
                },
                "paths": {
                    "best_model": "best_model.pth",
                    "checkpoint_dir": "checkpoints",
                    "log_dir": "logs"  # 新增：日志目录
                }
            }
            # 保存默认配置
            os.makedirs(os.path.dirname(args.config), exist_ok=True)
            with open(args.config, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"已创建默认配置文件: {args.config}")

        # 验证配置完整性
        validate_config(config)
        logger.info("配置文件验证通过")

    except Exception as e:
        logger.error(f"配置文件处理失败: {str(e)}", exc_info=True)
        return

    # 5. 确保输出目录存在
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)

    # 6. 根据运行模式执行对应逻辑
    try:
        if args.mode == 'train':
            from trainer import train_complete_model
            logger.info("===== 启动训练模式 =====")

            # 准备训练配置
            train_config = {
                **config['data'], **config['model'],
                **config['training'], **config['paths'],
                'device': device,
                'checkpoint': args.checkpoint  # 传递检查点路径
            }

            # 检查数据文件
            if not os.path.exists(train_config['h5_path']):
                logger.error(f"特征文件不存在: {train_config['h5_path']}")
                logger.info("请先运行: python -m utils.cwt_processor")
                return

            # 启动训练
            trainer = train_complete_model(train_config)
            logger.info("训练流程完成")

        elif args.mode == 'extract_features':
            from cwt_processor import ABIDEWaveletFeatureExtractor, Config
            logger.info("===== 启动特征提取模式 =====")

            # 配置特征提取器
            cfg = Config(
                h5_output_path=config['data']['h5_path'],
                label_path=config['data']['label_path']
            )
            extractor = ABIDEWaveletFeatureExtractor(cfg)
            extractor.run()
            logger.info("特征提取完成")

        elif args.mode == 'test':
            from trainer import BrainNetworkTrainer
            from model.classifier import BrainNetworkClassifier
            from utils.dataloader import create_data_loaders
            from utils.metrics import ClassificationMetrics

            logger.info("===== 启动测试模式 =====")

            if not args.checkpoint:
                logger.error("测试模式必须指定检查点路径: --checkpoint")
                return

            if not os.path.exists(args.checkpoint):
                logger.error(f"检查点文件不存在: {args.checkpoint}")
                return

            # 准备测试配置
            test_config = {
                **config['data'], **config['model'],
                **config['training'], **config['paths'],
                'device': device
            }

            # 加载测试数据
            _, test_loader, _ = create_data_loaders(
                test_config['h5_path'],
                test_config['label_path'],
                batch_size=test_config['batch_size'],
                train_ratio=test_config['train_ratio'],
                is_test=True  # 新增参数：标识为测试模式，返回测试集
            )

            # 创建模型
            model = BrainNetworkClassifier(
                num_rois=test_config['num_rois'],
                num_scales=test_config['num_scales'],
                num_bands=test_config['num_bands'],
                feat_dim=test_config['feat_dim'],
                hidden_dim=test_config['hidden_dim'],
                num_classes=test_config['num_classes']
            )

            # 创建训练器并加载检查点
            trainer = BrainNetworkTrainer(
                model, device,
                learning_rate=test_config['learning_rate'],
                weight_decay=test_config['weight_decay']
            )
            trainer.setup_training()
            trainer.load_checkpoint(args.checkpoint)

            # 执行测试
            logger.info("开始模型测试...")
            model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for features, labels in test_loader:
                    features = features.to(device)
                    labels = labels.to(device)
                    preds, _ = model(features)
                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())

            # 计算测试指标
            metrics_calc = ClassificationMetrics()
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            metrics = metrics_calc.compute()

            logger.info("===== 测试结果 =====")
            logger.info(f"测试准确率: {metrics['accuracy']:.4f}")
            logger.info(f"测试AUC: {metrics.get('auc', 0.0):.4f}")
            logger.info(f"测试精确率: {metrics['precision']:.4f}")
            logger.info(f"测试召回率: {metrics['recall']:.4f}")
            logger.info(f"测试F1分数: {metrics['f1']:.4f}")

    except Exception as e:
        logger.error(f"运行过程出错: {str(e)}", exc_info=True)
        return


if __name__ == "__main__":
    main()
