import numpy as np
import pywt
import pickle
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedShuffleSplit  # 用于分层抽样划分数据集


class CWT_Processor:
    def __init__(
            self, dataset: np.lib.npyio.NpzFile, band: np.ndarray, wavelet: str = 'cmor1.5-1.0',
    ) -> None:
        self.dataset = dataset
        self.band = band
        self.wavelet = wavelet

        # 检查数据集必需的键（新增labels键检查，用于划分时保持类别平衡）
        required_keys = ['data', 'tr', 'labels']
        if not all(key in self.dataset for key in required_keys):
            raise KeyError(f"The dataset must contain keys: {required_keys}")

    def _get_CWT(self, in_seq: np.ndarray, tr: float) -> np.ndarray:
        fs = 1 / tr
        freq = self.band / fs
        scale = pywt.frequency2scale(self.wavelet, freq)
        cwt_data, _ = pywt.cwt(in_seq, scale, self.wavelet, axis=0)
        cwt_data = cwt_data.transpose(1, 0, 2)
        return np.array(cwt_data)

    def get_CWT_dataset(self) -> np.ndarray:
        all_cwt = []

        for i in tqdm(range(len(self.dataset['data']))):
            data_raw = self.dataset['data'][i]
            t_r = self.dataset['tr'][i]
            cwt_data = self._get_CWT(data_raw, t_r)  # [Time, n_band, ROIs]
            all_cwt.append(cwt_data)

        return np.array(all_cwt, dtype=np.complex64)


if __name__ == '__main__':
    # 1. 配置参数（与ABIDE.yaml中的划分比例一致）
    raw_path = '../abide_cc200_combined.npz'  # 原始npz文件路径
    save_dir = '../BWN/data/'  # 保存pkl文件的目录
    test_size = 0.1  # 测试集比例（对应yaml中的test_set: 0.1）
    random_state = 42  # 随机种子，确保划分结果可复现
    band = np.linspace(0.01, 0.1, 5)  # 与原代码保持一致的频率带
    wavelet = 'cmor1.5-1.0'

    # 2. 创建保存目录（如果不存在）
    import os
    os.makedirs(save_dir, exist_ok=True)

    # 3. 加载原始数据并进行CWT处理
    dataset = np.load(raw_path, allow_pickle=True)
    cwt_processor = CWT_Processor(
        dataset=dataset,
        band=band,
        wavelet=wavelet
    )
    cwt_results = cwt_processor.get_CWT_dataset()  # CWT处理后的特征
    labels = dataset['labels']  # 标签（用于分层划分）
    # 若原始数据有样本键值（如被试ID），可一并提取（此处假设存在'keys'键）
    keys = dataset.get('keys', np.arange(len(labels)))  # 默认为索引

    # 4. 分层划分训练验证集和测试集（保持类别分布一致）
    # 注意：这里只划分测试集，训练集和验证集的进一步拆分由main.py完成
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )
    # 获取划分索引（只取第一次划分结果）
    train_valid_idx, test_idx = next(sss.split(cwt_results, labels))

    # 5. 拆分数据
    # 训练验证集（后续会被main.py拆分为训练集和验证集）
    train_valid_data = cwt_results[train_valid_idx]
    train_valid_labels = labels[train_valid_idx]
    train_valid_keys = keys[train_valid_idx]

    # 测试集（固定，不参与训练）
    test_data = cwt_results[test_idx]
    test_labels = labels[test_idx]
    test_keys = keys[test_idx]

    # 6. 保存为pkl文件（与yaml配置中的路径对应）
    train_valid_save_path = os.path.join(save_dir, 'cmor_train_valid_CWT_TR5.pkl')
    test_save_path = os.path.join(save_dir, 'cmor_test_CWT_TR5.pkl')

    # 保存训练验证集（包含特征、标签、键值）
    with open(train_valid_save_path, 'wb') as f:
        pickle.dump((train_valid_data, train_valid_labels, train_valid_keys), f)

    # 保存测试集（包含特征、标签、键值）
    with open(test_save_path, 'wb') as f:
        pickle.dump((test_data, test_labels, test_keys), f)

    print(f"CWT处理及划分完成，结果已保存：")
    print(f"训练验证集: {train_valid_save_path}，形状: {train_valid_data.shape}，样本数: {len(train_valid_data)}")
    print(f"测试集: {test_save_path}，形状: {test_data.shape}，样本数: {len(test_data)}")
