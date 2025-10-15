import numpy as np
import pandas as pd
import pywt
import nibabel as nib
import os
import glob
import json
import re
from scipy import interpolate
from scipy.stats import skew, kurtosis
from tqdm import tqdm
import h5py


class Config:
    def __init__(self):
        # 核心参数（仅保留必要项）
        self.target_length = 100  # 统一时间序列长度
        self.num_scales = 64  # 小波尺度数
        self.freq_range = (0.01, 0.1)  # 分析频率范围
        self.target_rois = 116  # 目标脑区数（AAL模板，保证格式统一）
        self.wavelet = 'cmor1.5-1.0'  # 复Morlet小波

        # 路径（请确认与实际一致）
        self.PHENOTYPIC_FILE = "C:/Users/29970/Desktop/abide-master/Phenotypic_V1_0b_preprocessed1.csv"
        self.TIME_SERIES_DIR = "C:/Users/29970/Desktop/data/aal/Outputs/cpac/filt_noglobal/rois_aal"
        self.OUTPUT_DIR = "C:/Users/29970/Desktop/BrainNetworkTransformer-main/data"  # 无需fMRI路径，用默认TR


class ABIDEWaveletFeatureExtractor:
    def __init__(self, config: Config):
        self.cfg = config
        # 仅保留核心参数
        self.TARGET_ROIS = config.target_rois
        self.TARGET_LEN = config.target_length
        self.NUM_SCALES = config.num_scales
        self.FREQ_RANGE = config.freq_range
        self.WAVELET = config.wavelet

    # -------------------------- 1. 简化表型数据加载（只取ID和标签） --------------------------
    def load_labels(self):
        """只加载受试者ID和诊断标签，剔除无用信息"""
        print("加载受试者标签...")
        df = pd.read_csv(self.cfg.PHENOTYPIC_FILE)
        labels = {}
        for _, row in df.iterrows():
            try:
                # 统一ID格式（7位）
                sub_id = f"{int(row['SUB_ID']):07d}"
                # 只保留ASD(1)和对照(0)
                labels[sub_id] = 1 if row['DX_GROUP'] == 1 else 0 if row['DX_GROUP'] == 2 else None
                if labels[sub_id] is None:
                    del labels[sub_id]
            except:
                continue
        print(f"有效标签数: {len(labels)}")
        return labels

    # -------------------------- 2. 简化1D文件加载（无视失败文件） --------------------------
    def load_useful_1d_data(self, labels):
        """只加载能正常读取、脑区数=116的1D文件，其他直接跳过"""
        print("加载有用的1D文件...")
        # 扫描1D文件
        all_1d = glob.glob(os.path.join(self.cfg.TIME_SERIES_DIR, "**/*.1D"), recursive=True)
        if not all_1d:
            all_1d = glob.glob(os.path.join(self.cfg.TIME_SERIES_DIR, "**/*.1d"), recursive=True)
        print(f"找到 {len(all_1d)} 个1D文件，开始筛选...")

        useful_data = {}
        for file in tqdm(all_1d, desc="筛选有用文件"):
            try:
                # 1. 提取受试者ID（从文件名）
                basename = os.path.basename(file)
                id_match = re.search(r'(\d{5,7})', basename)
                if not id_match:
                    continue
                sub_id = f"{int(id_match.group(1)):07d}"  # 统一为7位ID
                if sub_id not in labels:  # 标签中没有的ID直接跳过
                    continue

                # 2. 读取1D文件（简化逻辑，遇到错误直接跳过）
                data = np.loadtxt(file, dtype=np.float32)  # 强制float32，避免类型问题

                # 3. 维度校正：必须是 [116, 时间点]
                if data.ndim == 1:
                    continue  # 单维数据（非脑区时间序列）
                # 确保脑区数=116（格式统一关键）
                if data.shape[0] == self.TARGET_ROIS:
                    ts = data
                elif data.shape[1] == self.TARGET_ROIS:
                    ts = data.T
                else:
                    continue  # 脑区数不对，跳过

                # 4. 时间序列长度过滤（避免过短数据）
                if ts.shape[1] < 30:  # 时间点太少的跳过
                    continue

                useful_data[sub_id] = {
                    'ts': ts,
                    'label': labels[sub_id],
                    'tr': 2.0  # 简化：用默认TR=2.0（无需读取fMRI，减少错误）
                }
            except Exception as e:
                # 无视任何加载错误，直接跳过
                continue

        print(f"筛选完成：{len(useful_data)} 个有用的受试者数据")
        return useful_data

    # -------------------------- 3. 统一时间序列长度（保证输入格式一致） --------------------------
    def unify_ts_length(self, useful_data):
        """统一所有时间序列长度为TARGET_LEN"""
        print(f"统一时间序列长度为 {self.TARGET_LEN}...")
        unified = {}
        for sub_id, data in tqdm(useful_data.items(), desc="统一长度"):
            ts = data['ts']
            _, curr_len = ts.shape

            # 裁剪或插值
            if curr_len > self.TARGET_LEN:
                start = (curr_len - self.TARGET_LEN) // 2
                unified_ts = ts[:, start:start + self.TARGET_LEN]
            elif curr_len < self.TARGET_LEN:
                # 线性插值
                new_ts = []
                for roi in ts:
                    x_old = np.linspace(0, 1, curr_len)
                    x_new = np.linspace(0, 1, self.TARGET_LEN)
                    f = interpolate.interp1d(x_old, roi, kind='linear', fill_value="extrapolate")
                    new_ts.append(f(x_new))
                unified_ts = np.array(new_ts, dtype=np.float32)
            else:
                unified_ts = ts

            unified[sub_id] = {
                'ts': unified_ts,
                'label': data['label'],
                'tr': data['tr']
            }
        return unified

    # -------------------------- 4. 小波变换+特征提取（保证输出格式统一） --------------------------
    def compute_wavelet_features(self, unified_data):
        """计算小波特征，输出格式：[116, 64, 72]"""
        print("提取小波特征...")
        feature_dict = {}
        # 预计算尺度（用默认TR=2.0，格式统一）
        tr = 2.0
        min_freq, max_freq = self.FREQ_RANGE
        center_freq = 1.0
        samp_freq = 1.0 / tr
        min_scale = center_freq / (max_freq * samp_freq)
        max_scale = center_freq / (min_freq * samp_freq)
        scales = np.logspace(np.log10(min_scale), np.log10(max_scale), self.NUM_SCALES)

        for sub_id, data in tqdm(unified_data.items(), desc="计算特征"):
            ts = data['ts']  # [116, 100]
            label = data['label']

            try:
                # 1. 小波变换（每个脑区）
                cwt_all = np.zeros((self.TARGET_ROIS, self.NUM_SCALES, self.TARGET_LEN), dtype=np.complex64)
                for roi in range(self.TARGET_ROIS):
                    coeff, _ = pywt.cwt(ts[roi], scales, self.WAVELET)
                    cwt_all[roi] = coeff

                # 2. 提取72维特征（32实部+32虚部+8相位）
                features = np.zeros((self.TARGET_ROIS, self.NUM_SCALES, 72), dtype=np.float32)
                for roi in range(self.TARGET_ROIS):
                    roi_cwt = cwt_all[roi]
                    # 实部特征（32维）
                    real_feat = self._get_real_feat(roi_cwt.real)
                    # 虚部特征（32维）
                    imag_feat = self._get_real_feat(roi_cwt.imag)
                    # 相位特征（8维）
                    phase_feat = self._get_phase_feat(roi_cwt)
                    # 合并
                    features[roi] = np.concatenate([real_feat, imag_feat, phase_feat], axis=1)

                feature_dict[sub_id] = {
                    'features': features,  # [116,64,72]
                    'label': label
                }
            except Exception as e:
                continue  # 计算错误的跳过

        print(f"特征提取完成：{len(feature_dict)} 个受试者的特征")
        return feature_dict

    # 辅助：提取实部/虚部特征（32维）
    def _get_real_feat(self, real_coeff):
        # 基础统计量（7维→扩展到32维）
        feat = [
            np.mean(real_coeff, 1, keepdims=True),
            np.std(real_coeff, 1, keepdims=True),
            np.max(real_coeff, 1, keepdims=True),
            np.min(real_coeff, 1, keepdims=True),
            np.sum(real_coeff ** 2, 1, keepdims=True),
            skew(real_coeff, 1, keepdims=True),
            kurtosis(real_coeff, 1, keepdims=True)
        ]
        feat = np.concatenate(feat, 1)
        # 扩展到32维（格式统一）
        if feat.shape[1] < 32:
            feat = np.tile(feat, (1, (32 + feat.shape[1] - 1) // feat.shape[1]))[:, :32]
        return feat

    # 辅助：提取相位特征（8维）
    def _get_phase_feat(self, complex_coeff):
        phase = np.angle(complex_coeff)
        phase_complex = np.exp(1j * phase)
        return np.stack([
            np.mean(phase, 1),
            np.std(phase, 1),
            np.max(phase, 1),
            np.min(phase, 1),
            np.abs(np.mean(phase_complex, 1)),
            np.mean(np.cos(phase), 1),
            np.mean(np.sin(phase), 1),
            np.var(phase, 1)
        ], axis=1)

    # -------------------------- 5. 保存特征（格式统一为HDF5+CSV） --------------------------
    def save_features(self, feature_dict):
        """保存特征，保证格式统一"""
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        print(f"保存特征到 {self.cfg.OUTPUT_DIR}...")

        # 1. HDF5（主要特征）
        h5_path = os.path.join(self.cfg.OUTPUT_DIR, "useful_wavelet_features.h5")
        with h5py.File(h5_path, 'w') as f:
            for sub_id, data in feature_dict.items():
                grp = f.create_group(sub_id)
                grp.create_dataset('features', data=data['features'], compression='gzip')
                grp.attrs['label'] = data['label']

        # 2. CSV标签（方便后续建模）
        label_df = pd.DataFrame([
            {'subject_id': sub_id, 'label': data['label']}
            for sub_id, data in feature_dict.items()
        ])
        label_df.to_csv(os.path.join(self.cfg.OUTPUT_DIR, "labels.csv"), index=False)

        # 3. 元数据（记录格式）
        meta = {
            'feature_shape': (self.TARGET_ROIS, self.NUM_SCALES, 72),
            'num_subjects': len(feature_dict),
            'asd_count': sum(1 for d in feature_dict.values() if d['label'] == 1),
            'control_count': sum(1 for d in feature_dict.values() if d['label'] == 0)
        }
        with open(os.path.join(self.cfg.OUTPUT_DIR, "meta.json"), 'w') as f:
            json.dump(meta, f, indent=2)

        print("保存完成！特征格式统一为:", meta['feature_shape'])

    # -------------------------- 6. 核心流程（串联所有步骤） --------------------------
    def run(self):
        """一键运行：加载有用数据→处理→提取特征→保存"""
        # 步骤1：加载标签
        labels = self.load_labels()
        if not labels:
            print("无有效标签，退出！")
            return
        # 步骤2：加载有用的1D数据
        useful_data = self.load_useful_1d_data(labels)
        if not useful_data:
            print("无有用数据，退出！")
            return
        # 步骤3：统一时间序列长度
        unified_data = self.unify_ts_length(useful_data)
        # 步骤4：提取小波特征
        feature_dict = self.compute_wavelet_features(unified_data)
        if not feature_dict:
            print("无有效特征，退出！")
            return
        # 步骤5：保存结果
        self.save_features(feature_dict)
        print("\n=== 最终结果 ===")
        print(f"有效受试者数: {len(feature_dict)}")
        print(f"ASD: {sum(1 for d in feature_dict.values() if d['label'] == 1)}")
        print(f"健康对照: {sum(1 for d in feature_dict.values() if d['label'] == 0)}")
        print(f"特征格式: (116脑区, 64尺度, 72特征)")


if __name__ == "__main__":
    # 初始化配置和提取器
    cfg = Config()
    extractor = ABIDEWaveletFeatureExtractor(cfg)
    # 一键运行
    extractor.run()