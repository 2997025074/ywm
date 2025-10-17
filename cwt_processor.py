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
        # 核心参数
        self.target_length = 100  # 统一时间序列长度
        self.num_scales = 64  # 小波尺度数
        self.freq_range = (0.01, 0.1)  # 分析频率范围(Hz)
        self.target_rois = 116  # 目标脑区数（AAL模板）
        self.wavelet = 'cmor1.5-1.0'  # 复Morlet小波

        # 异常检测参数
        self.low_variance_threshold = 1e-6  # 低变异性阈值
        self.signal_range_threshold = 3.0  # 信号范围阈值(Z-score)
        self.min_valid_timepoints = 30  # 最小有效时间点数

        # 路径配置
        self.PHENOTYPIC_FILE = "C:/Users/29970/Desktop/abide-master/Phenotypic_V1_0b_preprocessed1.csv"
        self.TIME_SERIES_DIR = "C:/Users/29970/Desktop/data/aal/Outputs/cpac/filt_noglobal/rois_aal"
        self.FMRI_DATA_DIR = "C:/Users/29970/Desktop/data/aal/Outputs/cpac/filt_noglobal/func_preproc"
        self.OUTPUT_DIR = "C:/Users/29970/Desktop/Brainwavegnn/data"


class ABIDEWaveletFeatureExtractor:
    def __init__(self, config: Config):
        self.cfg = config
        self.TARGET_ROIS = config.target_rois
        self.TARGET_LEN = config.target_length
        self.NUM_SCALES = config.num_scales
        self.FREQ_RANGE = config.freq_range
        self.WAVELET = config.wavelet
        self.FMRI_DIR = config.FMRI_DATA_DIR

    def load_labels(self):
        """加载受试者ID和诊断标签"""
        print("加载受试者标签...")
        df = pd.read_csv(self.cfg.PHENOTYPIC_FILE)
        labels = {}
        for _, row in df.iterrows():
            try:
                sub_id = f"{int(row['SUB_ID']):07d}"
                dx_group = row['DX_GROUP']
                label = 1 if dx_group == 1 else 0 if dx_group == 2 else None
                if label is not None:
                    labels[sub_id] = label
            except:
                continue
        print(f"有效标签数: {len(labels)}")
        return labels

    def find_matching_fmri_file(self, subject_id):
        """根据受试者ID查找对应的fMRI文件"""
        patterns = [
            f"*{subject_id}*func_preproc.nii.gz",
            f"*_{subject_id}_*func_preproc.nii.gz",
        ]

        for pattern in patterns:
            search_path = os.path.join(self.FMRI_DIR, pattern)
            matching_files = glob.glob(search_path, recursive=True)
            if matching_files:
                return matching_files[0]
        return None

    def extract_and_process_tr(self, nifti_file_path):
        """从fMRI文件提取并处理TR"""
        try:
            img = nib.load(nifti_file_path)
            tr = img.header.get_zooms()[3]
            processed_tr = round(float(tr), 1)
            return processed_tr if 0 < processed_tr <= 10 else 2.0
        except:
            return 2.0

    def preprocess_time_series(self, time_series):
        """预处理时间序列，处理异常值"""
        ts_clean = np.nan_to_num(time_series, nan=0.0, posinf=0.0, neginf=0.0)

        # 检查变异性
        if np.std(ts_clean) < self.cfg.low_variance_threshold:
            # 低变异性信号，使用微小噪声
            ts_clean = ts_clean + np.random.normal(0, 1e-6, ts_clean.shape)

        # 稳健标准化
        median = np.median(ts_clean)
        mad = np.median(np.abs(ts_clean - median))
        if mad > 1e-8:
            z_scores = np.abs((ts_clean - median) / mad)
            if np.any(z_scores > self.cfg.signal_range_threshold):
                # Winsorizing处理
                q25, q75 = np.percentile(ts_clean, [25, 75])
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                ts_clean = np.clip(ts_clean, lower_bound, upper_bound)

        # 最终标准化
        return (ts_clean - np.mean(ts_clean)) / (np.std(ts_clean) + 1e-8)

    def load_useful_1d_data(self, labels):
        """加载脑区数=116的1D文件，同时读取对应fMRI的TR"""
        print("加载有用的1D文件并匹配fMRI的TR...")
        all_1d = glob.glob(os.path.join(self.cfg.TIME_SERIES_DIR, "**/*.1D"), recursive=True)
        if not all_1d:
            all_1d = glob.glob(os.path.join(self.cfg.TIME_SERIES_DIR, "**/*.1d"), recursive=True)

        useful_data = {}
        for file in tqdm(all_1d, desc="筛选文件"):
            try:
                basename = os.path.basename(file)
                id_match = re.search(r'(\d{5,7})', basename)
                if not id_match:
                    continue
                sub_id = f"{int(id_match.group(1)):07d}"
                if sub_id not in labels:
                    continue

                data = np.loadtxt(file, dtype=np.float32)
                if data.ndim == 1 or data.shape[1] < self.cfg.min_valid_timepoints:
                    continue

                # 维度校正
                if data.shape[0] == self.TARGET_ROIS:
                    ts = data
                elif data.shape[1] == self.TARGET_ROIS:
                    ts = data.T
                else:
                    continue

                # 预处理每个脑区
                ts_processed = np.zeros_like(ts)
                for roi_idx in range(ts.shape[0]):
                    ts_processed[roi_idx] = self.preprocess_time_series(ts[roi_idx])

                # 读取TR
                fmri_file = self.find_matching_fmri_file(sub_id)
                tr = self.extract_and_process_tr(fmri_file) if fmri_file else 2.0

                useful_data[sub_id] = {
                    'ts': ts_processed,
                    'label': labels[sub_id],
                    'tr': tr
                }
            except:
                continue

        print(f"筛选完成：{len(useful_data)} 个有用的受试者数据")
        return useful_data

    def unify_ts_length(self, useful_data):
        """统一所有时间序列长度为TARGET_LEN"""
        print(f"统一时间序列长度为 {self.TARGET_LEN}...")
        unified = {}
        for sub_id, data in tqdm(useful_data.items(), desc="统一长度"):
            ts = data['ts']
            _, curr_len = ts.shape

            if curr_len > self.TARGET_LEN:
                start = (curr_len - self.TARGET_LEN) // 2
                unified_ts = ts[:, start:start + self.TARGET_LEN]
            elif curr_len < self.TARGET_LEN:
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

    def _extract_meaningful_features(self, complex_coeff):
        """
        提取有意义的小波特征（28维）
        包含频带间联系信息，为后续图构建做准备
        """
        magnitude = np.abs(complex_coeff)
        phase = np.angle(complex_coeff)
        real_part = complex_coeff.real
        imag_part = complex_coeff.imag

        features = []

        # 1. 核心幅度特征 (8维)
        mag_features = [
            np.mean(magnitude, axis=1, keepdims=True),  # 平均幅度
            np.std(magnitude, axis=1, keepdims=True),  # 幅度变异性
            np.max(magnitude, axis=1, keepdims=True),  # 峰值幅度
            np.sum(magnitude ** 2, axis=1, keepdims=True),  # 总能量
            skew(magnitude, axis=1, keepdims=True),  # 幅度分布偏度
            kurtosis(magnitude, axis=1, keepdims=True),  # 幅度分布峰度
            np.median(magnitude, axis=1, keepdims=True),  # 幅度中位数
            np.mean(np.diff(magnitude, axis=1) ** 2, axis=1, keepdims=True)  # 幅度变化率
        ]
        features.extend(mag_features)

        # 2. 关键相位特征 (6维)
        phase_complex = np.exp(1j * phase)
        phase_features = [
            np.std(phase, axis=1, keepdims=True),  # 相位变异性
            np.abs(np.mean(phase_complex, axis=1, keepdims=True)),  # 相位同步性
            np.mean(np.abs(np.diff(phase, axis=1)), axis=1, keepdims=True),  # 相位变化率
            np.sum(np.abs(np.diff(phase, axis=1)) > np.pi / 2, axis=1, keepdims=True) / phase.shape[1],  # 相位跳变比例
            np.mean(np.cos(phase), axis=1, keepdims=True),  # 相位余弦均值
            np.mean(np.sin(phase), axis=1, keepdims=True)  # 相位正弦均值
        ]
        features.extend(phase_features)

        # 3. 实部虚部特征 (4维)
        complex_features = [
            np.mean(real_part, axis=1, keepdims=True),  # 实部均值
            np.std(real_part, axis=1, keepdims=True),  # 实部变异性
            np.mean(imag_part, axis=1, keepdims=True),  # 虚部均值
            np.std(imag_part, axis=1, keepdims=True),  # 虚部变异性
        ]
        features.extend(complex_features)

        # 4. 时频特性特征 (4维)
        time_freq_features = [
            np.argmax(np.abs(np.fft.fft(real_part, axis=1)), axis=1, keepdims=True) / real_part.shape[1],  # 主导频率位置
            np.sum(np.abs(np.fft.fft(real_part, axis=1))[:, :real_part.shape[1] // 4], axis=1,
                   keepdims=True) /  # 低频能量比例
            np.sum(np.abs(np.fft.fft(real_part, axis=1)), axis=1, keepdims=True),
            np.mean(np.diff(real_part, axis=1) ** 2, axis=1, keepdims=True),  # 实部时间变异性
            np.mean(np.diff(imag_part, axis=1) ** 2, axis=1, keepdims=True)  # 虚部时间变异性
        ]
        features.extend(time_freq_features)

        # 5. 尺度间关系特征 (6维) - 为频带划分准备
        if magnitude.shape[0] > 1:
            # 相邻尺度相关性
            scale_correlations = []
            for i in range(min(3, magnitude.shape[0] - 1)):
                corr_vals = []
                for j in range(magnitude.shape[0] - 1):
                    if np.std(magnitude[j]) > 1e-8 and np.std(magnitude[j + 1]) > 1e-8:
                        corr = np.corrcoef(magnitude[j], magnitude[j + 1])[0, 1]
                        corr_vals.append(corr)
                scale_correlations.append(np.mean(corr_vals) if corr_vals else 0)

            # 填充不足的维度
            while len(scale_correlations) < 3:
                scale_correlations.append(0)

            scale_features = [
                np.full((complex_coeff.shape[0], 1), scale_correlations[0]),  # 最近邻尺度相关性
                np.full((complex_coeff.shape[0], 1), scale_correlations[1]),  # 次近邻尺度相关性
                np.full((complex_coeff.shape[0], 1), scale_correlations[2]),  # 第三近邻尺度相关性
                np.full((complex_coeff.shape[0], 1), np.mean(scale_correlations)),  # 平均尺度相关性
                np.full((complex_coeff.shape[0], 1), np.std(scale_correlations) if len(scale_correlations) > 1 else 0),
                # 相关性变异性
                np.full((complex_coeff.shape[0], 1),
                        len([c for c in scale_correlations if c > 0.5]) / len(scale_correlations))  # 强相关性比例
            ]
        else:
            scale_features = [np.zeros((complex_coeff.shape[0], 1)) for _ in range(6)]

        features.extend(scale_features)

        # 合并所有特征
        return np.concatenate(features, axis=1)

    def compute_wavelet_features(self, unified_data):
        """计算小波特征"""
        print("提取小波特征...")
        feature_dict = {}

        for sub_id, data in tqdm(unified_data.items(), desc="计算特征"):
            ts = data['ts']
            label = data['label']
            tr = data['tr']

            try:
                # 计算小波尺度
                min_freq, max_freq = self.FREQ_RANGE
                center_freq = 1.0
                samp_freq = 1.0 / tr
                min_scale = center_freq / (max_freq * samp_freq)
                max_scale = center_freq / (min_freq * samp_freq)
                scales = np.logspace(np.log10(min_scale), np.log10(max_scale), self.NUM_SCALES)

                # 小波变换
                cwt_all = np.zeros((self.TARGET_ROIS, self.NUM_SCALES, self.TARGET_LEN), dtype=np.complex64)
                valid_rois = 0

                for roi in range(self.TARGET_ROIS):
                    try:
                        coeff, _ = pywt.cwt(ts[roi], scales, self.WAVELET)
                        cwt_all[roi] = coeff
                        valid_rois += 1
                    except Exception as e:
                        # 小波变换失败，使用零填充
                        cwt_all[roi] = np.zeros((self.NUM_SCALES, self.TARGET_LEN), dtype=np.complex64)
                        continue

                # 如果有效脑区太少，跳过该受试者
                if valid_rois < self.TARGET_ROIS * 0.8:  # 至少80%的脑区有效
                    print(f"警告: 受试者 {sub_id} 有效脑区过少 ({valid_rois}/{self.TARGET_ROIS})，跳过")
                    continue

                # 提取特征 (28维)
                features = np.zeros((self.TARGET_ROIS, self.NUM_SCALES, 28), dtype=np.float32)
                for roi in range(self.TARGET_ROIS):
                    features[roi] = self._extract_meaningful_features(cwt_all[roi])

                feature_dict[sub_id] = {
                    'features': features,
                    'label': label,
                    'tr': tr
                }
            except Exception as e:
                print(f"受试者 {sub_id} 特征提取失败: {e}")
                continue

        print(f"特征提取完成：{len(feature_dict)} 个受试者的特征")
        return feature_dict

    def save_features(self, feature_dict):
        """保存特征"""
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        print(f"保存特征到 {self.cfg.OUTPUT_DIR}...")

        # HDF5文件
        h5_path = os.path.join(self.cfg.OUTPUT_DIR, "useful_wavelet_features.h5")
        with h5py.File(h5_path, 'w') as f:
            for sub_id, data in feature_dict.items():
                grp = f.create_group(sub_id)
                grp.create_dataset('features', data=data['features'], compression='gzip')
                grp.attrs['label'] = data['label']
                grp.attrs['tr'] = data['tr']

        # CSV标签文件
        label_df = pd.DataFrame([
            {
                'subject_id': sub_id,
                'label': data['label'],
                'tr': data['tr']
            }
            for sub_id, data in feature_dict.items()
        ])
        label_df.to_csv(os.path.join(self.cfg.OUTPUT_DIR, "labels.csv"), index=False)

        # 元数据
        tr_values = sorted(set([d['tr'] for d in feature_dict.values()]))
        meta = {
            'feature_shape': (self.TARGET_ROIS, self.NUM_SCALES, 28),
            'num_subjects': len(feature_dict),
            'asd_count': sum(1 for d in feature_dict.values() if d['label'] == 1),
            'control_count': sum(1 for d in feature_dict.values() if d['label'] == 0),
            'tr_values': tr_values,
        }
        with open(os.path.join(self.cfg.OUTPUT_DIR, "meta.json"), 'w') as f:
            json.dump(meta, f, indent=2)

        print("保存完成！")
        print(f"特征格式: {meta['feature_shape']}")
        print(f"TR值分布: {tr_values}")

    def run(self):
        """一键运行"""
        labels = self.load_labels()
        if not labels:
            print("无有效标签，退出！")
            return

        useful_data = self.load_useful_1d_data(labels)
        if not useful_data:
            print("无有用数据，退出！")
            return

        unified_data = self.unify_ts_length(useful_data)
        feature_dict = self.compute_wavelet_features(unified_data)
        if not feature_dict:
            print("无有效特征，退出！")
            return

        self.save_features(feature_dict)

        print("\n=== 最终结果 ===")
        print(f"有效受试者数: {len(feature_dict)}")
        print(f"ASD: {sum(1 for d in feature_dict.values() if d['label'] == 1)}")
        print(f"健康对照: {sum(1 for d in feature_dict.values() if d['label'] == 0)}")
        print(f"特征格式: ({self.TARGET_ROIS}脑区, {self.NUM_SCALES}尺度, 28特征)")
        print(f"TR分布: {sorted(set([d['tr'] for d in feature_dict.values()]))} s")


if __name__ == "__main__":
    cfg = Config()
    if not os.path.exists(cfg.FMRI_DATA_DIR):
        print(f"错误：fMRI目录 {cfg.FMRI_DATA_DIR} 不存在，请检查路径！")
    else:
        extractor = ABIDEWaveletFeatureExtractor(cfg)
        extractor.run()