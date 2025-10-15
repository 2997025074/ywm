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

        # 路径配置（请确认与你的实际路径一致！）
        self.PHENOTYPIC_FILE = "C:/Users/29970/Desktop/abide-master/Phenotypic_V1_0b_preprocessed1.csv"
        self.TIME_SERIES_DIR = "C:/Users/29970/Desktop/data/aal/Outputs/cpac/filt_noglobal/rois_aal"
        self.FMRI_DATA_DIR = "C:/Users/29970/Desktop/data/aal/Outputs/cpac/filt_noglobal/func_preproc"  # 新增：fMRI文件目录
        self.OUTPUT_DIR = "C:/Users/29970/Desktop/Brainwavegnn/data"


class ABIDEWaveletFeatureExtractor:
    def __init__(self, config: Config):
        self.cfg = config
        # 加载核心参数
        self.TARGET_ROIS = config.target_rois
        self.TARGET_LEN = config.target_length
        self.NUM_SCALES = config.num_scales
        self.FREQ_RANGE = config.freq_range
        self.WAVELET = config.wavelet
        self.FMRI_DIR = config.FMRI_DATA_DIR  # 存储fMRI目录路径

    # -------------------------- 1. 加载表型数据（ID+标签） --------------------------
    def load_labels(self):
        """加载受试者ID和诊断标签"""
        print("加载受试者标签...")
        df = pd.read_csv(self.cfg.PHENOTYPIC_FILE)
        labels = {}
        for _, row in df.iterrows():
            try:
                # 统一ID格式（7位，补前导零）
                sub_id = f"{int(row['SUB_ID']):07d}"
                # 筛选有效标签（ASD=1，对照=0）
                dx_group = row['DX_GROUP']
                label = 1 if dx_group == 1 else 0 if dx_group == 2 else None
                if label is None:
                    continue
                labels[sub_id] = label
            except Exception as e:
                continue
        print(f"有效标签数: {len(labels)}")
        return labels

    # -------------------------- 新增1：查找匹配的fMRI文件 --------------------------
    def find_matching_fmri_file(self, subject_id):
        """
        根据受试者ID查找对应的fMRI文件（.nii.gz）
        支持多种文件名格式匹配（如ID在文件名中间/末尾，去前导零匹配）
        """
        # 定义匹配模式（覆盖ABIDE常见文件名格式）
        patterns = [
            f"*{subject_id}*func_preproc.nii.gz",  # 如：sub-0051456_func_preproc.nii.gz
            f"*_{subject_id}_*func_preproc.nii.gz",  # 如：NYU_0051456_func_preproc.nii.gz
            f"*{subject_id.lstrip('0')}*func_preproc.nii.gz"  # 去前导零匹配（如51456→0051456）
        ]

        # 遍历所有模式查找文件
        for pattern in patterns:
            search_path = os.path.join(self.FMRI_DIR, pattern)
            matching_files = glob.glob(search_path, recursive=True)
            if matching_files:
                return matching_files[0]  # 返回第一个匹配文件（通常唯一）

        # 未找到时提示
        print(f"警告：未找到受试者 {subject_id} 的fMRI文件，将使用默认TR=2.0")
        return None

    # -------------------------- 新增2：从fMRI文件提取并处理TR --------------------------
    def extract_and_process_tr(self, nifti_file_path):
        """
        从fMRI文件（.nii.gz）提取TR，处理为最多1位小数
        Args:
            nifti_file_path: fMRI文件路径
        Returns:
            tr: 处理后的TR（float，保留1位小数）
        """
        try:
            # 加载nifti文件并读取header
            img = nib.load(nifti_file_path)
            header = img.header
            # 提取TR（nifti文件中第4维度的zoom值，即时间分辨率）
            tr = header.get_zooms()[3]  # 格式：(x, y, z, tr)

            # 处理TR：保留1位小数（无论原小数位数多少）
            processed_tr = round(float(tr), 1)

            # 验证TR合理性（排除异常值，如TR>10s可能是读取错误）
            if processed_tr <= 0 or processed_tr > 10:
                print(f"警告：TR={processed_tr}s 异常，使用默认TR=2.0")
                return 2.0
            return processed_tr

        except Exception as e:
            # 提取失败时使用默认TR
            error_msg = str(e)[:50]  # 限制错误信息长度
            print(f"提取TR失败（{os.path.basename(nifti_file_path)}）: {error_msg}，使用默认TR=2.0")
            return 2.0

    # -------------------------- 2. 加载有用的1D数据（含TR读取） --------------------------
    def load_useful_1d_data(self, labels):
        """加载脑区数=116的1D文件，同时读取对应fMRI的TR"""
        print("加载有用的1D文件并匹配fMRI的TR...")
        # 扫描1D文件（.1D/.1d）
        all_1d = glob.glob(os.path.join(self.cfg.TIME_SERIES_DIR, "**/*.1D"), recursive=True)
        if not all_1d:
            all_1d = glob.glob(os.path.join(self.cfg.TIME_SERIES_DIR, "**/*.1d"), recursive=True)
        print(f"找到 {len(all_1d)} 个1D文件，开始筛选...")

        useful_data = {}
        for file in tqdm(all_1d, desc="筛选文件+读取TR"):
            try:
                # 1. 从文件名提取受试者ID（统一为7位）
                basename = os.path.basename(file)
                id_match = re.search(r'(\d{5,7})', basename)
                if not id_match:
                    continue
                sub_id = f"{int(id_match.group(1)):07d}"
                if sub_id not in labels:  # 跳过无标签的受试者
                    continue

                # 2. 读取1D文件，确保脑区数=116
                data = np.loadtxt(file, dtype=np.float32)
                if data.ndim == 1:
                    continue  # 跳过单维数据
                # 维度校正：确保为 [116, 时间点]
                if data.shape[0] == self.TARGET_ROIS:
                    ts = data
                elif data.shape[1] == self.TARGET_ROIS:
                    ts = data.T
                else:
                    continue  # 脑区数不匹配，跳过

                # 3. 过滤过短时间序列
                if ts.shape[1] < 30:
                    continue

                # 4. 读取并处理TR（新增核心逻辑）
                fmri_file = self.find_matching_fmri_file(sub_id)
                if fmri_file:
                    tr = self.extract_and_process_tr(fmri_file)
                else:
                    tr = 2.0  # 未找到fMRI文件时用默认TR

                # 5. 保存数据（含实际TR）
                useful_data[sub_id] = {
                    'ts': ts,
                    'label': labels[sub_id],
                    'tr': tr  # 存储处理后的TR
                }
            except Exception as e:
                # 无视加载错误，直接跳过
                continue

        print(f"筛选完成：{len(useful_data)} 个有用的受试者数据")
        # 统计TR分布（可选，便于验证）
        tr_values = set([d['tr'] for d in useful_data.values()])
        print(f"提取的TR值分布：{sorted(tr_values)}（单位：s）")
        return useful_data

    # -------------------------- 3. 统一时间序列长度 --------------------------
    def unify_ts_length(self, useful_data):
        """统一所有时间序列长度为TARGET_LEN"""
        print(f"统一时间序列长度为 {self.TARGET_LEN}...")
        unified = {}
        for sub_id, data in tqdm(useful_data.items(), desc="统一长度"):
            ts = data['ts']
            _, curr_len = ts.shape

            # 裁剪（长序列取中间部分）或插值（短序列线性扩展）
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

            # 保留TR信息
            unified[sub_id] = {
                'ts': unified_ts,
                'label': data['label'],
                'tr': data['tr']
            }
        return unified

    # -------------------------- 4. 小波变换+特征提取（用实际TR计算尺度） --------------------------
    def compute_wavelet_features(self, unified_data):
        """计算小波特征，使用每个受试者的实际TR计算尺度"""
        print("提取小波特征（基于实际TR计算尺度）...")
        feature_dict = {}

        for sub_id, data in tqdm(unified_data.items(), desc="计算特征"):
            ts = data['ts']  # [116, 100]
            label = data['label']
            tr = data['tr']  # 每个受试者的实际TR（已处理为1位小数）

            try:
                # 1. 基于实际TR计算小波尺度（动态尺度，非固定值）
                min_freq, max_freq = self.FREQ_RANGE
                center_freq = 1.0  # Morlet小波中心频率
                samp_freq = 1.0 / tr  # 采样频率 = 1/TR
                # 尺度计算公式：scale = 中心频率 / (频率 × 采样频率)
                min_scale = center_freq / (max_freq * samp_freq)
                max_scale = center_freq / (min_freq * samp_freq)
                scales = np.logspace(np.log10(min_scale), np.log10(max_scale), self.NUM_SCALES)

                # 2. 小波变换（每个脑区独立计算）
                cwt_all = np.zeros((self.TARGET_ROIS, self.NUM_SCALES, self.TARGET_LEN), dtype=np.complex64)
                for roi in range(self.TARGET_ROIS):
                    coeff, _ = pywt.cwt(ts[roi], scales, self.WAVELET)
                    cwt_all[roi] = coeff

                # 3. 提取72维特征（32实部+32虚部+8相位）
                features = np.zeros((self.TARGET_ROIS, self.NUM_SCALES, 72), dtype=np.float32)
                for roi in range(self.TARGET_ROIS):
                    roi_cwt = cwt_all[roi]
                    real_feat = self._get_real_feat(roi_cwt.real)  # 32维实部特征
                    imag_feat = self._get_real_feat(roi_cwt.imag)  # 32维虚部特征
                    phase_feat = self._get_phase_feat(roi_cwt)  # 8维相位特征
                    features[roi] = np.concatenate([real_feat, imag_feat, phase_feat], axis=1)

                # 4. 保存特征（含TR信息，便于后续验证）
                feature_dict[sub_id] = {
                    'features': features,  # [116, 64, 72]
                    'label': label,
                    'tr': tr  # 记录实际TR，便于后续分析
                }
            except Exception as e:
                continue  # 计算错误时跳过

        print(f"特征提取完成：{len(feature_dict)} 个受试者的特征")
        return feature_dict

    # 辅助：提取实部/虚部特征（32维）
    def _get_real_feat(self, real_coeff):
        # 基础统计量（7维）
        feat = [
            np.mean(real_coeff, axis=1, keepdims=True),
            np.std(real_coeff, axis=1, keepdims=True),
            np.max(real_coeff, axis=1, keepdims=True),
            np.min(real_coeff, axis=1, keepdims=True),
            np.sum(real_coeff ** 2, axis=1, keepdims=True),
            skew(real_coeff, axis=1, keepdims=True),
            kurtosis(real_coeff, axis=1, keepdims=True)
        ]
        feat = np.concatenate(feat, axis=1)
        # 扩展到32维（保证格式统一）
        if feat.shape[1] < 32:
            repeat_factor = (32 + feat.shape[1] - 1) // feat.shape[1]
            feat = np.tile(feat, (1, repeat_factor))[:, :32]
        return feat

    # 辅助：提取相位特征（8维）
    def _get_phase_feat(self, complex_coeff):
        phase = np.angle(complex_coeff)
        phase_complex = np.exp(1j * phase)
        return np.stack([
            np.mean(phase, axis=1),
            np.std(phase, axis=1),
            np.max(phase, axis=1),
            np.min(phase, axis=1),
            np.abs(np.mean(phase_complex, axis=1)),
            np.mean(np.cos(phase), axis=1),
            np.mean(np.sin(phase), axis=1),
            np.var(phase, axis=1)
        ], axis=1)

    # -------------------------- 5. 保存特征（含TR信息） --------------------------
    def save_features(self, feature_dict):
        """保存特征，新增TR信息到输出文件"""
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        print(f"保存特征到 {self.cfg.OUTPUT_DIR}...")

        # 1. HDF5文件（含特征、标签、TR）
        h5_path = os.path.join(self.cfg.OUTPUT_DIR, "useful_wavelet_features.h5")
        with h5py.File(h5_path, 'w') as f:
            for sub_id, data in feature_dict.items():
                grp = f.create_group(sub_id)
                grp.create_dataset('features', data=data['features'], compression='gzip')
                grp.attrs['label'] = data['label']
                grp.attrs['tr'] = data['tr']  # 新增：保存TR信息

        # 2. CSV标签文件（新增TR列）
        label_df = pd.DataFrame([
            {
                'subject_id': sub_id,
                'label': data['label'],
                'tr': data['tr']  # 新增：TR列，便于后续分析
            }
            for sub_id, data in feature_dict.items()
        ])
        label_df.to_csv(os.path.join(self.cfg.OUTPUT_DIR, "labels.csv"), index=False)

        # 3. 元数据（记录TR分布）
        tr_values = sorted(set([d['tr'] for d in feature_dict.values()]))
        meta = {
            'feature_shape': (self.TARGET_ROIS, self.NUM_SCALES, 72),
            'num_subjects': len(feature_dict),
            'asd_count': sum(1 for d in feature_dict.values() if d['label'] == 1),
            'control_count': sum(1 for d in feature_dict.values() if d['label'] == 0),
            'tr_values': tr_values,  # 新增：TR值分布
            'tr_count': len(tr_values)  # 新增：不同TR的数量
        }
        with open(os.path.join(self.cfg.OUTPUT_DIR, "meta.json"), 'w') as f:
            json.dump(meta, f, indent=2)

        print("保存完成！")
        print(f"特征格式: {meta['feature_shape']}")
        print(f"TR值分布: {tr_values}（共{len(tr_values)}种）")

    # -------------------------- 6. 核心流程（串联所有步骤） --------------------------
    def run(self):
        """一键运行：加载数据→处理TR→提取特征→保存"""
        # 步骤1：加载标签
        labels = self.load_labels()
        if not labels:
            print("无有效标签，退出！")
            return

        # 步骤2：加载1D数据并读取TR
        useful_data = self.load_useful_1d_data(labels)
        if not useful_data:
            print("无有用数据，退出！")
            return

        # 步骤3：统一时间序列长度
        unified_data = self.unify_ts_length(useful_data)

        # 步骤4：基于实际TR计算小波特征
        feature_dict = self.compute_wavelet_features(unified_data)
        if not feature_dict:
            print("无有效特征，退出！")
            return

        # 步骤5：保存结果（含TR信息）
        self.save_features(feature_dict)

        # 最终统计
        print("\n=== 最终结果 ===")
        print(f"有效受试者数: {len(feature_dict)}")
        print(f"ASD: {sum(1 for d in feature_dict.values() if d['label'] == 1)}")
        print(f"健康对照: {sum(1 for d in feature_dict.values() if d['label'] == 0)}")
        print(f"特征格式: (116脑区, 64尺度, 72特征)")
        print(f"TR分布: {sorted(set([d['tr'] for d in feature_dict.values()]))} s")


if __name__ == "__main__":
    # 初始化配置和提取器（需确认FMRI_DATA_DIR路径正确）
    cfg = Config()
    # 验证fMRI目录是否存在
    if not os.path.exists(cfg.FMRI_DATA_DIR):
        print(f"错误：fMRI目录 {cfg.FMRI_DATA_DIR} 不存在，请检查路径！")
    else:
        extractor = ABIDEWaveletFeatureExtractor(cfg)
        extractor.run()