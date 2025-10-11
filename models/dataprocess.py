import os
import re
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm


def extract_subject_id(filename):
    match = re.search(r'_(\d{7})_', filename)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"无法从文件名 {filename} 中提取被试ID")


def get_tr_from_fmri(fmri_path):
    img = nib.load(fmri_path)
    tr = img.header.get_zooms()[-1]
    return float(tr)


def main():
    # 路径设置（请替换为你的实际路径）
    dir_1d = "C:/Users/29970/Desktop/data/aal/Outputs/cpac/filt_noglobal/rois_aal"
    dir_fmri = "C:/Users/29970/Desktop/data/aal/Outputs/cpac/filt_noglobal/func_preproc"
    pheno_path = "C:/Users/29970/Desktop/abide-master/Phenotypic_V1_0b_preprocessed1.csv"
    output_npz = "./abide_aal_combined.npz"

    # 加载表型文件，保留1/2标签
    pheno_df = pd.read_csv(pheno_path)
    pheno_df = pheno_df[['SUB_ID', 'DX_GROUP']].dropna()
    pheno_df['SUB_ID'] = pheno_df['SUB_ID'].astype(str).str.zfill(7)
    id_to_label = dict(zip(pheno_df['SUB_ID'], pheno_df['DX_GROUP']))

    # 第一步：收集所有有效时间序列，记录长度
    all_time_series = []
    all_tr = []
    all_labels = []  # 初始为1（ASD）和2（对照组）
    time_lengths = []  # 记录每个序列的时间点数量
    missing_ids = []

    print("第一步：收集数据并检查时间序列长度...")
    for filename in tqdm(os.listdir(dir_1d), desc="处理1D文件"):
        if not filename.endswith('.1D'):
            continue

        try:
            subj_id = extract_subject_id(filename)
        except ValueError:
            print(f"跳过无效文件：{filename}")
            continue

        if subj_id not in id_to_label:
            missing_ids.append(subj_id)
            continue
        label = id_to_label[subj_id]  # 此时label为1或2

        # 读取fMRI文件获取TR
        fmri_files = [f for f in os.listdir(dir_fmri) if subj_id in f and f.endswith('.nii.gz')]
        if not fmri_files:
            print(f"未找到被试 {subj_id} 的fMRI文件，跳过")
            continue
        fmri_path = os.path.join(dir_fmri, fmri_files[0])
        tr = get_tr_from_fmri(fmri_path)

        # 读取时间序列
        ts_path = os.path.join(dir_1d, filename)
        try:
            time_series = np.loadtxt(ts_path)  # 形状：[T, 116]
        except:
            print(f"读取时间序列失败：{filename}")
            continue

        if time_series.ndim != 2 or time_series.shape[1] != 116:
            print(f"时间序列形状错误：{filename}，形状：{time_series.shape}")
            continue

        # 记录数据和长度（暂存原始标签1/2）
        all_time_series.append(time_series)
        all_tr.append(tr)
        all_labels.append(label)
        time_lengths.append(time_series.shape[0])

    # 确定统一的目标长度（使用最短序列长度，进行截断）
    if not time_lengths:
        raise ValueError("未收集到有效时间序列数据")
    target_length = min(time_lengths)
    print(f"\n将所有时间序列统一为 {target_length} 个时间点（采用截断方式，保留前{target_length}个时间点）")

    # 第二步：统一时间序列长度（仅截断，不填充）
    aligned_time_series = []
    for ts in all_time_series:
        aligned_ts = ts[:target_length, :]
        aligned_time_series.append(aligned_ts)

    # 转换为numpy数组，并将标签从1/2转为0/1
    all_time_series_np = np.array(aligned_time_series)  # [N, target_length, 116]
    all_tr_np = np.array(all_tr)
    all_labels_np = np.array(all_labels)  # 原始标签：1（ASD）、2（对照组）

    # 核心修改：将标签转换为0（ASD）和1（对照组）
    all_labels_np = np.where(all_labels_np == 1, 0, 1)  # 1→0，2→1

    # 统计信息（更新为转换后的标签）
    total = len(all_labels_np)
    asd_count = np.sum(all_labels_np == 0)  # ASD对应0
    control_count = np.sum(all_labels_np == 1)  # 对照组对应1
    print(f"\n处理完成：")
    print(f"有效被试数：{total}")
    print(f"统一后时间序列形状：{all_time_series_np.shape}")
    print(f"标签分布：ASD（0）{asd_count}例，对照组（1）{control_count}例")  # 打印更新
    print(f"未匹配到标签的ID：{len(missing_ids)}个")

    # 保存转换后的标签（0/1）
    np.savez(output_npz, data=all_time_series_np, tr=all_tr_np, labels=all_labels_np)
    print(f"数据已保存至：{output_npz}")


if __name__ == "__main__":
    main()
