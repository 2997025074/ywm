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
    # 路径设置（关键修改1：同步CC200数据路径，确保fmri路径正确）
    dir_1d = "C:/Users/29970/Desktop/data/cc200/Outputs/cpac/filt_noglobal/rois_cc200"  # CC200的1D时序文件路径
    dir_fmri = "C:/Users/29970/Desktop/data/aal/Outputs/cpac/filt_noglobal/func_preproc"  # 关键：改为CC200的fMRI路径（原是aal路径，需更新！）
    pheno_path = "C:/Users/29970/Desktop/abide-master/Phenotypic_V1_0b_preprocessed1.csv"  # 表型文件路径不变
    output_npz = "./abide_cc200_combined.npz"  # 关键修改2：输出文件名标注CC200，避免与AAL混淆

    # 加载表型文件，保留1/2标签（逻辑不变）
    pheno_df = pd.read_csv(pheno_path)
    pheno_df = pheno_df[['SUB_ID', 'DX_GROUP']].dropna()
    pheno_df['SUB_ID'] = pheno_df['SUB_ID'].astype(str).str.zfill(7)  # 统一SUB_ID为7位字符串
    id_to_label = dict(zip(pheno_df['SUB_ID'], pheno_df['DX_GROUP']))

    # 第一步：收集所有有效时间序列，记录长度
    all_time_series = []
    all_tr = []
    all_labels = []  # 初始为1（ASD）和2（对照组）
    time_lengths = []  # 记录每个序列的时间点数量（T）
    missing_ids = []

    print("第一步：收集CC200脑区划分的时间序列数据...")
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
        label = id_to_label[subj_id]  # 此时label为1（ASD）或2（对照组）

        # 读取fMRI文件获取TR（需确保dir_fmri是CC200的fMRI路径，否则会找不到文件）
        fmri_files = [f for f in os.listdir(dir_fmri) if subj_id in f and f.endswith('.nii.gz')]
        if not fmri_files:
            print(f"未找到被试 {subj_id} 的CC200 fMRI文件，跳过")
            continue
        fmri_path = os.path.join(dir_fmri, fmri_files[0])
        tr = get_tr_from_fmri(fmri_path)

        # 读取CC200时间序列（关键修改3：CC200的时序形状为 [T, 200]，而非[AAL的116]）
        ts_path = os.path.join(dir_1d, filename)
        try:
            time_series = np.loadtxt(ts_path)  # 形状：[时间点数量T, 脑区数量200]
        except Exception as e:
            print(f"读取时间序列失败：{filename}，错误：{str(e)}")
            continue

        # 关键修改4：形状检查改为200（CC200脑区数量），排除无效数据
        if time_series.ndim != 2 or time_series.shape[1] != 200:
            print(f"时间序列形状错误（需为[T,200]）：{filename}，实际形状：{time_series.shape}")
            continue

        # 记录有效数据
        all_time_series.append(time_series)
        all_tr.append(tr)
        all_labels.append(label)
        time_lengths.append(time_series.shape[0])  # 记录当前序列的时间点数量T

    # 确定统一的目标时间长度（截断到最短序列，避免填充引入噪声）
    if not time_lengths:
        raise ValueError("未收集到有效CC200时间序列数据，请检查路径和文件格式！")
    target_length = min(time_lengths)
    print(f"\n将所有CC200时间序列统一为 {target_length} 个时间点（截断保留前{target_length}个时间点）")

    # 第二步：统一时间序列长度（仅截断，不填充）
    aligned_time_series = []
    for ts in all_time_series:
        aligned_ts = ts[:target_length, :]  # 截断到目标长度，形状变为 [target_length, 200]
        aligned_time_series.append(aligned_ts)

    # 转换为numpy数组，并将标签从1/2转为0/1（逻辑不变，适配CrossEntropyLoss）
    all_time_series_np = np.array(aligned_time_series)  # 最终形状：[被试数N, target_length, 200]
    all_tr_np = np.array(all_tr)
    all_labels_np = np.array(all_labels)  # 原始标签：1（ASD）、2（对照组）

    # 标签转换：1→0（ASD），2→1（对照组）
    all_labels_np = np.where(all_labels_np == 1, 0, 1)

    # 统计信息（关键修改5：标注CC200脑区，避免混淆）
    total = len(all_labels_np)
    asd_count = np.sum(all_labels_np == 0)  # ASD对应标签0
    control_count = np.sum(all_labels_np == 1)  # 对照组对应标签1
    print(f"\nCC200数据处理完成：")
    print(f"有效被试数：{total}")
    print(f"统一后时间序列形状：{all_time_series_np.shape}（N={total}, T={target_length}, 脑区数=200）")
    print(f"标签分布：ASD（标签0）{asd_count}例，对照组（标签1）{control_count}例")
    print(f"未匹配到标签的被试ID数量：{len(missing_ids)}个")

    # 保存CC200数据（文件名已标注CC200，避免与AAL数据冲突）
    np.savez(output_npz, data=all_time_series_np, tr=all_tr_np, labels=all_labels_np)
    print(f"\nCC200数据已保存至：{os.path.abspath(output_npz)}")


if __name__ == "__main__":
    main()
