#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    base_dir = Path(".")
    ce_file = Path("cui_pnas_2023_ce_labels.csv")
    data_txt = Path("data1.txt")
    
    # 要去除的列名
    columns_to_remove = [
        '0V_Li_peak_enrichment',
        '0V_max_enrichment_solvent', 
        '0V_max_enrichment_solvent_peak',
        '0V_solvent_peak_sum',
        '4V_max_enrichment_solvent',
        '4V_bulk',
        'capacitance_uF_per_cm2',
        'total_potential_drop_V',
    ]
    
    # 1. 读取库伦效率数据（统一转为LCE）
    print("读取库伦效率数据...")
    
    # 1.1 从 cui_pnas_2023_ce_labels.csv 读取（已经是LCE）
    ce_df_primary = None
    if ce_file.exists():
        ce_df_primary = pd.read_csv(ce_file)
        ce_df_primary = ce_df_primary[['formulation_id', 'lce', 'cycle']].copy()
        ce_df_primary = ce_df_primary.rename(columns={
            'formulation_id': 'recipe_name', 
            'lce': 'LCE'
        })
        ce_df_primary = ce_df_primary.drop_duplicates(subset=['recipe_name'], keep='first')
        print(f"  从 {ce_file} 读取 {len(ce_df_primary)} 个配方（LCE已直接可用）")
        print(f"  包含cycle信息的配方: {ce_df_primary['cycle'].notna().sum()} 个")
    else:
        print(f"  警告: 找不到 {ce_file}")
    
    # 1.2 从 data1.txt 读取并计算LCE（备用数据源，现在包含3列：配方名, CE, cycle）
    ce_df_secondary = None
    if data_txt.exists():
        # 读取时将所有列都当作字符串处理
        # data1.txt 格式: recipe_name, CE, cycle
        data_df = pd.read_csv(data_txt, header=None, 
                              names=['recipe_name', 'CE', 'cycle'], 
                              dtype={'recipe_name': str, 'CE': str, 'cycle': str})
        
        # 去除可能的空格和特殊字符
        data_df['recipe_name'] = data_df['recipe_name'].astype(str).str.strip()
        data_df['CE'] = data_df['CE'].astype(str).str.strip()
        data_df['cycle'] = data_df['cycle'].astype(str).str.strip()
        
        # 过滤掉空行或无效行
        data_df = data_df[data_df['recipe_name'] != '']
        data_df = data_df[data_df['CE'] != '']
        data_df = data_df[~data_df['recipe_name'].str.contains('nan|NaN|None', na=False)]
        data_df = data_df[~data_df['CE'].str.contains('nan|NaN|None', na=False)]
        
        # 转换CE和cycle为数值
        data_df['CE'] = pd.to_numeric(data_df['CE'], errors='coerce')
        data_df['cycle'] = pd.to_numeric(data_df['cycle'], errors='coerce')
        data_df = data_df.dropna(subset=['CE'])
        
        if len(data_df) > 0:
            # 判断是百分比还是小数
            if data_df['CE'].max() > 1:
                data_df['CE_decimal'] = data_df['CE'] / 100
                print(f"  检测到CE为百分比形式，已转换为小数")
            else:
                data_df['CE_decimal'] = data_df['CE']
            
            # 计算 LCE = -log10(1 - CE)
            data_df['CE_decimal'] = data_df['CE_decimal'].clip(upper=0.999999)
            data_df['LCE'] = -np.log10(1 - data_df['CE_decimal'])
            
            # 保留配方名、LCE和cycle
            ce_df_secondary = data_df[['recipe_name', 'LCE', 'cycle']].copy()
            ce_df_secondary = ce_df_secondary.drop_duplicates(subset=['recipe_name'], keep='first')
            print(f"  从 {data_txt} 读取 {len(ce_df_secondary)} 个配方（已计算LCE）")
            print(f"  包含cycle信息的配方: {ce_df_secondary['cycle'].notna().sum()} 个")
            print(f"  示例配方: {ce_df_secondary[['recipe_name', 'cycle']].head(3).to_string()}")
        else:
            print(f"  警告: {data_txt} 中CE列无有效数据")
    else:
        print(f"  注意: 找不到 {data_txt}")
    
    # 1.3 合并两个数据源，优先使用 primary 的数据
    if ce_df_primary is not None and ce_df_secondary is not None:
        # 先合并，primary优先
        ce_df_combined = pd.merge(ce_df_secondary, ce_df_primary, 
                                  on='recipe_name', how='outer', 
                                  suffixes=('_secondary', '_primary'))
        
        # 优先使用 primary 的 LCE
        ce_df_combined['LCE'] = ce_df_combined['LCE_primary'].fillna(ce_df_combined['LCE_secondary'])
        
        # 处理 cycle 列：优先使用 primary 的 cycle，如果没有则使用 secondary 的
        if 'cycle_primary' in ce_df_combined.columns and 'cycle_secondary' in ce_df_combined.columns:
            ce_df_combined['cycle'] = ce_df_combined['cycle_primary'].fillna(ce_df_combined['cycle_secondary'])
        elif 'cycle_primary' in ce_df_combined.columns:
            ce_df_combined['cycle'] = ce_df_combined['cycle_primary']
        elif 'cycle_secondary' in ce_df_combined.columns:
            ce_df_combined['cycle'] = ce_df_combined['cycle_secondary']
        
        # 只保留需要的列
        ce_df = ce_df_combined[['recipe_name', 'LCE']].copy()
        if 'cycle' in ce_df_combined.columns:
            ce_df['cycle'] = ce_df_combined['cycle']
        
        # 删除临时列
        ce_df = ce_df.dropna(subset=['LCE'])
        
        print(f"\n合并后共有 {len(ce_df)} 个配方的LCE数据")
        print(f"  其中来自 primary 数据源: {ce_df_combined['LCE_primary'].notna().sum()}")
        print(f"  其中来自 secondary 数据源: {ce_df_combined['LCE_secondary'].notna().sum() - ce_df_combined['LCE_primary'].notna().sum()}")
        print(f"  有cycle数据的配方: {ce_df['cycle'].notna().sum()}")
        
    elif ce_df_primary is not None:
        ce_df = ce_df_primary.copy()
        print(f"\n仅使用 primary 数据源: {len(ce_df)} 个配方")
        print(f"  有cycle数据的配方: {ce_df['cycle'].notna().sum()}")
    elif ce_df_secondary is not None:
        ce_df = ce_df_secondary.copy()
        print(f"\n仅使用 secondary 数据源: {len(ce_df)} 个配方")
        print(f"  有cycle数据的配方: {ce_df['cycle'].notna().sum()}")
    else:
        print("错误: 未找到任何CE数据源")
        return
    
    # 2. 读取各个分析结果
    print("\n读取分析结果...")
    dfs = []
    
    # 辅助函数：删除指定列
    def remove_columns(df, cols_to_remove):
        for col in cols_to_remove:
            if col in df.columns:
                df = df.drop(columns=[col])
                print(f"    已删除列: {col}")
        return df
    
    # anion_adsorption
    file_path = base_dir / "anion_adsorption" / "anion_adsorption_summary.csv"
    if file_path.exists():
        df = pd.read_csv(file_path)
        if 'recipe_name' in df.columns:
            df = df.drop_duplicates(subset=['recipe_name'], keep='first')
        df = remove_columns(df, columns_to_remove)
        dfs.append(df)
        print(f"  anion_adsorption: {len(df)} 个配方")
    
    # anion_ratio
    file_path = base_dir / "anion_ratio" / "anion_ratio_summary.csv"
    if file_path.exists():
        df = pd.read_csv(file_path)
        if 'recipe_name' in df.columns:
            df = df.drop_duplicates(subset=['recipe_name'], keep='first')
        df = remove_columns(df, columns_to_remove)
        dfs.append(df)
        print(f"  anion_ratio: {len(df)} 个配方")
    
    # capacitance_potential_summary
    file_path = base_dir / "capacitance_potential_summary" / "capacitance_potential_summary.csv"
    if file_path.exists():
        df = pd.read_csv(file_path)
        if 'recipe_name' in df.columns:
            df = df.drop_duplicates(subset=['recipe_name'], keep='first')
        df = remove_columns(df, columns_to_remove)
        dfs.append(df)
        print(f"  capacitance_potential_summary: {len(df)} 个配方")
    
    # distribution
    file_path = base_dir / "distribution" / "cathode_interface_enrichment.csv"
    if file_path.exists():
        df = pd.read_csv(file_path)
        if 'recipe_name' in df.columns:
            df = df.drop_duplicates(subset=['recipe_name'], keep='first')
        df = remove_columns(df, columns_to_remove)
        dfs.append(df)
        print(f"  distribution: {len(df)} 个配方")
    
    # 3. 合并所有数据（特征）
    if dfs:
        print("\n合并特征数据...")
        merged_features = dfs[0]
        for df in dfs[1:]:
            merged_features = pd.merge(merged_features, df, on='recipe_name', how='outer')
        
        # 去重
        if merged_features['recipe_name'].duplicated().any():
            merged_features = merged_features.drop_duplicates(subset=['recipe_name'], keep='first')
        
        print(f"  特征数据总配方数: {len(merged_features)}")
        
        # 检查是否还有要删除的列（确保彻底删除）
        print("\n检查是否还有需要删除的列...")
        remaining_cols_to_remove = [col for col in columns_to_remove if col in merged_features.columns]
        if remaining_cols_to_remove:
            print(f"  警告: 以下列仍存在，将再次删除: {remaining_cols_to_remove}")
            merged_features = merged_features.drop(columns=remaining_cols_to_remove)
        else:
            print(f"  ✓ 所有指定列已成功删除")
        
        # ============================================
        # 生成最终数据集：只包含有 LCE 的配方
        # ============================================
        print("\n" + "="*60)
        print("生成最终数据集（只包含有 LCE 数据的配方）")
        print("="*60)
        
        # 只保留在 ce_df 中出现的配方（inner join）
        merged_final = pd.merge(merged_features, ce_df, on='recipe_name', how='inner')
        
        # 调整列顺序
        cols = ['recipe_name', 'LCE']
        if 'cycle' in merged_final.columns:
            cols.append('cycle')
        cols += [c for c in merged_final.columns if c not in cols]
        merged_final = merged_final[cols]
        
        # 保存最终版
        output_path = base_dir / "merged_analysis_summary.csv"
        merged_final.to_csv(output_path, index=False)
        
        print(f"  输出文件: {output_path}")
        print(f"  总配方数: {len(merged_final)}")
        print(f"  总特征数: {len(merged_final.columns) - 1}")
        print(f"  所有配方都有LCE数据")
        
        if merged_final['LCE'].notna().any():
            print(f"  LCE范围: {merged_final['LCE'].min():.4f} - {merged_final['LCE'].max():.4f}")
        
        if 'cycle' in merged_final.columns:
            print(f"  有cycle数据的配方: {merged_final['cycle'].notna().sum()}")
        
        # 显示前几行
        print("\n前10行预览:")
        display_cols = ['recipe_name', 'LCE']
        if 'cycle' in merged_final.columns:
            display_cols.append('cycle')
        print(merged_final[display_cols].head(10).to_string())
        
        # 检查是否有缺失值
        print("\n缺失值统计:")
        missing_counts = merged_final.isna().sum()
        missing_cols = missing_counts[missing_counts > 0]
        if len(missing_cols) > 0:
            print(f"  有 {len(missing_cols)} 列存在缺失值:")
            for col, count in missing_cols.head(10).items():
                print(f"    {col}: {count} ({count/len(merged_final)*100:.1f}%)")
        else:
            print("  ✓ 无缺失值")
        
        print(f"\n✅ 合并完成！文件已保存: {output_path}")
    else:
        print("未找到任何CSV文件")

if __name__ == "__main__":
    main()
