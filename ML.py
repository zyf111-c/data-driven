# ============================================
# ENHANCED HYPERPARAMETER TUNING WITH VISUALIZATIONS (包括 cycle 特征)
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100

print("="*60)
print("COMPREHENSIVE HYPERPARAMETER TUNING WITH VISUALIZATIONS (包括 cycle 特征)")
print("="*60)

# Load data
df = pd.read_csv("merged_analysis_summary_with_data1.csv")

# Data preparation - 使用 LCE 列
target_col = 'LCE' if 'LCE' in df.columns else 'CE_percent'

print(f"Target variable: {target_col}")

# Clean data
df_clean = df.dropna(subset=[target_col])

# ============================================
# 修改点: 将 cycle 也作为特征
# ============================================
# 定义特征列：排除 recipe_name 和 target_col，但保留 cycle
feature_cols = [col for col in df_clean.columns if col not in ['recipe_name', target_col]]

print(f"Features: {len(feature_cols)}")
print(f"Sample count: {len(df_clean)}")
print(f"Sample/Feature ratio: {len(df_clean)/len(feature_cols):.1f}")

# 检查 cycle 是否在特征中
if 'cycle' in feature_cols:
    print(f"✓ cycle 已包含在特征中")
    print(f"  cycle 范围: {df_clean['cycle'].min():.0f} - {df_clean['cycle'].max():.0f}")
    print(f"  cycle 缺失数量: {df_clean['cycle'].isna().sum()}")
else:
    print(f"⚠️ cycle 不在特征中")

# Handle NaN - 对 cycle 也用中位数填充
df_filled = df_clean.copy()
for col in feature_cols:
    if df_filled[col].isna().any():
        median_val = df_filled[col].median()
        df_filled[col] = df_filled[col].fillna(median_val)
        print(f"  填充 {col} 缺失值: {median_val:.4f}")

X = df_filled[feature_cols].copy()
y = df_filled[target_col].copy()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Store all results and predictions
all_results = []
all_predictions = {}
all_models = {}

# ============================================
# 1. RIDGE REGRESSION
# ============================================
print("\n" + "="*60)
print("1. RIDGE REGRESSION (L2)")
print("="*60)

ridge = Ridge(random_state=42)
ridge_params = {
    'alpha': [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1, 3, 5, 10, 30, 50, 100],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr']
}
ridge_grid = GridSearchCV(ridge, ridge_params, cv=cv, scoring='r2', n_jobs=-1, verbose=0)
ridge_grid.fit(X_scaled, y)

y_pred = cross_val_predict(ridge_grid.best_estimator_, X_scaled, y, cv=cv)
r2_final = r2_score(y, y_pred)
mae_final = mean_absolute_error(y, y_pred)
rmse_final = np.sqrt(mean_squared_error(y, y_pred))

all_predictions['Ridge'] = y_pred
all_models['Ridge'] = ridge_grid.best_estimator_

print(f"Best alpha: {ridge_grid.best_params_['alpha']}")
print(f"Final R²: {r2_final:.4f}")

all_results.append({
    'Model': 'Ridge',
    'CV R²': ridge_grid.best_score_,
    'Final R²': r2_final,
    'MAE': mae_final,
    'RMSE': rmse_final
})

# ============================================
# 2. RANDOM FOREST
# ============================================
print("\n" + "="*60)
print("2. RANDOM FOREST")
print("="*60)

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_params = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.8]
}
rf_random = RandomizedSearchCV(rf, rf_params, n_iter=50, cv=cv, scoring='r2', 
                                random_state=42, n_jobs=-1, verbose=0)
rf_random.fit(X_scaled, y)

y_pred = cross_val_predict(rf_random.best_estimator_, X_scaled, y, cv=cv)
r2_final = r2_score(y, y_pred)
mae_final = mean_absolute_error(y, y_pred)
rmse_final = np.sqrt(mean_squared_error(y, y_pred))

all_predictions['RandomForest'] = y_pred
all_models['RandomForest'] = rf_random.best_estimator_

print(f"Best params: {rf_random.best_params_}")
print(f"Final R²: {r2_final:.4f}")

all_results.append({
    'Model': 'RandomForest',
    'CV R²': rf_random.best_score_,
    'Final R²': r2_final,
    'MAE': mae_final,
    'RMSE': rmse_final
})

# Feature importance
rf_best = rf_random.best_estimator_
rf_best.fit(X_scaled, y)
rf_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_best.feature_importances_
}).sort_values('importance', ascending=False)

# 特别显示 cycle 的重要性
cycle_importance = rf_importance[rf_importance['feature'] == 'cycle']['importance'].values
if len(cycle_importance) > 0:
    print(f"  cycle 特征重要性: {cycle_importance[0]:.4f}")

# ============================================
# 3. GRADIENT BOOSTING
# ============================================
print("\n" + "="*60)
print("3. GRADIENT BOOSTING")
print("="*60)

gbr = GradientBoostingRegressor(random_state=42)
gbr_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.3],
    'max_depth': [3, 4, 5, 6, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.7, 0.8, 0.9, 1.0]
}
gbr_random = RandomizedSearchCV(gbr, gbr_params, n_iter=50, cv=cv, scoring='r2',
                                 random_state=42, n_jobs=-1, verbose=0)
gbr_random.fit(X_scaled, y)

y_pred = cross_val_predict(gbr_random.best_estimator_, X_scaled, y, cv=cv)
r2_final = r2_score(y, y_pred)
mae_final = mean_absolute_error(y, y_pred)
rmse_final = np.sqrt(mean_squared_error(y, y_pred))

all_predictions['GradientBoosting'] = y_pred
all_models['GradientBoosting'] = gbr_random.best_estimator_

print(f"Best params: {gbr_random.best_params_}")
print(f"Final R²: {r2_final:.4f}")

all_results.append({
    'Model': 'GradientBoosting',
    'CV R²': gbr_random.best_score_,
    'Final R²': r2_final,
    'MAE': mae_final,
    'RMSE': rmse_final
})

# Feature importance
gbr_best = gbr_random.best_estimator_
gbr_best.fit(X_scaled, y)
gbr_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': gbr_best.feature_importances_
}).sort_values('importance', ascending=False)

# 特别显示 cycle 的重要性
cycle_importance = gbr_importance[gbr_importance['feature'] == 'cycle']['importance'].values
if len(cycle_importance) > 0:
    print(f"  cycle 特征重要性: {cycle_importance[0]:.4f}")

# ============================================
# 4. ELASTIC NET
# ============================================
print("\n" + "="*60)
print("4. ELASTIC NET")
print("="*60)

elastic = ElasticNet(random_state=42, max_iter=10000)
elastic_params = {
    'alpha': [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
}
elastic_grid = GridSearchCV(elastic, elastic_params, cv=cv, scoring='r2', n_jobs=-1, verbose=0)
elastic_grid.fit(X_scaled, y)

y_pred = cross_val_predict(elastic_grid.best_estimator_, X_scaled, y, cv=cv)
r2_final = r2_score(y, y_pred)

all_predictions['ElasticNet'] = y_pred

print(f"Final R²: {r2_final:.4f}")

all_results.append({
    'Model': 'ElasticNet',
    'CV R²': elastic_grid.best_score_,
    'Final R²': r2_final,
    'MAE': mean_absolute_error(y, y_pred),
    'RMSE': np.sqrt(mean_squared_error(y, y_pred))
})

# ============================================
# 5. POLYNOMIAL + RIDGE
# ============================================
print("\n" + "="*60)
print("5. POLYNOMIAL (Degree 2) + RIDGE")
print("="*60)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)
scaler_poly = StandardScaler()
X_poly_scaled = scaler_poly.fit_transform(X_poly)

ridge_poly = Ridge(random_state=42)
ridge_poly_params = {'alpha': [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]}
ridge_poly_grid = GridSearchCV(ridge_poly, ridge_poly_params, cv=cv, scoring='r2', n_jobs=-1, verbose=0)
ridge_poly_grid.fit(X_poly_scaled, y)

y_pred = cross_val_predict(ridge_poly_grid.best_estimator_, X_poly_scaled, y, cv=cv)
r2_final = r2_score(y, y_pred)

all_predictions['Poly+Ridge'] = y_pred

print(f"Final R²: {r2_final:.4f}")

all_results.append({
    'Model': 'Poly+Ridge',
    'CV R²': ridge_poly_grid.best_score_,
    'Final R²': r2_final,
    'MAE': mean_absolute_error(y, y_pred),
    'RMSE': np.sqrt(mean_squared_error(y, y_pred))
})

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "="*60)
print("FINAL SUMMARY - ALL MODELS COMPARED")
print("="*60)

results_df = pd.DataFrame(all_results).round(4)
results_df = results_df.sort_values('Final R²', ascending=False)
print(results_df.to_string(index=False))

# Get top 3 models
top3_models = results_df.head(3)['Model'].tolist()
print(f"\n🏆 Top 3 Models: {top3_models}")

# ============================================
# 额外分析: cycle 的影响
# ============================================
print("\n" + "="*60)
print("CYCLE FEATURE ANALYSIS")
print("="*60)

# 检查 cycle 的系数（从 Ridge 模型）
ridge_model = all_models.get('Ridge')
if ridge_model is not None:
    ridge_coefs = dict(zip(feature_cols, ridge_model.coef_))
    if 'cycle' in ridge_coefs:
        print(f"Ridge - cycle 系数: {ridge_coefs['cycle']:.4f}")
        if ridge_coefs['cycle'] > 0:
            print(f"  → cycle 越大，LCE 越高（正相关）")
        else:
            print(f"  → cycle 越大，LCE 越低（负相关）")

# 检查 cycle 的特征重要性（从 Random Forest）
if 'cycle' in feature_cols:
    rf_cycle_imp = rf_importance[rf_importance['feature'] == 'cycle']['importance'].values
    gbr_cycle_imp = gbr_importance[gbr_importance['feature'] == 'cycle']['importance'].values
    
    if len(rf_cycle_imp) > 0:
        print(f"Random Forest - cycle 重要性: {rf_cycle_imp[0]:.4f}")
    if len(gbr_cycle_imp) > 0:
        print(f"Gradient Boosting - cycle 重要性: {gbr_cycle_imp[0]:.4f}")

# ============================================
# VISUALIZATION: Actual vs Predicted for Top 3 Models
# ============================================
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, model_name in enumerate(top3_models):
    ax = axes[idx]
    y_pred = all_predictions[model_name]
    r2 = results_df[results_df['Model'] == model_name]['Final R²'].values[0]
    
    ax.scatter(y, y_pred, alpha=0.6, edgecolors='k', s=60, c='steelblue')
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual LCE')
    ax.set_ylabel('Predicted LCE')
    ax.set_title(f'{model_name}\nR² = {r2:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.95, f'n={len(y)}', transform=ax.transAxes, fontsize=9, verticalalignment='top')

plt.tight_layout()
plt.savefig('top3_models_actual_vs_predicted_with_cycle.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# VISUALIZATION: Residuals Distribution for Top 3 Models
# ============================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, model_name in enumerate(top3_models):
    ax = axes[idx]
    y_pred = all_predictions[model_name]
    residuals = y - y_pred
    
    ax.hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(x=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Residual')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{model_name}\nMean Residual = {residuals.mean():.4f}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('top3_models_residuals_with_cycle.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# VISUALIZATION: Feature Importance with cycle highlighted
# ============================================
tree_models_in_top3 = [m for m in top3_models if m in ['RandomForest', 'GradientBoosting']]

if tree_models_in_top3:
    fig, axes = plt.subplots(1, len(tree_models_in_top3), figsize=(8*len(tree_models_in_top3), 6))
    if len(tree_models_in_top3) == 1:
        axes = [axes]
    
    for idx, model_name in enumerate(tree_models_in_top3):
        ax = axes[idx]
        
        if model_name == 'RandomForest':
            importance_df = rf_importance.head(15)
        else:
            importance_df = gbr_importance.head(15)
        
        # 为 cycle 设置不同颜色
        colors = ['coral' if feat == 'cycle' else 'steelblue' for feat in importance_df['feature'].values]
        
        ax.barh(range(len(importance_df)), importance_df['importance'].values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'].values, fontsize=9)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'{model_name}\nTop 15 Feature Importance (cycle in coral)')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('top3_models_feature_importance_with_cycle.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================
# VISUALIZATION: Performance Comparison
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: R² Comparison
ax1 = axes[0]
top5_results = results_df.head(5)
colors = ['gold' if i == 0 else 'silver' if i == 1 else 'steelblue' for i in range(len(top5_results))]
ax1.barh(range(len(top5_results)), top5_results['Final R²'].values, color=colors, alpha=0.8)
ax1.set_yticks(range(len(top5_results)))
ax1.set_yticklabels(top5_results['Model'].values)
ax1.set_xlabel('R² Score')
ax1.set_title('Top 5 Models by R² (with cycle)')
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3)

# Plot 2: MAE vs RMSE Comparison
ax2 = axes[1]
x = np.arange(len(top5_results))
width = 0.35
ax2.bar(x - width/2, top5_results['MAE'].values, width, label='MAE', alpha=0.8, color='steelblue')
ax2.bar(x + width/2, top5_results['RMSE'].values, width, label='RMSE', alpha=0.8, color='coral')
ax2.set_xticks(x)
ax2.set_xticklabels(top5_results['Model'].values, rotation=45, ha='right')
ax2.set_ylabel('Error Value')
ax2.set_title('MAE vs RMSE Comparison (with cycle)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('top_models_performance_comparison_with_cycle.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# BEST MODEL DETAILED ANALYSIS
# ============================================
print("\n" + "="*60)
print("BEST MODEL DETAILED ANALYSIS")
print("="*60)

best_model = results_df.iloc[0]['Model']
best_r2 = results_df.iloc[0]['Final R²']
best_mae = results_df.iloc[0]['MAE']
best_rmse = results_df.iloc[0]['RMSE']

print(f"\n🏆 Best Model: {best_model}")
print(f"   R² = {best_r2:.4f}")
print(f"   MAE = {best_mae:.4f}")
print(f"   RMSE = {best_rmse:.4f}")

# Feature importance for best model
if best_model in ['RandomForest', 'GradientBoosting']:
    print(f"\n📈 Top 10 Feature Importance for {best_model}:")
    if best_model == 'RandomForest':
        imp_df = rf_importance.head(10)
    else:
        imp_df = gbr_importance.head(10)
    
    for i, row in imp_df.iterrows():
        marker = " [CYCLE]" if row['feature'] == 'cycle' else ""
        print(f"   {i+1:2d}. {row['feature']:40s} {row['importance']:.4f}{marker}")

print("\n" + "="*60)
print("✅ Analysis complete! Visualizations saved:")
print("   1. top3_models_actual_vs_predicted_with_cycle.png")
print("   2. top3_models_residuals_with_cycle.png")
if tree_models_in_top3:
    print("   3. top3_models_feature_importance_with_cycle.png")
print("   4. top_models_performance_comparison_with_cycle.png")
print("="*60)
