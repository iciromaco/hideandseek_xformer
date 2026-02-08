from sklearn.inspection import partial_dependence

# 元のGP読み込み
with open('gp_model.pkl', 'rb') as f:
    gp = pickle.load(f)

y_mean = np.mean(y)  # ≈128.0
y_std = np.std(y)    # ≈13.5

# ENT_COEFのPDP計算と逆標準化
results, grids = partial_dependence(gp, X, features=[0], grid_resolution=50)
pd_norm = results[0].squeeze()  # 標準化PDP
pd_actual = pd_norm * y_std + y_mean  # 実際のスケール

# プロット（手動）
plt.plot(grids[0], pd_actual)
plt.xlabel('ENT_COEF')
plt.ylabel('Actual Partial Dependence (value)')
plt.show()