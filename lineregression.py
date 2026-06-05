import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- 步驟 1: 生成數據 ---
print("--- 1. 正在生成數據... ---")

# 設定參數
num_points = 200  # 生成 200 個數據點
x_min, x_max = -100, 100
a_min, a_max = -50, 50
b_min, b_max = 0, 100
var_min, var_max = 0, 300

# 隨機選擇一個真實的 a, b 和 var
a_true = np.random.uniform(a_min, a_max)
b_true = np.random.uniform(b_min, b_max)
var_true = np.random.uniform(var_min, var_max)
std_dev_true = np.sqrt(var_true) # 標準差是變異數的平方根

# 生成 x 值和雜訊
x = np.random.uniform(x_min, x_max, num_points)
# N(0, var) 代表平均值為 0，標準差為 sqrt(var) 的常態分佈雜訊
noise = np.random.normal(0, std_dev_true, num_points)

# 根據公式 y = ax + b + noise 生成 y 值
y = a_true * x + b_true + noise

print(f"真實的參數 (用來生成數據):")
print(f"  - 真實 a: {a_true:.4f}")
print(f"  - 真實 b: {b_true:.4f}")
print(f"  - 真實 Variance: {var_true:.4f} (Standard Deviation: {std_dev_true:.4f})")
print("-" * 20)


# --- 步驟 2: 執行線性迴歸 ---
print("--- 2. 正在執行線性迴歸... ---")

# scikit-learn 需要 X 是二維陣列，所以我們需要 reshape x
# (-1, 1) 的意思是 "行數自動計算，1 欄"
X = x.reshape(-1, 1)

# 建立並擬合模型
model = LinearRegression()
model.fit(X, y)

# --- 步驟 3: 找出最佳 a (斜率) 和 b (截距) ---
# 從模型中提取擬合結果
a_fit = model.coef_[0]
b_fit = model.intercept_

print("線性迴歸擬合結果:")
print(f"  - 擬合出的最佳 a (斜率): {a_fit:.4f}")
print(f"  - 擬合出的 b (截距): {b_fit:.4f}")
print("比較：擬合出的 a, b 應該會很接近真實的 a, b。")
print("-" * 20)


# --- 步驟 4: 找出前 10 名的離群值 (Outliers) ---
print("--- 3. 正在找出前 10 名的離群值... ---")

# 預測所有點的 y 值
y_pred = model.predict(X)

# 計算每個點的殘差 (真實值與預測值的差距)
residuals = y - y_pred

# 使用 pandas DataFrame 來方便地處理數據
df = pd.DataFrame({
    'x': x,
    'y_true': y,
    'y_predicted': y_pred,
    'residual': residuals,
    'abs_residual': np.abs(residuals) # 我們關心的是距離，所以取絕對值
})

# 根據絕對殘差從大到小排序，並選出前 10 名
top_10_outliers = df.sort_values(by='abs_residual', ascending=False).head(10)

print("前 10 名離群值 (距離迴歸線最遠的點):")
print(top_10_outliers[['x', 'y_true', 'residual']])
print("-" * 20)


# --- 步驟 5: 視覺化結果 ---
print("--- 4. 正在繪製結果圖表... ---")

# 處理 matplotlib 中文顯示問題
# 請確認您的系統中有支援中文的字體，例如 'Microsoft JhengHei' (Windows) 或 'PingFang TC' (macOS)
# 下方是一個範例，如果出錯可以嘗試更換為您系統中的字體名稱
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
    plt.rcParams['axes.unicode_minus'] = False # 解決負號顯示問題
except:
    print("警告：找不到 'Microsoft JhengHei' 字體，圖表中的中文可能無法正常顯示。")


plt.figure(figsize=(15, 8))

# 1. 繪製原始數據點
plt.scatter(x, y, alpha=0.6, label='原始數據點')

# 2. 繪製真實的生成線 (用來生成數據的線)
plt.plot(x, a_true * x + b_true, 'g--', linewidth=2, label=f'真實生成線 (a={a_true:.2f})')

# 3. 繪製擬合的迴歸線
plt.plot(x, y_pred, 'r-', linewidth=2, label=f'線性迴歸線 (a={a_fit:.2f})')

# 4. 標示出前 10 名的離群值
plt.scatter(
    top_10_outliers['x'], 
    top_10_outliers['y_true'], 
    s=100,                  # 放大標示
    facecolors='none',      # 空心圓
    edgecolors='k',         # 黑色邊框
    linewidths=1.5,
    label='前 10 名離群值'
)

plt.title('線性迴歸分析與離群值識別', fontsize=16)
plt.xlabel('X 值', fontsize=12)
plt.ylabel('Y 值', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
