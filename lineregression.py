import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# --- 頁面與狀態設定 ---
st.set_page_config(page_title="線性迴歸模擬器", page_icon="📈", layout="wide")

# 初始化隨機種子，讓滑動滑桿時數據點不會瘋狂亂跳
if "random_seed" not in st.session_state:
    st.session_state.random_seed = 42

st.title("📈 動態線性迴歸與離群值分析模擬器")
st.write("透過左側欄位動態調整變數，立即查看線性迴歸與離群值 (Outliers) 的變化。")

# 顯示線性迴歸公式
st.markdown("##### 數據生成公式：")
st.latex(r"y = a \cdot x + b + \epsilon \quad (\text{其中雜訊 } \epsilon \sim \mathcal{N}(0, \text{var}))")

# --- 側邊欄：動態變數調整 ---
with st.sidebar:
    st.header("⚙️ 動態參數設定")
    
    # 根據你的需求，設定 n, a, b, var 的動態滑桿
    num_points = st.slider("數據點數量 (n)", min_value=50, max_value=1000, value=200, step=50)
    
    st.subheader("設定真實參數")
    # a=-50~50, b=0~100, var=0~300
    a_true = st.slider("真實斜率 (a)", min_value=-50.0, max_value=50.0, value=25.0, step=1.0)
    b_true = st.slider("真實截距 (b)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    var_true = st.slider("雜訊變異數 (var)", min_value=0.0, max_value=300.0, value=150.0, step=10.0)
    
    st.divider()
    st.subheader("🎨 顏色設定")
    color_data = st.color_picker("數據點顏色", "#87CEEB")      # 預設：天藍色
    color_true = st.color_picker("真實線顏色", "#2CA02C")      # 預設：綠色
    color_reg = st.color_picker("迴歸線顏色", "#D62728")       # 預設：紅色
    color_outlier = st.color_picker("離群值標示顏色", "#FF0000") # 預設：大紅色

    st.divider()
    if st.button("🎲 重新生成隨機雜訊"):
        # 點擊按鈕時更換種子，重新打亂點的分佈
        st.session_state.random_seed = np.random.randint(0, 10000)
        
    st.info("💡 提示：調整上方滑桿，右側圖表與數據會即時更新。點擊骰子按鈕可重新生成分佈。")

# --- 步驟 1: 生成數據 ---
# 固定隨機種子以獲得平滑的 UI 體驗
np.random.seed(st.session_state.random_seed)

std_dev_true = np.sqrt(var_true)
x = np.random.uniform(-100, 100, num_points)
noise = np.random.normal(0, std_dev_true, num_points)
y = a_true * x + b_true + noise

# --- 步驟 2: 執行線性迴歸 ---
X = x.reshape(-1, 1)
model = LinearRegression()
model.fit(X, y)

a_fit = model.coef_[0]
b_fit = model.intercept_

# --- 步驟 3: 找出前 10 名離群值 ---
y_pred = model.predict(X)
residuals = y - y_pred

df = pd.DataFrame({
    'x': x,
    'y_true': y,
    'y_predicted': y_pred,
    'residual': residuals,
    'abs_residual': np.abs(residuals)
})

top_10_outliers = df.sort_values(by='abs_residual', ascending=False).head(10)

# 計算 R 平方值 (R-squared)
r_squared = model.score(X, y)

# --- 步驟 4: 介面呈現與視覺化 ---

# 顯示對比指標
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("最佳擬合斜率 (a)", f"{a_fit:.4f}", f"與真實差: {a_fit - a_true:.4f}")
with col2:
    st.metric("最佳擬合截距 (b)", f"{b_fit:.4f}", f"與真實差: {b_fit - b_true:.4f}")
with col3:
    st.metric("雜訊標準差", f"{std_dev_true:.4f}", f"變異數 (var): {var_true}")
with col4:
    st.metric("R 平方值 (R²)", f"{r_squared:.4f}", "模型解釋力")

st.divider()

col_chart, col_data = st.columns([2, 1])

with col_chart:
    st.subheader("📊 迴歸分析與離群值圖表")
    
    # 建立圖表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 為了避免 Streamlit 伺服器上缺少中文字體導致亂碼，圖表內部文字使用英文
    ax.scatter(x, y, alpha=0.6, label='Data Points', color=color_data)
    ax.plot(x, a_true * x + b_true, linestyle='--', color=color_true, linewidth=2, label=f'True Line (a={a_true:.2f})')
    ax.plot(x, y_pred, linestyle='-', color=color_reg, linewidth=2, label=f'Regression Line (a={a_fit:.2f})')
    
    # 標示出離群值
    ax.scatter(
        top_10_outliers['x'], top_10_outliers['y_true'], 
        s=150, facecolors='none', edgecolors=color_outlier, linewidths=2, label='Top 10 Outliers'
    )
    
    ax.set_title('Linear Regression and Outliers Analysis')
    ax.set_xlabel('X Value')
    ax.set_ylabel('Y Value')
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig)

with col_data:
    st.subheader("🚨 前 10 名離群值資料")
    display_df = top_10_outliers[['x', 'y_true', 'residual']].copy()
    display_df.columns = ['X 座標', '真實 Y 值', '殘差 (Residual)']
    # 使用 dataframe 展示表格
    st.dataframe(display_df, use_container_width=True, hide_index=True)
