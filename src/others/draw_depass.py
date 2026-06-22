import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# =========================
# 1️⃣ 全局风格设置
# =========================

mpl.rcParams['pdf.fonttype'] = 42      # 避免 Type 3 字体
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Liberation Sans']
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['xtick.major.width'] = 1.2
mpl.rcParams['ytick.major.width'] = 1.2

# =========================
# 2️⃣ 自定义配色（可修改）
# =========================

# color_A = "#D3D6AA"     # 深蓝
color_A = '#f8d4c1'  # 浅黄
# color_B = "#BAD2E7"     # 深红
color_B = '#7CD0F1'  # 浅红
color_line = "#aaaaaa"  # 连线灰色

# =========================
# 3️⃣ 读取数据
# =========================

df_A = pd.read_csv("work_dirs/others/no_cot.csv")
df_B = pd.read_csv("work_dirs/others/cot_pos.csv")

# df_B中只保留yes_prob大于0.5的行
df_B = df_B[df_B['yes_prob'] > 0.5]

df_A.columns = df_A.columns.str.strip()
df_B.columns = df_B.columns.str.strip()

df = pd.merge(
    df_B[['ID', 'B1', 'B2']],
    df_A[['ID', 'A1', 'A2']],
    on='ID',
    how='left'
).dropna()

eps = 1e-12
A_ratio = (df['A1'] + eps) / (df['A2'] + eps)
B_ratio = (df['B1'] + eps) / (df['B2'] + eps)
ratio_of_ratios = A_ratio / B_ratio

# =========================
# 4️⃣ 画图
# =========================

fig, axes = plt.subplots(1, 2, figsize=(8, 3))  # 适合双栏论文宽度

# ----- 左图：配对 -----
for i in range(len(A_ratio)):
    axes[0].plot([1, 2],
                 [A_ratio.iloc[i], B_ratio.iloc[i]],
                 color=color_line,
                 alpha=0.4,
                 linewidth=0.8)

axes[0].scatter(np.ones(len(A_ratio)),
                A_ratio,
                color=color_A,
                s=25,
                zorder=3,
                label="Vanilla")

axes[0].scatter(np.ones(len(B_ratio))*2,
                B_ratio,
                color=color_B,
                s=25,
                zorder=3,
                label="LGS")

axes[0].set_xticks([1, 2])
axes[0].set_xticklabels(["Vanilla", "LGS"])
axes[0].set_ylabel("Ratio")
# axes[0].set_title("Paired Ratios", fontsize=11)
axes[0].text(
    -0.12, 1.05,
    "(a) Paired Ratios",
    transform=axes[0].transAxes,
    fontsize=11,
    fontweight='bold',
    va='bottom',
    ha='left',
    clip_on=False
)
# axes[0].legend(frameon=False)

# ----- 右图：相对倍数 -----
vp = axes[1].violinplot(ratio_of_ratios,
                        positions=[1],
                        showmeans=True)

for body in vp['bodies']:
    body.set_facecolor(color_A)
    body.set_alpha(0.5)

axes[1].scatter(np.ones(len(ratio_of_ratios)),
                ratio_of_ratios,
                color='black',
                s=18,
                alpha=0.6)

axes[1].axhline(1.0,
                linestyle='--',
                linewidth=1,
                color='black')

axes[1].set_xticks([1])
axes[1].set_xticklabels(["(Vanilla/LGS)"])
axes[1].set_ylabel("Relative Ratio")
# axes[1].set_title("Relative Ratio Distribution", fontsize=11)
axes[1].text(
    -0.15, 1.05,
    "(b) Relative Ratio Distribution",
    transform=axes[1].transAxes,
    fontsize=11,
    fontweight='bold',
    va='bottom',
    ha='left',
    clip_on=False
)

plt.tight_layout()

# =========================
# 5️⃣ 导出矢量 PDF
# =========================

plt.savefig("depass.pdf",
            format="pdf",
            bbox_inches="tight")

plt.close()

print("Saved as depass.pdf")