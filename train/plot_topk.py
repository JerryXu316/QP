# plot_topk.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取数据
df = pd.read_csv('top1_to_243.csv')  # columns: top_k, acc

# 2. 画图
sns.set_style('whitegrid')
plt.figure(figsize=(6, 4))
plt.plot(df['top_k'], df['acc'] * 100,
         color='#0072BD', linewidth=1.8, label='Top-k Accuracy')
plt.step(df['top_k'], df['acc'] * 100,
         where='post', color='#D95319', linewidth=0.8, alpha=0.7)

# 3. 细节
plt.title('Top-1 ~ Top-243 Accuracy Curve (95 %–100 %)')
plt.xlabel('k')
plt.ylabel('Accuracy (%)')
plt.xlim(1, 243)
plt.ylim(95, 100)        # <-- 关键：从 95 % 开始
plt.legend()
plt.tight_layout()
# 4. 保存 & 显示
plt.savefig('topk_curve.pdf', dpi=300)
plt.savefig('topk_curve.png', dpi=300)
plt.show()