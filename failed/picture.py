import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv('carpet_miqp.csv')

def du2rgb(du):
    if du > 0.4:
        c = min(du / 2, 1)
        return (c, 0, 0)
    elif du < -0.4:
        c = min(-du / 2, 1)
        return (0, 0, c)
    else:
        return (1, 1, 1)

# 1. y(t) vs u(t-delay)
colors = [du2rgb(v) for v in df['delta_u_0']]
plt.figure(figsize=(6,5))
plt.scatter(df['y_t'], df['u_t'], c=colors, s=8, alpha=0.7)
plt.xlabel('y(t)')
plt.ylabel('u(t-delay)')
plt.title('Delta-u0 Decision Heatmap')
plt.grid(True, ls='--', alpha=0.3)
plt.show()

# 2. PCA 2D
pca2 = PCA(n_components=2)
X2 = pca2.fit_transform(df[['y_t', 'u_t']])
plt.figure(figsize=(6,5))
plt.scatter(X2[:,0], X2[:,1], c=colors, s=8, alpha=0.7)
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.title('PCA 2D Colored by Delta-u0')
plt.grid(True, ls='--', alpha=0.3)
plt.show()