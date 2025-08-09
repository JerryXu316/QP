import re
import matplotlib.pyplot as plt

# 1) 把日志粘过来，或从文件读
log_text = """
Epoch   1/100 | train loss 0.065784  train acc 72.58% | val loss 0.030483  val acc 85.19% | lr 1.00e-04
Epoch   2/100 | train loss 0.029175  train acc 85.54% | val loss 0.028043  val acc 86.15% | lr 1.00e-04
Epoch   3/100 | train loss 0.024392  train acc 87.70% | val loss 0.033873  val acc 83.62% | lr 1.00e-04
Epoch   4/100 | train loss 0.021870  train acc 88.87% | val loss 0.026060  val acc 87.10% | lr 1.00e-04
Epoch   5/100 | train loss 0.020073  train acc 89.73% | val loss 0.024830  val acc 88.30% | lr 1.00e-04
Epoch   6/100 | train loss 0.018498  train acc 90.49% | val loss 0.025613  val acc 87.78% | lr 1.00e-04
Epoch   7/100 | train loss 0.017360  train acc 91.06% | val loss 0.026477  val acc 87.71% | lr 1.00e-04
Epoch   8/100 | train loss 0.016498  train acc 91.50% | val loss 0.022593  val acc 88.83% | lr 1.00e-04
Epoch   9/100 | train loss 0.015635  train acc 91.97% | val loss 0.020867  val acc 89.27% | lr 1.00e-04
Epoch  10/100 | train loss 0.014999  train acc 92.26% | val loss 0.023967  val acc 87.88% | lr 1.00e-04
Epoch  11/100 | train loss 0.014499  train acc 92.48% | val loss 0.027485  val acc 86.44% | lr 1.00e-04
Epoch  12/100 | train loss 0.013983  train acc 92.78% | val loss 0.025911  val acc 87.96% | lr 1.00e-04
Epoch  13/100 | train loss 0.013668  train acc 92.95% | val loss 0.021035  val acc 89.43% | lr 1.00e-04
Epoch  14/100 | train loss 0.013180  train acc 93.19% | val loss 0.018755  val acc 90.74% | lr 1.00e-04
Epoch  15/100 | train loss 0.012830  train acc 93.40% | val loss 0.018415  val acc 90.99% | lr 1.00e-04
Epoch  16/100 | train loss 0.012508  train acc 93.56% | val loss 0.019352  val acc 90.29% | lr 1.00e-04
Epoch  17/100 | train loss 0.012275  train acc 93.65% | val loss 0.020827  val acc 89.53% | lr 1.00e-04
Epoch  18/100 | train loss 0.011935  train acc 93.85% | val loss 0.026141  val acc 87.20% | lr 1.00e-04
Epoch  19/100 | train loss 0.011785  train acc 93.92% | val loss 0.019816  val acc 90.36% | lr 1.00e-04
Epoch  20/100 | train loss 0.011534  train acc 94.08% | val loss 0.018595  val acc 91.12% | lr 1.00e-04
Epoch  21/100 | train loss 0.011378  train acc 94.16% | val loss 0.019522  val acc 90.61% | lr 1.00e-04
Epoch  22/100 | train loss 0.011134  train acc 94.26% | val loss 0.020748  val acc 89.87% | lr 1.00e-04
Epoch  23/100 | train loss 0.011040  train acc 94.32% | val loss 0.020534  val acc 90.29% | lr 1.00e-04
Epoch  24/100 | train loss 0.010842  train acc 94.43% | val loss 0.025695  val acc 88.77% | lr 1.00e-04
Epoch  25/100 | train loss 0.010696  train acc 94.50% | val loss 0.019228  val acc 91.06% | lr 1.00e-04
Epoch  26/100 | train loss 0.010414  train acc 94.65% | val loss 0.016135  val acc 92.25% | lr 1.00e-04
Epoch  27/100 | train loss 0.010506  train acc 94.65% | val loss 0.023600  val acc 89.36% | lr 1.00e-04
Epoch  28/100 | train loss 0.010215  train acc 94.81% | val loss 0.017827  val acc 91.27% | lr 1.00e-04
Epoch  29/100 | train loss 0.009995  train acc 94.90% | val loss 0.018838  val acc 91.48% | lr 1.00e-04
Epoch  30/100 | train loss 0.010016  train acc 94.89% | val loss 0.017264  val acc 91.33% | lr 1.00e-04
Epoch  31/100 | train loss 0.009817  train acc 95.01% | val loss 0.020774  val acc 90.62% | lr 1.00e-04
Epoch  32/100 | train loss 0.009631  train acc 95.11% | val loss 0.018968  val acc 91.21% | lr 1.00e-04
Epoch  33/100 | train loss 0.009515  train acc 95.15% | val loss 0.021702  val acc 90.53% | lr 1.00e-04
Epoch  34/100 | train loss 0.009502  train acc 95.18% | val loss 0.021550  val acc 91.04% | lr 1.00e-04
Epoch  35/100 | train loss 0.009326  train acc 95.28% | val loss 0.016899  val acc 91.91% | lr 1.00e-04
Epoch  36/100 | train loss 0.009233  train acc 95.32% | val loss 0.019425  val acc 90.66% | lr 1.00e-04
Epoch  37/100 | train loss 0.009227  train acc 95.32% | val loss 0.019149  val acc 91.38% | lr 5.00e-05
Epoch  38/100 | train loss 0.006976  train acc 96.49% | val loss 0.016633  val acc 93.09% | lr 5.00e-05
Epoch  39/100 | train loss 0.006832  train acc 96.57% | val loss 0.016422  val acc 92.77% | lr 5.00e-05
Epoch  40/100 | train loss 0.006721  train acc 96.63% | val loss 0.016682  val acc 92.30% | lr 5.00e-05
Epoch  41/100 | train loss 0.006681  train acc 96.64% | val loss 0.019780  val acc 92.01% | lr 5.00e-05
Epoch  42/100 | train loss 0.006628  train acc 96.65% | val loss 0.019573  val acc 91.83% | lr 5.00e-05
Epoch  43/100 | train loss 0.006583  train acc 96.70% | val loss 0.020563  val acc 91.33% | lr 5.00e-05
Epoch  44/100 | train loss 0.006498  train acc 96.74% | val loss 0.020202  val acc 91.86% | lr 5.00e-05
Epoch  45/100 | train loss 0.006499  train acc 96.73% | val loss 0.019760  val acc 92.10% | lr 5.00e-05
Epoch  46/100 | train loss 0.006471  train acc 96.77% | val loss 0.014042  val acc 93.50% | lr 5.00e-05
Epoch  47/100 | train loss 0.006429  train acc 96.79% | val loss 0.018600  val acc 91.78% | lr 5.00e-05
Epoch  48/100 | train loss 0.006388  train acc 96.81% | val loss 0.019853  val acc 91.55% | lr 5.00e-05
Epoch  49/100 | train loss 0.006404  train acc 96.80% | val loss 0.018157  val acc 92.14% | lr 5.00e-05
Epoch  50/100 | train loss 0.006344  train acc 96.84% | val loss 0.018241  val acc 91.45% | lr 5.00e-05
Epoch  51/100 | train loss 0.006326  train acc 96.85% | val loss 0.020199  val acc 91.54% | lr 5.00e-05
Epoch  52/100 | train loss 0.006216  train acc 96.88% | val loss 0.016969  val acc 92.61% | lr 5.00e-05
Epoch  53/100 | train loss 0.006218  train acc 96.90% | val loss 0.017180  val acc 92.38% | lr 5.00e-05
Epoch  54/100 | train loss 0.006213  train acc 96.88% | val loss 0.019568  val acc 91.82% | lr 5.00e-05
Epoch  55/100 | train loss 0.006171  train acc 96.93% | val loss 0.019805  val acc 91.59% | lr 5.00e-05
Epoch  56/100 | train loss 0.006124  train acc 96.95% | val loss 0.023940  val acc 90.50% | lr 5.00e-05
Epoch  57/100 | train loss 0.006057  train acc 96.98% | val loss 0.019482  val acc 92.16% | lr 2.50e-05
Epoch  58/100 | train loss 0.004754  train acc 97.66% | val loss 0.019960  val acc 91.85% | lr 2.50e-05
Epoch  59/100 | train loss 0.004662  train acc 97.71% | val loss 0.021062  val acc 91.74% | lr 2.50e-05
Epoch  60/100 | train loss 0.004618  train acc 97.74% | val loss 0.020105  val acc 91.82% | lr 2.50e-05
Epoch  61/100 | train loss 0.004576  train acc 97.75% | val loss 0.018078  val acc 92.49% | lr 2.50e-05
Epoch  62/100 | train loss 0.004593  train acc 97.76% | val loss 0.020567  val acc 92.00% | lr 2.50e-05
Epoch  63/100 | train loss 0.004545  train acc 97.78% | val loss 0.021478  val acc 91.62% | lr 2.50e-05
Epoch  64/100 | train loss 0.004570  train acc 97.76% | val loss 0.020744  val acc 91.75% | lr 2.50e-05
Epoch  65/100 | train loss 0.004528  train acc 97.79% | val loss 0.021878  val acc 91.96% | lr 2.50e-05
Epoch  66/100 | train loss 0.004496  train acc 97.81% | val loss 0.022188  val acc 91.04% | lr 2.50e-05
Epoch  67/100 | train loss 0.004489  train acc 97.80% | val loss 0.020265  val acc 92.09% | lr 2.50e-05
Epoch  68/100 | train loss 0.004456  train acc 97.82% | val loss 0.020663  val acc 91.59% | lr 1.25e-05
Epoch  69/100 | train loss 0.003687  train acc 98.23% | val loss 0.020824  val acc 91.95% | lr 1.25e-05
Epoch  70/100 | train loss 0.003638  train acc 98.26% | val loss 0.019316  val acc 92.26% | lr 1.25e-05
Epoch  71/100 | train loss 0.003633  train acc 98.26% | val loss 0.021704  val acc 91.82% | lr 1.25e-05
Epoch  72/100 | train loss 0.003588  train acc 98.29% | val loss 0.021284  val acc 91.95% | lr 1.25e-05
Epoch  73/100 | train loss 0.003608  train acc 98.28% | val loss 0.020517  val acc 92.03% | lr 1.25e-05
Epoch  74/100 | train loss 0.003531  train acc 98.30% | val loss 0.021850  val acc 91.69% | lr 1.25e-05
Epoch  75/100 | train loss 0.003554  train acc 98.32% | val loss 0.020206  val acc 92.26% | lr 1.25e-05
Epoch  76/100 | train loss 0.003518  train acc 98.32% | val loss 0.021248  val acc 91.87% | lr 1.25e-05
Epoch  77/100 | train loss 0.003521  train acc 98.32% | val loss 0.021732  val acc 91.98% | lr 1.25e-05
Epoch  78/100 | train loss 0.003508  train acc 98.32% | val loss 0.020992  val acc 92.16% | lr 1.25e-05
Epoch  79/100 | train loss 0.003493  train acc 98.33% | val loss 0.021465  val acc 91.86% | lr 6.25e-06
Epoch  80/100 | train loss 0.003082  train acc 98.56% | val loss 0.020929  val acc 92.10% | lr 6.25e-06
Epoch  81/100 | train loss 0.003053  train acc 98.59% | val loss 0.020583  val acc 92.21% | lr 6.25e-06
Epoch  82/100 | train loss 0.003039  train acc 98.57% | val loss 0.020744  val acc 92.12% | lr 6.25e-06
Epoch  83/100 | train loss 0.003039  train acc 98.59% | val loss 0.020987  val acc 92.14% | lr 6.25e-06
Epoch  84/100 | train loss 0.003011  train acc 98.62% | val loss 0.021865  val acc 92.00% | lr 6.25e-06
Epoch  85/100 | train loss 0.003013  train acc 98.60% | val loss 0.021867  val acc 91.74% | lr 6.25e-06
Epoch  86/100 | train loss 0.002991  train acc 98.61% | val loss 0.021134  val acc 92.18% | lr 6.25e-06
Epoch  87/100 | train loss 0.003014  train acc 98.61% | val loss 0.020034  val acc 92.36% | lr 6.25e-06
Epoch  88/100 | train loss 0.003000  train acc 98.60% | val loss 0.022061  val acc 91.97% | lr 6.25e-06
Epoch  89/100 | train loss 0.002971  train acc 98.61% | val loss 0.021443  val acc 92.06% | lr 6.25e-06
Epoch  90/100 | train loss 0.002956  train acc 98.63% | val loss 0.021872  val acc 91.95% | lr 3.13e-06
Epoch  91/100 | train loss 0.002735  train acc 98.74% | val loss 0.022811  val acc 91.78% | lr 3.13e-06
Epoch  92/100 | train loss 0.002730  train acc 98.76% | val loss 0.022032  val acc 91.93% | lr 3.13e-06
Epoch  93/100 | train loss 0.002733  train acc 98.76% | val loss 0.021856  val acc 91.86% | lr 3.13e-06
Epoch  94/100 | train loss 0.002726  train acc 98.75% | val loss 0.021272  val acc 92.10% | lr 3.13e-06
Epoch  95/100 | train loss 0.002719  train acc 98.76% | val loss 0.021294  val acc 92.16% | lr 3.13e-06
Epoch  96/100 | train loss 0.002705  train acc 98.78% | val loss 0.022152  val acc 91.93% | lr 3.13e-06
Epoch  97/100 | train loss 0.002708  train acc 98.77% | val loss 0.021997  val acc 92.02% | lr 3.13e-06
Epoch  98/100 | train loss 0.002715  train acc 98.77% | val loss 0.020979  val acc 92.18% | lr 3.13e-06
Epoch  99/100 | train loss 0.002684  train acc 98.79% | val loss 0.021677  val acc 91.89% | lr 3.13e-06
Epoch 100/100 | train loss 0.002704  train acc 98.77% | val loss 0.022198  val acc 91.71% | lr 3.13e-06
"""

# 2) 正则提取数字
pattern = re.compile(
    r'Epoch\s+(\d+)/\d+.*?train loss ([\d\.]+).*?train acc ([\d\.]+).*?'
    r'val loss ([\d\.]+).*?val acc ([\d\.]+)'
)

epochs, tr_loss, val_loss, tr_acc, val_acc = [], [], [], [], []
for e, tl, ta, vl, va in pattern.findall(log_text):
    epochs.append(int(e))
    tr_loss.append(float(tl))
    val_loss.append(float(vl))
    tr_acc.append(float(ta))
    val_acc.append(float(va))

# 3) 画图
plt.style.use('seaborn-v0_8')
fig, ax = plt.subplots(1, 2, figsize=(12,4))

# Loss
ax[0].plot(epochs, tr_loss, label='Train Loss')
ax[0].plot(epochs, val_loss, label='Val Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('MSE Loss')
ax[0].set_yscale('log')
ax[0].legend()
ax[0].grid(True)

# Accuracy
ax[1].plot(epochs, tr_acc, label='Train Row Acc')
ax[1].plot(epochs, val_acc, label='Val Row Acc')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy (%)')
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.savefig('training_curve.png', dpi=300)
plt.show()