from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 你的拼接图路径
image_path = "epoch-400.png"

# 1. 读取图片
img = Image.open(image_path).convert("RGB")
w, h = img.size

# 2. 按高度切成上下两部分
original_img = img.crop((0, 0, w, h // 2))
encoded_img = img.crop((0, h // 2, w, h))

# 3. 转成 numpy array，并归一化到 [0, 1]
original = np.asarray(original_img).astype(np.float32) / 255.0
encoded = np.asarray(encoded_img).astype(np.float32) / 255.0

# 4. 计算逐像素差异
diff = np.abs(encoded - original)

# 5. 对 RGB 三个 channel 取平均，得到二维 heatmap
heatmap = diff.mean(axis=2)

# 6. 为了可视化，把差异放大
amplification = 20
heatmap_vis = np.clip(heatmap * amplification, 0, 1)

# 7. 保存原图、encoded图、heatmap
original_img.save("original_row.png")
encoded_img.save("encoded_row.png")

plt.figure(figsize=(16, 2))
plt.imshow(heatmap_vis, cmap="gray", vmin=0, vmax=1)
plt.axis("off")
plt.savefig("difference_heatmap.png", dpi=300, bbox_inches="tight", pad_inches=0)
plt.close()

print("Saved:")
print("original_row.png")
print("encoded_row.png")
print("difference_heatmap.png")