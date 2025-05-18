import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_one_hot_map(bboxes, output_size):
    batch_size, num_classes, _ = bboxes.shape
    one_hot_map = torch.zeros(batch_size, num_classes, *output_size, device=bboxes.device)

    for i in range(batch_size):
        for j in range(num_classes):
            xmin, ymin, xmax, ymax = map(int, bboxes[i, j].tolist())
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(output_size[1], xmax), min(output_size[0], ymax)
            one_hot_map[i, j, ymin:ymax, xmin:xmax] = 1
    
    return one_hot_map

def onehot_to_mask(mask, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = colour_codes[x.astype(np.uint8)]
    return x

def visualize_one_hot_map(one_hot_map, palette):
    batch_size, _, height, width = one_hot_map.shape
    one_hot_map_np = one_hot_map.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, K)
    
    for i in range(batch_size):
        color_mask = onehot_to_mask(one_hot_map_np[i], palette)
        plt.figure(figsize=(6, 6))
        plt.imshow(color_mask)
        plt.title(f'Batch {i} - Merged Mask')
        plt.axis('off')
        plt.show()

# 示例 bboxes 数据（6类）
bboxes = torch.tensor([
    [[0, 0, 0, 0], [120, 120, 180, 180], [200, 200, 280, 280],
     [300, 300, 380, 380], [400, 400, 480, 480], [50, 200, 150, 300]],
    [[60, 60, 130, 130], [140, 140, 200, 200], [220, 220, 300, 300],
     [320, 320, 400, 400], [420, 420, 500, 500], [100, 250, 180, 350]]
], dtype=torch.float32)

output_size = (512, 512)
palette = [[0, 0, 0],  [170, 0, 255], [0, 85, 255], [170, 255, 0],[85, 255, 0],   
           [255, 255, 127]]  # one-hot的颜色表

one_hot_map = generate_one_hot_map(bboxes, output_size)
visualize_one_hot_map(one_hot_map, palette)
