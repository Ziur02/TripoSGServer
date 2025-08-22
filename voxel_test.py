import torch
from triposg.models.autoencoders import TripoSGVAEModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16

latents = torch.load("./latents.pt", weights_only=True).to(device=device, dtype=dtype)
resolution = 40

dim = torch.linspace(-1.005, 1.005, resolution, dtype=dtype)
grid = torch.stack(torch.meshgrid(dim, dim, dim, indexing="ij"), dim=-1).reshape(-1, 3)
grid = grid.unsqueeze(0).to(device=device, dtype=dtype)  # [1, N, 3]

vae = TripoSGVAEModel().to(device, dtype=dtype)
from safetensors.torch import load_file
state_dict = load_file(r"D:\dev\TripoSGServer\pretrained_weights\TripoSG-scribble\vae\diffusion_pytorch_model.safetensors")
vae.load_state_dict(state_dict)
vae.eval()

with torch.no_grad():
    grid_logits = vae.decode(latents, grid).sample.to(dtype=dtype).view(resolution,resolution,resolution)
# print(grid_logits)
# grid_logits = grid_logits[1:-1, 1:-1, 1:-1]
sdf = torch.sigmoid(grid_logits) * 2 - 1  

import numpy as np
import trimesh
# 创建布尔体素
voxel_bools = (sdf < 0.0).cpu().numpy().astype(np.bool_)
np.set_printoptions(threshold=np.inf)
print(voxel_bools)
np.savetxt("voxel_bool.txt", voxel_bools.reshape(1, -1), fmt="%d")

grid = trimesh.voxel.VoxelGrid(encoding=voxel_bools)
grid.as_boxes().show()


# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider
# import numpy as np

# # voxel_bools 是 3D bool numpy array，形状 [res, res, res]
# slices = voxel_bools.astype(np.uint8)  # 转为 0/1 展示

# # 设置初始展示层
# init_layer = slices.shape[2] // 2

# fig, ax = plt.subplots()
# plt.subplots_adjust(bottom=0.25)

# # 显示初始层
# im = ax.imshow(slices[:, :, init_layer], cmap='gray', origin='lower')
# ax.set_title(f"Voxel slice z={init_layer}")

# # 添加滑动条
# ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
# slider = Slider(ax_slider, 'Z Layer', 0, slices.shape[2]-1, valinit=init_layer, valstep=1)

# # 回调函数更新图像
# def update(val):
#     layer = int(slider.val)
#     im.set_data(slices[:, :, layer])
#     ax.set_title(f"Voxel slice z={layer}")
#     fig.canvas.draw_idle()

# slider.on_changed(update)

# plt.show()


