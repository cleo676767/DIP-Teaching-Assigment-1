import numpy as np
import torch
import torch.nn as nn
from pytorch3d.transforms import euler_angles_to_matrix
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
points2d = np.load("data/points2d.npz")
colors = np.load("data/points3d_colors.npy")

# Extract observations
view_keys = sorted(points2d.keys())
n_views = len(view_keys)
n_points = len(colors)

obs_list = []
vis_list = []
for key in view_keys:
    data = points2d[key]  # (N, 3): [x, y, vis]
    obs_list.append(data[:, :2])
    vis_list.append(data[:, 2])

observations = torch.tensor(np.stack(obs_list), dtype=torch.float32)  # (50, 20000, 2)
visibility = torch.tensor(np.stack(vis_list), dtype=torch.float32)    # (50, 20000)

# Initialize parameters
img_size = 1024
cx, cy = img_size / 2, img_size / 2

focal = nn.Parameter(torch.tensor(1000.0))
euler_angles = nn.Parameter(torch.zeros(n_views, 3))
translations = nn.Parameter(torch.tensor([[0.0, 0.0, -2.5]] * n_views))
points3d = nn.Parameter(torch.randn(n_points, 3) * 0.5)

optimizer = torch.optim.Adam([focal, euler_angles, translations, points3d], lr=1e-2)

# Training
losses = []
for epoch in range(500):
    optimizer.zero_grad()
    
    R = euler_angles_to_matrix(euler_angles, "XYZ")  # (50, 3, 3)
    Xc = torch.einsum('vij,pj->vpi', R, points3d) + translations.unsqueeze(1)  # (50, 20000, 3)
    
    u = -focal * Xc[..., 0] / Xc[..., 2] + cx
    v = focal * Xc[..., 1] / Xc[..., 2] + cy
    projected = torch.stack([u, v], dim=-1)  # (50, 20000, 2)
    
    error = (projected - observations) * visibility.unsqueeze(-1)
    loss = (error ** 2).sum() / visibility.sum()
    
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Focal: {focal.item():.2f}")

# Plot loss
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Bundle Adjustment Loss")
plt.grid(True)
plt.savefig("loss_curve.png", dpi=150)
plt.close()
print("Saved loss_curve.png")

# Save point cloud
points_np = points3d.detach().cpu().numpy()
colors_np = colors / 255.0

with open("reconstructed_points.obj", "w") as f:
    for i in range(n_points):
        x, y, z = points_np[i]
        r, g, b = colors_np[i]
        f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n")

print("Saved reconstructed_points.obj")
