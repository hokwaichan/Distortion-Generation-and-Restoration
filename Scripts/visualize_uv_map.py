import numpy as np
import matplotlib.pyplot as plt
import os

input_uv_maps_dir = " "
output_dir = " "

os.makedirs(output_dir, exist_ok=True)

uv_map_files = [f for f in os.listdir(input_uv_maps_dir) if f.lower().endswith('.npy')]

for uv_map_file in uv_map_files:
    uv_map_path = os.path.join(input_uv_maps_dir, uv_map_file)
    
    uv_map = np.load(uv_map_path)
    uv_x = uv_map[..., 0]
    uv_y = uv_map[..., 1]

    base_name = os.path.splitext(uv_map_file)[0]

    plt.figure()
    plt.imshow(uv_x, cmap='viridis')
    plt.colorbar()
    plt.title(f"{base_name} - UV X Map")
    plt.savefig(os.path.join(output_dir, f"{base_name}_uv_x.png"))
    plt.close()

    plt.figure()
    plt.imshow(uv_y, cmap='viridis')
    plt.colorbar()
    plt.title(f"{base_name} - UV Y Map")
    plt.savefig(os.path.join(output_dir, f"{base_name}_uv_y.png"))
    plt.close()

    print(f"UV maps for {base_name} saved to: {output_dir}")

print("All UV maps visualized and saved.")
