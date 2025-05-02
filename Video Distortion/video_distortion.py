import os
import cv2
import numpy as np
import random
import json
from noise import pnoise2


def create_checkerboard(size, num_squares):
    checkerboard = np.zeros((size, size, 3), dtype=np.uint8)
    square_size = size // num_squares

    for i in range(num_squares):
        for j in range(num_squares):
            if (i + j) % 2 == 0:
                checkerboard[i * square_size:(i + 1) * square_size,
                             j * square_size:(j + 1) * square_size] = [255, 255, 255]
    return checkerboard


def generate_smooth_random_field(width, height, scale, distortion_strength, offset_x, offset_y):
    field_x = np.zeros((height, width), np.float32)
    field_y = np.zeros((height, width), np.float32)
    for y in range(height):
        for x in range(width):
            noise_x = pnoise2((x + offset_x) / scale, (y + offset_y) / scale, octaves=6, persistence=0.5)
            noise_y = pnoise2((x + offset_x + 100) / scale, (y + offset_y + 100) / scale, octaves=6, persistence=0.5)
            field_x[y, x] = distortion_strength * noise_x
            field_y[y, x] = distortion_strength * noise_y
    return field_x, field_y


def apply_vortex_transformation(image, strength):
    height, width = image.shape[:2]
    center_x, center_y = width / 2, height / 2

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    dx = x - center_x
    dy = y - center_y
    distance = np.sqrt(dx**2 + dy**2) + 1e-5

    angle = strength / (distance + 1e-5)
    map_x = center_x + dx * np.cos(angle) - dy * np.sin(angle)
    map_y = center_y + dx * np.sin(angle) + dy * np.cos(angle)

    map_x = np.clip(map_x, 0, width - 1).astype(np.float32)
    map_y = np.clip(map_y, 0, height - 1).astype(np.float32)

    transformed_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return transformed_image


def apply_radial_distortion(image, params=None, json_path=" "):
    if params is None:
        with open(json_path, "r") as file:
            param_ranges = json.load(file)

        params = {key: random.uniform(*param_ranges[key]) for key in param_ranges}

    k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4 = [params[key] for key in
                                                      ['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'p1', 'p2', 's1', 's2', 's3', 's4']]

    h, w = image.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    x_normalized = (x - w / 2) / (w / 2)
    y_normalized = (y - h / 2) / (h / 2)
    r = np.sqrt(x_normalized**2 + y_normalized**2)

    radial_x = 1 + k1 * r**2 + k2 * r**4 + k3 * r**6
    radial_y = 1 + k4 * r**2 + k5 * r**4 + k6 * r**6

    x_tangent = 2 * p1 * x_normalized * y_normalized + p2 * (r**2 + 2 * x_normalized**2)
    y_tangent = p1 * (r**2 + 2 * y_normalized**2) + 2 * p2 * x_normalized * y_normalized

    x_shear = s1 * r**2 + s2 * r**4
    y_shear = s3 * r**2 + s4 * r**4

    x_distorted = (x_normalized * radial_x / radial_y + x_tangent + x_shear) * w / 2 + w / 2
    y_distorted = (y_normalized * radial_x / radial_y + y_tangent + y_shear) * h / 2 + h / 2

    uv_map = np.zeros((h, w, 2), dtype=np.float32)
    uv_map[..., 0] = x_distorted
    uv_map[..., 1] = y_distorted

    distorted_image = cv2.remap(image, x_distorted.astype(np.float32), y_distorted.astype(np.float32), interpolation=cv2.INTER_LINEAR)

    return distorted_image, uv_map, params


def save_uv_map(uv_map, output_dir, base_name):
    uv_output_path = os.path.join(output_dir, f"{base_name}_uv_map.npy")
    np.save(uv_output_path, uv_map)
    print(f"UV map saved: {uv_output_path}")


def process_video(input_video_path, output_dir, transformation_type, output_video_path=None, max_frames=None):
    grid_dir = os.path.join(output_dir, 'grid')
    distorted_dir = os.path.join(output_dir, 'distorted')
    original_dir = os.path.join(output_dir, 'original')
    uv_dir = os.path.join(output_dir, 'uv_maps')

    os.makedirs(grid_dir, exist_ok=True)
    os.makedirs(distorted_dir, exist_ok=True)
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(uv_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        base_name = f"frame_{frame_idx:04d}"
        transformed_output_path = os.path.join(distorted_dir, f"{base_name}_{transformation_type}.png")
        grid_output_path = os.path.join(grid_dir, f"{base_name}_grid_{transformation_type}.png")
        original_output_path = os.path.join(original_dir, f"{base_name}_original.png")

        cv2.imwrite(original_output_path, frame)

        checkerboard = create_checkerboard(size=frame.shape[1], num_squares=10)

        if transformation_type == 'random_field':
            offset_x = random.uniform(0, 1000)
            offset_y = random.uniform(0, 1000)
            field_x, field_y = generate_smooth_random_field(width, height, 50, 15, offset_x, offset_y)

            uv_map = np.zeros((height, width, 2), dtype=np.float32)
            uv_map[..., 0] = field_x
            uv_map[..., 1] = field_y

            map_x, map_y = np.meshgrid(np.arange(width), np.arange(height))
            distorted_map_x = (map_x + field_x).astype(np.float32)
            distorted_map_y = (map_y + field_y).astype(np.float32)
            transformed_image = cv2.remap(frame, distorted_map_x, distorted_map_y, interpolation=cv2.INTER_LINEAR)
            distorted_grid = cv2.remap(checkerboard, distorted_map_x, distorted_map_y, interpolation=cv2.INTER_LINEAR)

            save_uv_map(uv_map, uv_dir, base_name)

        elif transformation_type == 'radial_distortion':
            transformed_image, uv_map, distortion_params = apply_radial_distortion(frame)
            distorted_grid, _, _ = apply_radial_distortion(checkerboard, params=distortion_params)

            save_uv_map(uv_map, uv_dir, base_name)

        elif transformation_type == 'vortex_transformation':
            transformed_image = apply_vortex_transformation(frame, strength=500)
            distorted_grid = apply_vortex_transformation(checkerboard, strength=500)

        else:
            raise ValueError(f"Unknown transformation type: {transformation_type}")

        cv2.imwrite(transformed_output_path, transformed_image)
        cv2.imwrite(grid_output_path, distorted_grid)
        print(f"Frame {frame_idx} processed and saved.")

        if output_video_path:
            video_writer.write(transformed_image)

        frame_idx += 1

    cap.release()
    if output_video_path:
        video_writer.release()
    print("Video processing complete.")


if __name__ == "__main__":
    input_video_path = " "
    transformation_type = "random_field"
    output_dir = " "
    output_video_path = " "

    process_video(
        input_video_path,
        output_dir,
        transformation_type,
        output_video_path,
        max_frames=20
    )
    