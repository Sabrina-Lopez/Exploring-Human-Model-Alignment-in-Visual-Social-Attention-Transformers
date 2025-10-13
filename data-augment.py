import os
import cv2
import numpy as np
import random
from PIL import Image
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
import argparse
import shutil


random.seed(42)

AUGS_MAP = {
    0: ("angle",),                      # angle only
    1: ("zoom",),                       # zoom only
    2: ("lighting",),                   # lighting only
    3: ("angle", "zoom"),               # angle + zoom
    4: ("angle", "lighting"),           # angle + lighting
    5: ("zoom", "lighting"),            # zoom + lighting
    6: ("angle", "zoom", "lighting")    # all three
}


# Helper function to apply all three torchvision transforms
def apply_torchvision_transformations(frame, angle_deg, brightness_factor, zoom_factor):
    # Convert BGR to RGB then to PIL image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Affine: combine rotation (angle_deg) and scale (zoom_factor) in one op
    img = F.affine(
        img,
        angle=float(angle_deg),
        translate=[0, 0],
        scale=float(zoom_factor),
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=0  # black pad when zooming out/rotating
    )

    # Lighting (brightness)
    if abs(brightness_factor - 1.0) > 1e-6:
        img = F.adjust_brightness(img, brightness_factor)

    # Convert back to OpenCV BGR image
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main(args): 
    # Video length and augmentation mode
    vid_length = args.vid_length
    augs_choice = args.augs
    active_augs = AUGS_MAP[augs_choice]

    # Setup data paths
    org_vid_path = os.path.join(args.org_vid_path + f'_{vid_length}')
    train_folder = os.path.join(org_vid_path, 'train')
    test_folder = os.path.join(org_vid_path, 'test')
    data_folders = [train_folder, test_folder]
    class_folders = ['Help', 'Hinder', 'Physical']

    # Output path that reflects chosen augment set
    output_data_path = args.org_vid_path + f'_{vid_length}_{augs_choice}'
    output_train_folder = os.path.join(output_data_path, 'train')
    output_test_folder = os.path.join(output_data_path, 'test')
    output_data_folders = [output_train_folder, output_test_folder]

    # Create output directories if needed
    if not os.path.exists(output_data_path):
        os.mkdir(output_data_path)
    for folder in output_data_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
        for class_name in class_folders:
            class_out_path = os.path.join(folder, class_name)
            if not os.path.exists(class_out_path):
                os.mkdir(class_out_path)

    # Define augmentation parameter lists
    angles = args.angles  
    # For lighting, these are added to 1 (e.g., 1 + 0.1 = 1.1 means a slight brightness boost)
    lighting = args.lighting  
    zooms = args.zooms


    # Process each video in both train and test folders
    for src_root, dst_root in zip(data_folders, output_data_folders):
        for class_folder in class_folders:
            src_class = os.path.join(src_root, class_folder)
            dst_class = os.path.join(dst_root, class_folder)

            for video in os.listdir(src_class):
                video_path = os.path.join(src_class, video)

                stem, ext = os.path.splitext(video)
                org_output_path = os.path.join(dst_class, f"{stem}{ext}")
                # Avoid overwriting if reran
                if not os.path.exists(org_output_path):
                    shutil.copy2(video_path, org_output_path)

                # Get video frames for augmentation
                cap = cv2.VideoCapture(video_path)
                frames = []

                while True:
                    ret, frame = cap.read()
                    if not ret: 
                        break
                    frames.append(frame)

                cap.release()
                if not frames:
                    continue

                # Sample once per clip (temporal consistency)
                # Defaults when aug not selected
                angle_idx = zoom_idx = light_idx = None
                chosen_angle = 0
                chosen_zoom  = 1.0
                bright_fac   = 1.0

                if "angle" in active_augs:
                    angle_idx = random.randrange(len(angles))
                    chosen_angle = angles[angle_idx]
                if "zoom" in active_augs:
                    zoom_idx = random.randrange(len(zooms))
                    chosen_zoom = zooms[zoom_idx]
                if "lighting" in active_augs:
                    light_idx = random.randrange(len(lighting))
                    bright_fac = 1.0 + lighting[light_idx]
                
                # Apply (angle+zoom) or zoom-only via F.affine; brightness via adjust_brightness
                frames_aug = [
                    apply_torchvision_transformations(f, chosen_angle, bright_fac, chosen_zoom)
                    for f in frames
                ]

                a = angle_idx if angle_idx is not None else "x"
                z = zoom_idx  if zoom_idx  is not None else "x"
                l = light_idx if light_idx is not None else "x"
                output_name = f"{stem}_augs_{augs_choice}_a{a}_z{z}_l{l}.mp4"
                output_path = os.path.join(dst_class, output_name)

                # Write video with original dimensions
                h, w = frames_aug[0].shape[:2]
                out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
                
                for f in frames_aug:
                    out.write(f)
                
                out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune pre-trained model")
    parser.add_argument(
        "--vid_length",
        type=int,
        default=100,
        help="Length of video (100%, 67%, 33%)"
    )
    parser.add_argument(
        "--org_vid_path",
        type=str,
        default='./org_vid_path',
        help="Path to all the videos in the custom video dataset organized by training and testing sets; can be the same as in text-to-data.py"
    )
    parser.add_argument(
        "--angles",
        nargs='+', 
        type=int,
        default=[90, 180, 270],
        help="List of angles for video augmentations"
    )
    parser.add_argument(
        "--lighting",
        nargs='+', 
        type=float,
        default=[-0.4, -0.2, 0.2, 0.4],
        help="List of brightness changes for video augmentations (e.g., 1 + (-0.4) = 0.6)"
    )
    parser.add_argument(
        "--zooms",
        nargs='+', 
        type=float,
        default=[0.8, 0.9, 1.1, 1.2],
        help="List of zoom values for video augmentations"
    )
    parser.add_argument(
        "--augs", 
        type=int, 
        choices=list(AUGS_MAP.keys()), 
        default=6,
        help="0 = angle, 1 = zoom, 2 = lighting, 3 = angle + zoom, 4 = angle + lighting, 5 = zoom + lighting, 6 = all"
    )
    args = parser.parse_args()
    main(args)