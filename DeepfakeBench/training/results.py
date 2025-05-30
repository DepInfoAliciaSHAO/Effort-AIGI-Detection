import demo
import dlib
import cv2
import os
from datetime import datetime
import torch
import sys
import numpy as np
import csv

ROOT = r"E:\ShareID\TestDataSets\Celeb-DF-v2\laa-net_test_dataset\Celeb-Df-v2"
FAKE_TYPES = ["Celeb-real", "Celeb-synthesis", "YouTube-real"]
MAX_FRAMES = 32
VID_EXTENSIONS = {".mp4"}
DETECTOR_CONFIG = "training/config/detector/effort.yaml"
WEIGHTS = "./training/weights/effort_clip_L14_trainOn_FaceForensic.pth"
LANDMARK_MODEL = "./preprocessing/shape_predictor_81_face_landmarks.dat"

def results_one_image(frames, face_detector, shape_predictor):
    prob_list = []
    # ---------- infer ----------
    for idx, img in enumerate(frames, 1):
        if img is None:
            print(f"[Warning] loading wrong，skip: {idx}", file=sys.stderr)
            continue

        cls, prob = demo.infer_single_image(img, face_detector, shape_predictor, model, device)
        prob_list.append(prob.item())
        print(
            f"[{idx}/{len(frames)}] {idx} | Pred Label: {cls} "
            f"(0=Real, 1=Fake) | Fake Prob: {prob:.4f}"
        )
    return cls, prob_list

def get_frames(video_path, n):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = n
    if (n > frame_count):
        n_frames = frame_count
    # Calculate frame indices to capture
    indices = np.linspace(0, frame_count - 1, n_frames, dtype=int)
    frames = []
    for i, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            print(f"Failed to read frame at index {frame_idx}", file=sys.stderr)
    cap.release()
    print(f"Loaded {len(frames)} frames from video into memory.")
    return frames

def save_as_csv(vid_probs, root):
    res_dir = os.path.join(root, "results")
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_effort")
    file_name = os.path.join(res_dir, now_str) + '.csv'
    # Find max number of values to create header dynamically
    max_len = max(len(v) for v in vid_probs.values())

    # Save to CSV
    with open(file_name, "w", newline='') as f:
        writer = csv.writer(f, delimiter=';')
        header = ["ID"] + [f"value_{i}" for i in range(max_len)]
        writer.writerow(header)

        for key, values in vid_probs.items():
            # Pad with empty strings if values are shorter
            padded_values = values + [""] * (max_len - len(values))
            writer.writerow([key] + padded_values)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = demo.load_detector(DETECTOR_CONFIG, WEIGHTS, device)
    if LANDMARK_MODEL != "":
        face_det = dlib.get_frontal_face_detector()
        shape_predictor = dlib.shape_predictor(LANDMARK_MODEL)
    else:
        face_det, shape_predictor = None, None

    vid_probs = {}
    for fake_type in FAKE_TYPES:
        fake_type_dir = os.path.join(ROOT, fake_type)
        for file in os.listdir(fake_type_dir):
            vid_name, extension = os.path.splitext(file)
            if extension in VID_EXTENSIONS:
                frames = get_frames(os.path.join(fake_type_dir, file), MAX_FRAMES)
                cls, probs = results_one_image(frames, face_det, shape_predictor)
                vid_probs[vid_name] = probs
    save_as_csv(vid_probs, ROOT)