# @title Full SBS to Red-Cyan Anaglyph
#!/usr/bin/env python3
# SBS → Red/Cyan Anaglyph (Automatic, Cross-correlation guided focus)
# CPU-only or CPU+CUDA threaded pipeline
#
# 3-stage threaded design:
# - Stage 1: CPU decode
# - Stage 2: CPU process or CUDA process
# - Stage 3: CPU encode

import os
import sys
import time
import threading
import queue
from typing import Tuple, Optional
from tqdm import tqdm
import cv2
import numpy as np
import multiprocessing

# -----------------------
# Helpers
# -----------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def split_full_sbs(frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = frame_bgr.shape[:2]
    half = w // 2
    left = frame_bgr[:, :half]
    right = frame_bgr[:, half:]
    return left, right

# -----------------------
# Cross-correlation shift estimation
# -----------------------

def estimate_shift_crosscorr(left_gray, right_gray, max_disp=300):
    h, w = left_gray.shape
    band_h = h // 3
    y1 = h//2 - band_h//2
    y2 = h//2 + band_h//2
    left_band = left_gray[y1:y2, :]
    right_band = right_gray[y1:y2, :]

    pad = max_disp
    right_padded = cv2.copyMakeBorder(right_band, 0, 0, pad, pad, borderType=cv2.BORDER_REPLICATE)

    res = cv2.matchTemplate(right_padded, left_band, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)

    shift = max_loc[0] - pad
    return clamp(shift, -max_disp, max_disp)

# -----------------------
# Anaglyph creation (CPU and CUDA)
# -----------------------

def make_anaglyph_cpu(left: np.ndarray, right: np.ndarray, focus_px: int, mode: str) -> np.ndarray:
    h, w = left.shape[:2]
    right = cv2.resize(right, (w, h), interpolation=cv2.INTER_AREA)

    half_sep = int(focus_px) // 2
    M_L = np.float32([[1, 0, +half_sep], [0, 1, 0]])
    M_R = np.float32([[1, 0, -half_sep], [0, 1, 0]])

    left_s = cv2.warpAffine(left, M_L, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    right_s = cv2.warpAffine(right, M_R, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    if mode == "Color":
        out = np.zeros_like(left_s)
        out[:, :, 2] = left_s[:, :, 2]
        out[:, :, 1] = right_s[:, :, 1]
        out[:, :, 0] = right_s[:, :, 0]
    elif mode == "Half-color":
        grayL = cv2.cvtColor(left_s, cv2.COLOR_BGR2GRAY)
        out = np.zeros_like(left_s)
        out[:, :, 2] = grayL
        out[:, :, 1] = right_s[:, :, 1]
        out[:, :, 0] = right_s[:, :, 0]
    else:
        grayL = cv2.cvtColor(left_s, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(right_s, cv2.COLOR_BGR2GRAY)
        out = cv2.merge([grayR, grayR, grayL])
    return out

def make_anaglyph_cuda(left: np.ndarray, right: np.ndarray, focus_px: int, mode: str) -> np.ndarray:
    if not (hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0):
        return make_anaglyph_cpu(left, right, focus_px, mode)

    h, w = left.shape[:2]
    gL = cv2.cuda_GpuMat(); gR = cv2.cuda_GpuMat()
    gL.upload(left); gR.upload(right)
    if right.shape[:2] != (h, w):
        gR = cv2.cuda.resize(gR, (w, h), interpolation=cv2.INTER_AREA)

    half_sep = int(focus_px) // 2
    M_L = np.float32([[1, 0, +half_sep], [0, 1, 0]])
    M_R = np.float32([[1, 0, -half_sep], [0, 1, 0]])

    gLs = cv2.cuda.warpAffine(gL, M_L, (w, h))
    gRs = cv2.cuda.warpAffine(gR, M_R, (w, h))

    if mode == "Color":
        chR = cv2.cuda.split(gRs)
        chL = cv2.cuda.split(gLs)
        gOut = cv2.cuda.merge([chR[0], chR[1], chL[2]])
    elif mode == "Half-color":
        gGrayL = cv2.cuda.cvtColor(gLs, cv2.COLOR_BGR2GRAY)
        chR = cv2.cuda.split(gRs)
        gOut = cv2.cuda.merge([chR[0], chR[1], gGrayL])
    else:
        gGrayL = cv2.cuda.cvtColor(gLs, cv2.COLOR_BGR2GRAY)
        gGrayR = cv2.cuda.cvtColor(gRs, cv2.COLOR_BGR2GRAY)
        gOut = cv2.cuda.merge([gGrayR, gGrayR, gGrayL])

    return gOut.download()

# -----------------------
# FPS selection
# -----------------------

def choose_export_fps(cap: cv2.VideoCapture, method: str, custom_fps: Optional[float]) -> float:
    if method == 'custom' and custom_fps is not None:
        return max(1.0, float(custom_fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps_native = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if method == 'frames_duration' and total_frames > 0:
        if fps_native > 0:
            duration = total_frames / fps_native
        else:
            duration = total_frames / 30.0
        return total_frames / max(1e-6, duration)
    return fps_native if fps_native > 0 else 30.0

# -----------------------
# Main pipeline (CPU or CUDA, with multi-worker processing)
# -----------------------

def process_video(input_path: str, output_path: str, color_mode: str, fps_method: str,
                  custom_fps: Optional[float], use_cuda: bool = False, queue_size: int = 16):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("ERROR: cannot open input:", input_path)
        return 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    pbar = tqdm(total=total_frames, desc="Processing", unit="frame")

    export_fps = choose_export_fps(cap, fps_method, custom_fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame0 = cap.read()
    if not ok or frame0.shape[1] % 2 != 0:
        print("ERROR: invalid SBS input.")
        return 1

    left0, right0 = split_full_sbs(frame0)
    if use_cuda:
        sample_out = make_anaglyph_cuda(left0, right0, focus_px=0, mode=color_mode)
    else:
        sample_out = make_anaglyph_cpu(left0, right0, focus_px=0, mode=color_mode)
    h, w = sample_out.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, export_fps, (w, h))
    if not writer.isOpened():
        print("ERROR: cannot open writer for:", output_path)
        return 1

    q_dec = queue.Queue(maxsize=queue_size)
    q_proc = queue.Queue(maxsize=queue_size)

    stop = threading.Event()

    def t_decode():
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            idx = 0
            while not stop.is_set():
                ok, frm = cap.read()
                if not ok:
                    break
                q_dec.put((idx, frm))
                idx += 1
        finally:
            q_dec.put(None)

    def worker_process():
        try:
            while True:
                item = q_dec.get()
                if item is None:
                    q_dec.put(None)  # propagate sentinel
                    break
                idx, frm = item
                left, right = split_full_sbs(frm)
                grayL = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
                grayR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
                focus_px = estimate_shift_crosscorr(grayL, grayR)
                if use_cuda:
                    out = make_anaglyph_cuda(left, right, focus_px=focus_px, mode=color_mode)
                else:
                    out = make_anaglyph_cpu(left, right, focus_px=focus_px, mode=color_mode)
                q_proc.put((idx, out))
        finally:
            q_proc.put(None)

    def t_encode():
        nonlocal pbar
        next_idx = 0
        buffer = {}
        end_workers = 0
        total_workers = multiprocessing.cpu_count()
        while True:
            item = q_proc.get()
            if item is None:
                end_workers += 1
                if end_workers >= total_workers:
                    break
                continue
            idx, frame = item
            buffer[idx] = frame
            while next_idx in buffer:
                writer.write(buffer.pop(next_idx))
                next_idx += 1
                pbar.update(1)
        pbar.close()

    th_dec = threading.Thread(target=t_decode, daemon=True)

    num_workers = multiprocessing.cpu_count()
    workers = [threading.Thread(target=worker_process, daemon=True) for _ in range(num_workers)]
    th_enc = threading.Thread(target=t_encode, daemon=True)

    t0 = time.time()
    th_dec.start()
    for wkr in workers: wkr.start()
    th_enc.start()

    try:
        while th_enc.is_alive():
            time.sleep(0.02)
    finally:
        stop.set(); th_dec.join();
        for wkr in workers: wkr.join()
        th_enc.join()
        cap.release(); writer.release()

    print(f"Done.\nSaved: {output_path} | Elapsed: {time.time()-t0:.1f}s\n")
    return 0

# -----------------------
# Prompts
# -----------------------

def prompt_choices():
    print("=== SBS → Anaglyph ===")

    in_path = input("Input full-width SBS video path: ").strip().strip('"').strip("'")
    out_path = input("Output video path (e.g., output.mp4): ").strip().strip('"').strip("'")

    print("\nFPS method:")
    print("  1) Custom FPS")
    print("  2) Frames and Duration Based")
    print("  3) OpenCV Native")
    fps_choice = input("Choose [1/2/3]: ").strip()
    if fps_choice == "1":
        fps_method = "custom"
        try:
            custom_fps = float(input("Enter FPS (>=1): ").strip())
        except Exception:
            custom_fps = 30.0
    elif fps_choice == "2":
        fps_method = "frames_duration"; custom_fps = None
    else:
        fps_method = "opencv_native"; custom_fps = None

    print("\nColor mode:")
    print("  1) Color")
    print("  2) Half-color")
    print("  3) Gray")
    cm_choice = input("Choose [1/2/3]: ").strip()
    color_mode = "Color" if cm_choice == "1" else "Half-color" if cm_choice == "2" else "Gray"

    print("\nExport processing device:")
    print("  1) CPU only")
    print("  2) CPU + GPU (CUDA)")
    dev_choice = input("Choose [1/2]: ").strip()
    use_cuda = (dev_choice == "2")

    return (in_path, out_path, color_mode, fps_method, custom_fps, use_cuda)

# -----------------------
# Main
# -----------------------

def main():
    try:
        cv2.setNumThreads(cv2.getNumberOfCPUs())
    except Exception:
        pass

    in_path, out_path, color_mode, fps_method, custom_fps, use_cuda = prompt_choices()

    if not os.path.isfile(in_path):
        print("ERROR: Input file not found:", in_path); return 1

    if use_cuda and not (hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0):
        print("WARN: CUDA not available, falling back to CPU.\n")
        use_cuda = False

    ret = process_video(
        input_path=in_path,
        output_path=out_path,
        color_mode=color_mode,
        fps_method=fps_method,
        custom_fps=custom_fps,
        use_cuda=use_cuda
    )
    sys.exit(ret)

if __name__ == "__main__":
    main()