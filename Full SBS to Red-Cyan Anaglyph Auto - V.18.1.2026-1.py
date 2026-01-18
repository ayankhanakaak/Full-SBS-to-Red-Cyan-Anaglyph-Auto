# @title Full SBS to Red-Cyan Anaglyph Auto - V.18.1.2026-1
#!/usr/bin/env python3
"""
SBS → Red/Cyan Anaglyph Converter
Version: 18.1.2026-1
Made by: Ayan Khan

Features:
- PyTorch GPU acceleration (works on Colab, Kaggle, local)
- FFmpeg NVDEC/NVENC hardware video decode/encode
- Multi-threaded pipeline
- Automatic focus detection via cross-correlation
- Multiple color modes (Color, Half-color, Gray)
"""

import os
import sys
import time
import threading
import queue
import subprocess
from typing import Tuple, Optional
from tqdm import tqdm
import cv2
import numpy as np
import multiprocessing

# Suppress OpenCV warnings
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"

# -----------------------
# PyTorch Setup
# -----------------------

import torch
import torch.nn.functional as F

def check_torch_cuda():
    """Check if PyTorch CUDA is available"""
    if torch.cuda.is_available():
        return True, torch.cuda.get_device_name(0)
    return False, None

def check_nvenc_support():
    """Check if FFmpeg NVENC is available"""
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True, text=True, timeout=10
        )
        return 'h264_nvenc' in result.stdout
    except:
        return False

def check_nvdec_support():
    """Check if FFmpeg NVDEC is available"""
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-hwaccels'],
            capture_output=True, text=True, timeout=10
        )
        return 'cuda' in result.stdout
    except:
        return False

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
    """Estimate horizontal shift using cross-correlation"""
    h, w = left_gray.shape
    band_h = h // 3
    y1 = h // 2 - band_h // 2
    y2 = h // 2 + band_h // 2
    left_band = left_gray[y1:y2, :]
    right_band = right_gray[y1:y2, :]

    pad = max_disp
    right_padded = cv2.copyMakeBorder(right_band, 0, 0, pad, pad, borderType=cv2.BORDER_REPLICATE)

    res = cv2.matchTemplate(right_padded, left_band, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)

    shift = max_loc[0] - pad
    return clamp(shift, -max_disp, max_disp)

# -----------------------
# Video Readers
# -----------------------

class NVDECReader:
    """Hardware-accelerated video reader using FFmpeg NVDEC"""
    def __init__(self, video_path, w, h, fps, start_frame=0, end_frame=None):
        self.w, self.h = w, h
        self.frame_size = w * h * 3
        self.frames_read = 0
        self.end_frame = end_frame
        
        start_time = start_frame / fps if start_frame > 0 else 0
        
        cmd = [
            'ffmpeg', '-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda',
            '-ss', str(start_time), '-i', video_path,
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-vsync', '0', 'pipe:1'
        ]
        
        if end_frame is not None:
            frames_to_read = end_frame - start_frame
            cmd.insert(-4, '-frames:v')
            cmd.insert(-4, str(frames_to_read))
        
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=self.frame_size * 10
        )
        
        # Test read
        test_data = self.process.stdout.read(1)
        if not test_data:
            raise RuntimeError("NVDEC failed to start")
        self._first_byte = test_data
        self._first_frame = True
    
    def read(self):
        if self.end_frame is not None and self.frames_read >= (self.end_frame - (self.end_frame - self.frames_read)):
            pass  # Let it continue until EOF
        
        if self._first_frame:
            raw = self._first_byte + self.process.stdout.read(self.frame_size - 1)
            self._first_frame = False
        else:
            raw = self.process.stdout.read(self.frame_size)
        
        if len(raw) != self.frame_size:
            return False, None
        
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.h, self.w, 3)).copy()
        self.frames_read += 1
        return True, frame
    
    def release(self):
        try:
            self.process.terminate()
            self.process.wait(timeout=5)
        except:
            try:
                self.process.kill()
            except:
                pass

class StandardReader:
    """Standard OpenCV video reader"""
    def __init__(self, video_path, start_frame=0):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open video")
        if start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.frames_read = 0
    
    def read(self):
        ret, frame = self.cap.read()
        if ret:
            self.frames_read += 1
        return ret, frame
    
    def release(self):
        self.cap.release()

# -----------------------
# Video Writers
# -----------------------

class NVENCWriter:
    """Hardware-accelerated video writer using FFmpeg NVENC"""
    def __init__(self, output_path, w, h, fps, crf=20):
        self.w, self.h = w, h
        cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{w}x{h}', '-r', str(fps), '-i', 'pipe:0',
            '-c:v', 'h264_nvenc', '-preset', 'p4', '-rc', 'vbr',
            '-cq', str(crf), '-b:v', '0', '-pix_fmt', 'yuv420p', output_path
        ]
        self.process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            bufsize=w * h * 3 * 10
        )
        self.frames_written = 0
    
    def write(self, frame):
        try:
            self.process.stdin.write(frame.tobytes())
            self.frames_written += 1
            return True
        except BrokenPipeError:
            return False
    
    def release(self):
        try:
            self.process.stdin.close()
            self.process.wait(timeout=30)
        except:
            try:
                self.process.kill()
            except:
                pass

class StandardWriter:
    """Standard OpenCV video writer"""
    def __init__(self, output_path, w, h, fps):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to create writer for {output_path}")
        self.frames_written = 0
    
    def write(self, frame):
        self.writer.write(frame)
        self.frames_written += 1
        return True
    
    def release(self):
        self.writer.release()

# -----------------------
# Anaglyph Creation (PyTorch GPU)
# -----------------------

def make_anaglyph_torch(left: np.ndarray, right: np.ndarray, focus_px: int, 
                         mode: str, device: torch.device) -> np.ndarray:
    """
    Create anaglyph using PyTorch GPU acceleration.
    
    Args:
        left: Left eye image (BGR, uint8)
        right: Right eye image (BGR, uint8)
        focus_px: Focus shift in pixels
        mode: "Color", "Half-color", or "Gray"
        device: torch device (cuda or cpu)
    
    Returns:
        Anaglyph image (BGR, uint8)
    """
    h, w = left.shape[:2]
    
    # Convert to torch tensors (H, W, C) and move to device
    left_t = torch.from_numpy(left).to(device).float()
    right_t = torch.from_numpy(right).to(device).float()
    
    # Resize right if dimensions don't match
    if right.shape[:2] != (h, w):
        # Reshape to (1, C, H, W) for interpolate
        right_t = right_t.permute(2, 0, 1).unsqueeze(0)
        right_t = F.interpolate(right_t, size=(h, w), mode='area')
        right_t = right_t.squeeze(0).permute(1, 2, 0)
    
    half_sep = int(focus_px) // 2
    
    # Apply horizontal shifts using padding
    if half_sep > 0:
        # Shift left image RIGHT by half_sep
        # Pad left side, crop right side
        left_p = left_t.permute(2, 0, 1)  # (C, H, W)
        left_p = F.pad(left_p, (half_sep, 0, 0, 0), mode='replicate')
        left_shifted = left_p[:, :, :w].permute(1, 2, 0)
        
        # Shift right image LEFT by half_sep
        # Pad right side, crop left side
        right_p = right_t.permute(2, 0, 1)
        right_p = F.pad(right_p, (0, half_sep, 0, 0), mode='replicate')
        right_shifted = right_p[:, :, half_sep:half_sep + w].permute(1, 2, 0)
        
    elif half_sep < 0:
        abs_shift = abs(half_sep)
        
        # Shift left image LEFT
        left_p = left_t.permute(2, 0, 1)
        left_p = F.pad(left_p, (0, abs_shift, 0, 0), mode='replicate')
        left_shifted = left_p[:, :, abs_shift:abs_shift + w].permute(1, 2, 0)
        
        # Shift right image RIGHT
        right_p = right_t.permute(2, 0, 1)
        right_p = F.pad(right_p, (abs_shift, 0, 0, 0), mode='replicate')
        right_shifted = right_p[:, :, :w].permute(1, 2, 0)
    else:
        left_shifted = left_t
        right_shifted = right_t
    
    # Create anaglyph
    # BGR format: channel 0=Blue, 1=Green, 2=Red
    out = torch.zeros_like(left_shifted)
    
    if mode == "Color":
        # Red from left, Green+Blue from right
        out[:, :, 2] = left_shifted[:, :, 2]   # Red
        out[:, :, 1] = right_shifted[:, :, 1]  # Green
        out[:, :, 0] = right_shifted[:, :, 0]  # Blue
        
    elif mode == "Half-color":
        # Grayscale left for red channel, color right for green+blue
        # ITU-R BT.601 weights: 0.299*R + 0.587*G + 0.114*B
        gray_left = (0.299 * left_shifted[:, :, 2] + 
                     0.587 * left_shifted[:, :, 1] + 
                     0.114 * left_shifted[:, :, 0])
        out[:, :, 2] = gray_left
        out[:, :, 1] = right_shifted[:, :, 1]
        out[:, :, 0] = right_shifted[:, :, 0]
        
    else:  # Gray
        gray_left = (0.299 * left_shifted[:, :, 2] + 
                     0.587 * left_shifted[:, :, 1] + 
                     0.114 * left_shifted[:, :, 0])
        gray_right = (0.299 * right_shifted[:, :, 2] + 
                      0.587 * right_shifted[:, :, 1] + 
                      0.114 * right_shifted[:, :, 0])
        out[:, :, 2] = gray_left
        out[:, :, 1] = gray_right
        out[:, :, 0] = gray_right
    
    # Convert back to numpy uint8
    out = out.clamp(0, 255).byte().cpu().numpy()
    return out

# -----------------------
# Anaglyph Creation (CPU fallback)
# -----------------------

def make_anaglyph_cpu(left: np.ndarray, right: np.ndarray, focus_px: int, mode: str) -> np.ndarray:
    """Create anaglyph using CPU (OpenCV)"""
    h, w = left.shape[:2]
    
    if right.shape[:2] != (h, w):
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
    else:  # Gray
        grayL = cv2.cvtColor(left_s, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(right_s, cv2.COLOR_BGR2GRAY)
        out = cv2.merge([grayR, grayR, grayL])
    
    return out

# -----------------------
# Batch Processing (GPU optimized)
# -----------------------

def process_batch_torch(frames: list, color_mode: str, device: torch.device) -> list:
    """
    Process a batch of SBS frames into anaglyphs using PyTorch.
    
    Args:
        frames: List of SBS frames (BGR, uint8)
        color_mode: "Color", "Half-color", or "Gray"
        device: torch device
    
    Returns:
        List of anaglyph frames
    """
    results = []
    
    for frame in frames:
        left, right = split_full_sbs(frame)
        
        # Estimate focus
        gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        focus_px = estimate_shift_crosscorr(gray_l, gray_r)
        
        # Create anaglyph
        anaglyph = make_anaglyph_torch(left, right, focus_px, color_mode, device)
        results.append(anaglyph)
    
    return results

# -----------------------
# FPS Selection
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
# Main Processing Pipeline
# -----------------------

def process_video(input_path: str, output_path: str, color_mode: str, 
                  fps_method: str, custom_fps: Optional[float],
                  use_gpu: bool = True, use_nvdec: bool = False, 
                  use_nvenc: bool = False, batch_size: int = 1,
                  queue_size: int = 32):
    """
    Main video processing pipeline.
    
    Uses a 3-stage threaded design:
    - Stage 1: Decode frames
    - Stage 2: Process frames (GPU or CPU)
    - Stage 3: Encode frames
    """
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("ERROR: Cannot open input:", input_path)
        return 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps_native = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    export_fps = choose_export_fps(cap, fps_method, custom_fps)
    cap.release()
    
    # Validate SBS
    if w % 2 != 0:
        print("ERROR: Video width must be even for SBS format")
        return 1
    
    out_w = w // 2
    out_h = h
    
    print(f"\n📹 Input: {total_frames} frames, {w}x{h} @ {fps_native:.2f} FPS")
    print(f"📤 Output: {out_w}x{out_h} @ {export_fps:.2f} FPS")
    
    # Setup device
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        use_gpu = False
        print("💻 Using CPU")
    
    # Setup reader
    reader = None
    if use_nvdec:
        try:
            reader = NVDECReader(input_path, w, h, fps_native)
            print("✅ NVDEC enabled")
        except Exception as e:
            print(f"⚠️ NVDEC failed: {e}")
            reader = None
    
    if reader is None:
        reader = StandardReader(input_path)
        if use_nvdec:
            print("📖 Falling back to standard reader")
    
    # Setup writer
    writer = None
    if use_nvenc:
        try:
            writer = NVENCWriter(output_path, out_w, out_h, export_fps)
            print("✅ NVENC enabled")
        except Exception as e:
            print(f"⚠️ NVENC failed: {e}")
            writer = None
    
    if writer is None:
        writer = StandardWriter(output_path, out_w, out_h, export_fps)
        if use_nvenc:
            print("📝 Falling back to standard writer")
    
    # Queues and threading
    q_decode = queue.Queue(maxsize=queue_size)
    q_process = queue.Queue(maxsize=queue_size)
    stop_event = threading.Event()
    
    pbar = tqdm(total=total_frames, desc="Processing", unit="frame", dynamic_ncols=True)
    
    # Decode thread
    def decode_thread():
        idx = 0
        try:
            while not stop_event.is_set():
                ok, frame = reader.read()
                if not ok:
                    break
                q_decode.put((idx, frame))
                idx += 1
        except Exception as e:
            print(f"\n❌ Decode error: {e}")
        finally:
            q_decode.put(None)
    
    # Process thread
    def process_thread():
        try:
            while True:
                item = q_decode.get()
                if item is None:
                    q_decode.put(None)  # Propagate sentinel
                    break
                
                idx, frame = item
                
                # Split SBS
                left, right = split_full_sbs(frame)
                
                # Estimate focus
                gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
                focus_px = estimate_shift_crosscorr(gray_l, gray_r)
                
                # Create anaglyph
                if use_gpu:
                    anaglyph = make_anaglyph_torch(left, right, focus_px, color_mode, device)
                else:
                    anaglyph = make_anaglyph_cpu(left, right, focus_px, color_mode)
                
                q_process.put((idx, anaglyph))
                
        except Exception as e:
            print(f"\n❌ Process error: {e}")
        finally:
            q_process.put(None)
    
    # Encode thread
    def encode_thread():
        next_idx = 0
        buffer = {}
        num_workers = max(1, multiprocessing.cpu_count() // 2)
        workers_done = 0
        
        while True:
            item = q_process.get()
            if item is None:
                workers_done += 1
                if workers_done >= num_workers:
                    break
                continue
            
            idx, frame = item
            buffer[idx] = frame
            
            # Write frames in order
            while next_idx in buffer:
                writer.write(buffer.pop(next_idx))
                next_idx += 1
                pbar.update(1)
    
    # Start threads
    th_decode = threading.Thread(target=decode_thread, daemon=True)
    
    num_workers = max(1, multiprocessing.cpu_count() // 2)
    th_workers = [threading.Thread(target=process_thread, daemon=True) for _ in range(num_workers)]
    
    th_encode = threading.Thread(target=encode_thread, daemon=True)
    
    t_start = time.time()
    
    th_decode.start()
    for th in th_workers:
        th.start()
    th_encode.start()
    
    # Wait for completion
    try:
        while th_encode.is_alive():
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted!")
        stop_event.set()
    finally:
        stop_event.set()
        th_decode.join(timeout=2)
        for th in th_workers:
            th.join(timeout=2)
        th_encode.join(timeout=2)
        reader.release()
        writer.release()
        pbar.close()
    
    elapsed = time.time() - t_start
    fps_actual = total_frames / elapsed if elapsed > 0 else 0
    
    print(f"\n✅ Done!")
    print(f"   Frames: {total_frames}")
    print(f"   Time: {elapsed:.1f}s")
    print(f"   Speed: {fps_actual:.2f} FPS")
    print(f"   Output: {output_path}")
    
    return 0

# -----------------------
# User Interface
# -----------------------

def prompt_choices():
    """Interactive prompts for user configuration"""
    
    print("\n" + "=" * 50)
    print("  SBS → Red/Cyan Anaglyph Converter")
    print("  Version 18.1.2026-1")
    print("=" * 50)
    
    # Input path
    while True:
        in_path = input("\n📁 Input SBS video path: ").strip().strip('"').strip("'")
        if os.path.isfile(in_path):
            break
        print("   ❌ File not found!")
    
    # Output path
    while True:
        out_path = input("📁 Output video path: ").strip().strip('"').strip("'")
        if out_path:
            if not out_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                out_path += '.mp4'
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            break
        print("   ❌ Cannot be empty!")
    
    # FPS method
    print("\n📊 FPS method:")
    print("   1) Custom FPS")
    print("   2) Frames/Duration based")
    print("   3) OpenCV native")
    
    fps_choice = input("   Choose [1/2/3] (default: 3): ").strip() or "3"
    
    if fps_choice == "1":
        fps_method = "custom"
        try:
            custom_fps = float(input("   Enter FPS (≥1): ").strip())
            custom_fps = max(1.0, custom_fps)
        except:
            custom_fps = 30.0
    elif fps_choice == "2":
        fps_method = "frames_duration"
        custom_fps = None
    else:
        fps_method = "opencv_native"
        custom_fps = None
    
    # Color mode
    print("\n🎨 Color mode:")
    print("   1) Color (full color anaglyph)")
    print("   2) Half-color (gray left, color right)")
    print("   3) Gray (monochrome)")
    
    cm_choice = input("   Choose [1/2/3] (default: 1): ").strip() or "1"
    color_mode = "Color" if cm_choice == "1" else "Half-color" if cm_choice == "2" else "Gray"
    
    # GPU detection
    has_cuda, gpu_name = check_torch_cuda()
    has_nvenc = check_nvenc_support()
    has_nvdec = check_nvdec_support()
    
    print("\n🔧 Hardware detection:")
    if has_cuda:
        print(f"   ✅ PyTorch CUDA: {gpu_name}")
    else:
        print("   ❌ PyTorch CUDA: Not available")
    print(f"   {'✅' if has_nvdec else '❌'} FFmpeg NVDEC (HW decode)")
    print(f"   {'✅' if has_nvenc else '❌'} FFmpeg NVENC (HW encode)")
    
    # Processing device
    use_gpu = False
    use_nvdec = False
    use_nvenc = False
    
    if has_cuda:
        print("\n⚡ Processing device:")
        print("   1) GPU (PyTorch CUDA)")
        print("   2) CPU only")
        
        dev_choice = input("   Choose [1/2] (default: 1): ").strip() or "1"
        use_gpu = (dev_choice == "1")
        
        if use_gpu:
            if has_nvdec:
                nvdec_choice = input("\n   Use NVDEC for video decode? [Y/n]: ").strip().lower()
                use_nvdec = nvdec_choice != 'n'
            
            if has_nvenc:
                nvenc_choice = input("   Use NVENC for video encode? [Y/n]: ").strip().lower()
                use_nvenc = nvenc_choice != 'n'
    else:
        print("\n💻 Using CPU (no CUDA available)")
    
    return {
        'input_path': in_path,
        'output_path': out_path,
        'color_mode': color_mode,
        'fps_method': fps_method,
        'custom_fps': custom_fps,
        'use_gpu': use_gpu,
        'use_nvdec': use_nvdec,
        'use_nvenc': use_nvenc,
    }

# -----------------------
# Main Entry Point
# -----------------------

def main():
    """Main entry point"""
    
    # Set OpenCV threads
    try:
        cv2.setNumThreads(cv2.getNumberOfCPUs())
    except:
        pass
    
    # Get user configuration
    config = prompt_choices()
    
    print("\n" + "=" * 50)
    print("  Starting conversion...")
    print("=" * 50)
    
    # Process video
    ret = process_video(
        input_path=config['input_path'],
        output_path=config['output_path'],
        color_mode=config['color_mode'],
        fps_method=config['fps_method'],
        custom_fps=config['custom_fps'],
        use_gpu=config['use_gpu'],
        use_nvdec=config['use_nvdec'],
        use_nvenc=config['use_nvenc'],
    )
    
    sys.exit(ret)

if __name__ == "__main__":
    main()