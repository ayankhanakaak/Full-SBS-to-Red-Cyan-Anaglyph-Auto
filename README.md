# Full SBS to Red-Cyan Anaglyph Converter

This tool converts **full-width SBS (Side-by-Side) 3D videos** into **Red/Cyan Anaglyph** videos with **automatic cross-correlation guided alignment**.  
It is optimized for speed with a **multi-threaded CPU pipeline**, and optionally supports **CUDA GPU acceleration** if available.

---

## ✨ Features
- 📼 Input: Full-width SBS 3D video  
- 🎨 Output: Red/Cyan Anaglyph (Color, Half-color, or Gray)  
- ⚡ Automatic focus alignment using cross-correlation  
- 🧵 Multi-threaded frame processing (uses all CPU cores)  
- 🔄 CUDA acceleration support (if GPU supported and OpenCV compiled with CUDA)  
- 🎚 Configurable FPS export method (Custom, Frames and Duration Based, OenCV Native)  
- 🖥 Simple command-line prompts for all options  

---

## 🚀 Usage
Run the script:
```bash
python "Full SBS to Red-Cyan Anaglyph Auto - V.05.09.2025-2.py"
```

You will be prompted for:
1. **Input SBS video path**  
2. **Output video path**  
3. **FPS export method** (custom / duration-based / native)  
4. **Color mode** (Color / Half-color / Gray)  
5. **Processing device** (CPU only / CPU + CUDA)

---

## ⚙️ Requirements
- Python 3.8+ (Recommended 3.13.7)  
- [OpenCV](https://opencv.org/) with `opencv-contrib-python`  
- [tqdm](https://pypi.org/project/tqdm/)  
- CUDA-enabled OpenCV build (optional, for GPU acceleration)

Install dependencies:
```bash
pip install opencv-python opencv-contrib-python tqdm numpy
```

---

## ⚠️ Notes
- CUDA acceleration requires a GPU supported by your installed CUDA + OpenCV build.  
- On unsupported GPUs (e.g., older NVIDIA Kepler cards), the program automatically falls back to CPU.  
- Multi-threaded CPU mode already provides significant performance improvement.

---

## 📜 License
GPL-3.0
