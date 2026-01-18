# Full SBS to Red-Cyan Anaglyph Converter

This tool converts **full-width SBS (Side-by-Side) 3D videos** into **Red/Cyan Anaglyph** videos with **automatic cross-correlation guided alignment**.  
It is optimized for speed with a **multi-threaded CPU pipeline**, and supports **PyTorch CUDA GPU acceleration** if available.

---

## ✨ Features
- 📼 Input: Full-width SBS 3D video  
- 🎨 Output: Red/Cyan Anaglyph (Color, Half-color, or Gray)  
- ⚡ Automatic focus alignment using cross-correlation  
- 🧵 Multi-threaded frame processing (uses all CPU cores)  
- 🔄 PyTorch CUDA acceleration support (if GPU supports it)  
- 🎚 Configurable FPS export method (Custom, Frames and Duration Based, OenCV Native)  
- 🖥 Simple command-line prompts for all options
- 🎥 FFmpeg NVDEC/NVENC hardware video decode/encode

---

## 🚀 Usage
Run the script:
```bash
python "Full SBS to Red-Cyan Anaglyph Auto - V.18.1.2026-1.py"
```

Then follow on-screen prompts as per your choice.

---

## ⚠️ Notes
- On unsupported GPUs (e.g., older NVIDIA Kepler cards), the program automatically falls back to CPU.

---

## 📜 License
GPL-3.0
