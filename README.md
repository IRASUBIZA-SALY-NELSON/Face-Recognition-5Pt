# Face Recognition with ArcFace ONNX & 5-Point Alignment

**Student Implementation based on the Guide by Gabriel Baziramwabo**

This project builds a robust, explainable, and CPU-friendly face recognition system from scratch. It moves beyond "black box" APIs to implement a transparent pipeline: Face Detection, 5-Point Landmark Localization, Geometric Alignment, and Deep Feature Extraction using ArcFace.

---

## 🌟 Acknowledgements

I would like to express my deepest gratitude to my teacher and mentor, **Gabriel Baziramwabo** (Benax Technologies Ltd · Rwanda Coding Academy).

Thank you for:
*   Demystifying complex computer vision concepts.
*   Providing a clear, step-by-step roadmap ("The Book") that made building this system possible.
*   Guiding us to understand *why* things work, not just *how* to run them.
*   Pushing for stability, explainability, and practical engineering over simple "magic" scripts.

This project is a testament to your excellent teaching. Thank you for empowering us to build real-world AI systems!

---

## 🚀 Project Overview

The system operates in two main modes:
1.  **Enrollment**: capturing faces, aligning them, and storing their mathematical embeddings.
2.  **Recognition**: Real-time identification by comparing live faces against the enrolled database.

**Key Technology Stack:**
*   **Language**: Python 3.11+
*   **Face Detection**: Haar Cascades (OpenCV)
*   **Landmarks**: MediaPipe FaceMesh (Google)
*   **Alignment**: Similarity Transform (Affine)
*   **Embedding**: ArcFace (ResNet50) via ONNX Runtime (CPU)

---

## 🛠️ Installation & Setup

### 1. Environment Setup
```bash
# Create a virtual environment
python -m venv mp_env3
source mp_env3/bin/activate  # or mp_env3\Scripts\activate on Windows

# Install dependencies
pip install opencv-python numpy onnxruntime scipy tqdm mediapipe
```

### 2. Initialize Project Structure
We used the provided script to generate the canonical folder structure:
```bash
python init_project.py
```

### 3. Model Setup
The ArcFace model is required for embedding extraction.
*   Downloaded `arcface.onnx` (ResNet50) from HuggingFace/Model Zoo.
*   Placed it at: `models/embedder_arcface.onnx`

---

## 📸 How to Run

### Phase 1: Pipeline Validation
We validated each stage independently to ensure robustness.

*   **Camera Test**: `python -m src.camera`
*   **Detection**: `python -m src.detect` (Red box around face)
*   **Landmarks**: `python -m src.landmarks` (5 green points on eyes, nose, mouth)
*   **Alignment**: `python -m src.align` (Shows the "warped" 112x112 face)
*   **Embedding**: `python -m src.embed` (Outputs the 512-D vector)

### Phase 2: Enrollment (The Database)
This step creates the identity database.
```bash
python -m src.enroll
```
1.  Enter the person's name (e.g., `Nelson`).
2.  Press **SPACE** to capture samples (capture at least 10 varied angles).
3.  Press **`s`** to **SAVE** the database.
4.  Press **`q`** to quit.

### Phase 3: Recognition (Live)
Run the real-time recognizer:
```bash
python -m src.recognize
```
*   **Green**: Recognized identity (matches database).
*   **Red**: Unknown.
*   **Controls**:
    *   `+` / `-`: Adjust sensitivity threshold.
    *   `r`: Reload database.
    *   `d`: Toggle debug info.

### Phase 4: Evaluation
To mathematically tune the threshold (requires at least 2 enrolled people):
```bash
python -m src.evaluate
```

---

## 🧠 Lessons Learned

Following this guide taught me:
1.  **Alignment is Critical**: Raw faces vary too much. Warping them to a canonical 112x112 pose drastically improves accuracy.
2.  **Embeddings over Images**: We don't save photos; we save 512-dimensional vectors. Recognition is just measuring the angle (cosine similarity) between these vectors.
3.  **Modular Design**: Building `detect.py` -> `align.py` -> `embed.py` separately made debugging easy. When recognition failed, I knew exactly which module to check.

---

*"You do not merely run face recognition—you build it, understand it, and control it."*
# Face-Recognition-5Pt
