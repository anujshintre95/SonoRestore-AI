# 🩺 SonoRestore AI

### AI-Powered Ultrasound Image Restoration & Super-Resolution

SonoRestore AI is a deep-learning based ultrasound image restoration prototype designed to enhance low-resolution grayscale ultrasound images.

The system uses a lightweight Convolutional Neural Network (CNN) trained for ultrasound image restoration and deployed as an ONNX model for browser-based inference.

---

## 🎯 Problem

Ultrasound imaging can be affected by:

- Low spatial resolution
- Image degradation
- Limited image detail
- Noise and artifacts

These limitations can make visual interpretation more difficult.

SonoRestore AI explores how deep learning can be used to restore and enhance degraded ultrasound images while preserving important structural information.

> **Note:** SonoRestore AI is a research/hackathon prototype and is not intended for medical diagnosis or clinical decision-making.

---

# 💡 Our Solution

SonoRestore AI takes a degraded grayscale ultrasound image and generates a higher-resolution restored image using a trained CNN.

### Pipeline

```text
Ultrasound Image
       ↓
Preprocessing
       ↓
128 × 128 Grayscale Input
       ↓
SonoRestore CNN
       ↓
256 × 256 Restored Image
       ↓
Browser Display