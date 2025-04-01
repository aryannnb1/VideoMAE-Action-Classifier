# Action Recognition with VideoMAE Transformer on UCF101 Subset  
**By Aryan S. Bhardwaj**

## 📌 Project Overview  
This repository showcases a complete video action recognition pipeline using the **VideoMAE Transformer** model from Hugging Face, trained and tested on a curated subset of the **UCF101** dataset. The goal is to classify human activities based on video clips using a self-supervised, transformer-based approach.

This project includes:
- Dataset preparation and video frame extraction  
- Fine-tuning a VideoMAE transformer model  
- Performance visualization and evaluation on test/validation sets  
- Real-world testing on YouTube videos  

> **Best Model Weights:**  
[Download here](https://drive.google.com/file/d/1_l_j_iB3nXob4eXEkNMtY_48eIeLPt7o/view?usp=sharing)

---

## 🎯 Dataset: UCF101 Subset  
UCF101 is a benchmark dataset for human action recognition containing 13K+ video clips across 101 categories. This project uses a custom subset with **5 action classes**:

- 🏇 HorseRiding  
- 🏂 Skiing  
- 🏃‍♂️ LongJump  
- 🤸‍♀️ PoleVault  
- 🏹 JavelinThrow  

Each video is split into RGB frames and organized into class-wise folders for **training, validation, and testing**.

---

## 🧠 Model Architecture  
We use **VideoMAE** (Video Masked Autoencoder), a transformer-based architecture that:
- Leverages masked autoencoding for spatiotemporal representation learning  
- Is pretrained on large-scale video datasets  
- Allows fine-tuning on small action recognition tasks efficiently  

Pretrained weights from Hugging Face are used and fine-tuned on the selected dataset split.

---

## 📊 Results  

### 🔁 Training (Epoch 10)  
- **Train Loss**: `0.0039`  

### ✅ Validation Set  
- **Loss**: `0.0757`  
- **Accuracy**: `98.51%`  
- **Precision**: `98.57%`  
- **Recall**: `98.75%`  
- **F1-Score**: `98.61%`  

### 🧪 Test Set  
- **Loss**: `0.0745`  
- **Accuracy**: `97.30%`  
- **Precision**: `97.03%`  
- **Recall**: `97.03%`  
- **F1-Score**: `97.03%`  

> **Insight:** The model shows excellent generalization with balanced performance on validation and test sets. Very low train loss indicates strong convergence.

---

## 🛠️ Libraries Used  

| Library | Purpose |
|--------|---------|
| `torch` | Model training, tensor operations |
| `transformers` | VideoMAE model loading from Hugging Face |
| `torchvision` | Image transformations |
| `cv2` | Frame extraction from videos |
| `yt_dlp` | Downloading YouTube videos for real-world testing |
| `imbalanced-learn` | Resampling strategies (SMOTE/oversampling) |
| `matplotlib`, `seaborn` | Evaluation visualization |
| `scikit-learn` | Classification metrics (precision, recall, F1) |

---

## 🎥 Real-World Testing  
This project includes inference on real YouTube videos using the trained model. Frame extraction is done via `yt_dlp` + `OpenCV`, and predictions are run on fixed-length video segments.

---

## 📈 Future Improvements  
- Incorporate multi-modal learning using audio  
- Real-time classification with webcam integration  
- Deploy as a web application with Streamlit or Gradio  

---

## 📎 Best Model Download  
🔗 [Click here to download the best model checkpoint (.pt)](https://drive.google.com/file/d/1_l_j_iB3nXob4eXEkNMtY_48eIeLPt7o/view?usp=sharing)

---

## 👤 Author  
Aryan S. Bhardwaj  
_Deep Learning | Computer Vision | Transformers_  
📫 [Connect on LinkedIn]([https://www.linkedin.com](https://www.linkedin.com/in/aryanb03/)) (replace with actual link)  
