
# AI Driven Context Detection for Comprehensive Surveillance

![Project Badge](https://img.shields.io/badge/Final_Year_Project_May,_2025-blueviolet)  

*By: Shreyas Sai R and Syed Azim  – SSN College of Engineering* 

---

##  Problem Statement

Traditional video-based surveillance systems face several limitations:

- **Privacy Concerns** – Continuous video recording can infringe on personal privacy.
- **Storage Overhead** – Uncompressed video data demands large storage and bandwidth.
- **Manual Monitoring** – Requires human supervision, no real time alerts and prone to fatigue and error.

There is an need for AI-driven, multimodal surveillance that goes beyond passive video feeds. Our system aims to detect, analyze, and summarize events, optimizing surveillance while respecting privacy.

---

##  Proposed Solution

###  Edge-Level Feature Extraction

- Captures **video + audio** from edge devices.
- Extracts **lightweight, non-reversible features** (not raw media).
- **Instantly discards video**, ensuring **privacy & minimal storage**.
- Sends compact feature vectors to the cloud.

###  Cloud-Based Multimodal Fusion

- Combines **audio + visual** features using a **fusion model**.
- Employs **attention mechanisms** to handle dominance of one modality.
- Works even in **low-light or noisy** conditions for **robust detection**.

###  Context Detection with LLM

- Uses **LLMs (LLaMA 3.1b)** to generate **semantic context summaries**.
- Outputs phrases like:  
  _“Suspicious activity near the entrance”_  
- Promotes **real-time awareness**, reducing operator fatigue.
- Enables **alerting without revealing private visuals**.

---

##  System Output

- Final output: a **text file** describing the **context of indoor activities**.
- If a **critical surveillance activity** is detected:
  - An **alert** is sent.
  - Related **video is live-streamed and saved** for future use.

---

##  Technical Overview

| Component | Description |
|----------|-------------|
| **Baseline Model** | Inspired by Audio-Visual Event Segmentation (Tian et al.) |
| **Dataset** | Filtered, curated, and augmented subset of **Kinetics-400 Tiny**, adapted for surveillance |
| **Visual Feature Extraction** | `MobileNet` |
| **Audio Feature Extraction** | `MFCC` |
| **Frame Selection** | `Fuzzy C-Means Clustering` |
| **Fusion Method** | `Feature Concatenation` |
| **Architecture** | `Unimodal Attention + Cross-Modality Attention` |
| **Final Output** | Passed into `LLaMA 3.1b` for natural-language context generation |

![image](https://github.com/user-attachments/assets/dcdfc7b9-5b64-4895-8d71-040e693b22a3)

##  Model Accuracy

- Model Accuracy is 92.3%

![image](https://github.com/user-attachments/assets/c0a226a9-ec6c-4186-a359-0e0a14188d09)

  
---

## 🏗️ System Architecture

- Input: 10s audio-video stream (any resolution or noise level).
- Divided into **5 chunks of 2s** (our context window).
- Model predicts **activity per 2s chunk** → array of 5 elements.
- Array passed to LLM → summarized context.
- Displayed on **Streamlit-based UI**, and **logged to file**.
- **Alerts & video capture** activated on surveillance-critical detection.

---

## 📸 Output UI (Streamlit)

>![image](https://github.com/user-attachments/assets/30e47f79-c99e-44e7-9181-4d6a0d574110)



---

## 📁 Folder Structure

```bash
├── models/
│   ├── feature_extractors/
│   ├── fusion_module/
│   └── llama_integration/
├── data/
│   ├── kinetics_tiny_curated/
├── app.py
├── README.md
└── requirements.txt
```

---

## 🚀 To Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/ai-context-surveillance.git
   cd ai-context-surveillance
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start Streamlit app:
   ```bash
   streamlit run streamlit_ui/app.py
   ```

---

## 🤖 Future Scope

- Hardware Implementation

---

## Acknowledgements

This project was developed as part of the Final Year Engineering Project by:

**Shreyas Sai R** and **Syed Azim**  
**Mentored by:** *Dr. P. Vijayalakshmi*  
Department of Electronics and Communication Engineering  
SSN College of Engineering
