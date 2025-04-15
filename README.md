
# AI Driven Context Detection for Surveillance

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

---

## WorkFlow

![image](https://github.com/user-attachments/assets/dcdfc7b9-5b64-4895-8d71-040e693b22a3)

---

##  Model Accuracy

- Model Accuracy is 92.3%

![image](https://github.com/user-attachments/assets/c0a226a9-ec6c-4186-a359-0e0a14188d09)

  
---

##  System Architecture

- Input: 10s audio-video stream (any resolution or noise level).
- Divided into **5 chunks of 2s** (our context window).
- Model predicts **activity per 2s chunk** → array of 5 elements.
- Array passed to LLM → summarized context.
- Displayed on **Streamlit-based UI**, and **logged to file**.
- **Alerts & video capture** activated on surveillance-critical detection.

---

##  Output UI (Streamlit)

>![image](https://github.com/user-attachments/assets/30e47f79-c99e-44e7-9181-4d6a0d574110)

---

## Folder Structure

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

##  How to Run the Project

###  Clone the Repository

```bash
git clone https://github.com/syed-azim-git/AI_Indoor_Surveillance.git
cd AI_Indoor_Surveillance
```

---

###  Cloud Setup with JarvisLabs

We use [JarvisLabs](https://jarvislabs.ai/) to run GPU computations on the cloud (e.g., RTX5000).

####  Generate SSH Key

On your local machine:

```bash
ssh-keygen -t rsa -b 4096 -C "samplemail@.email.com"
```

- When prompted for **file location**, press `Enter`
- When prompted for **passphrase**, press `Enter`

Output:
```
Your public key has been saved in /c/Users/Comp/.ssh/id_rsa.pub
```

Copy the contents of the `id_rsa.pub` file and add it to your **JarvisLabs SSH agent** via their dashboard.

---

####  Launch and Start Your Cloud Instance

- Create and start a **cloud instance** with an **RTX5000 GPU**
- Note down the **SSH port number** (example: `11114`)

---

####  Upload Required Files

All files under the `cloud/` directory of this repo must be copied to the cloud.

**Tip:**  To upload files to the cloud use:
- Jupyter Notebook in Jarvis Lab (or)
- FTP for SSH, eg. FireZilla, Username: `root`

---

###  Local Setup

####  Save Your Videos

Save all recorded videos into a directory named: Videos/


####  Extract Video Features

Run the feature extraction scripts from the repo on your local machine.  
Save the resulting feature files in a directory named: Features/


---

###  Update `app.py`

In the `app.py` file, make the following changes:

- Update the paths to:
  - `Videos/` directory
  - `Features/` directory
- Replace the placeholder values in the `scp` and `ssh` commands with:
  - Your SSH port (e.g., `11114`)
  - Your `.h5` file paths

---

###  Run the App

On your local machine:

```bash
streamlit run app.py
```

---

##  Folder Structure

```
AI_Indoor_Surveillance/
│
├── cloud/                    # Files to be uploaded to JarvisLabs
├── Videos/                   # Store recorded input videos here (locally)
├── Features/                 # Store extracted .h5 feature files here (locally)
├── app.py                    # Streamlit UI app
├── feature_extraction/       # Local scripts for video feature extraction
├── setup_ai_surveillance.sh  # Setup script (optional)
└── README.md                 # You're here!
```

##  Author

[Syed Azim](https://www.linkedin.com/in/i-syed-azim/) | [Shreyas Sai](https://www.linkedin.com/in/shreyassai/)   
Final Year, ECE  
SSN College of Engineering  

---

##  Future Scope

- Hardware Implementation

---

## Acknowledgements

This project was developed as part of the Final Year Engineering Project by:

**Shreyas Sai R** and **Syed Azim**  
**Mentored by:** *Dr. P. Vijayalakshmi*  
Department of Electronics and Communication Engineering  
SSN College of Engineering
