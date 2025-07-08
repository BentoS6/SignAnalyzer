# ✋ Sign Analyzer: Static Hand Gesture Recognition

**Sign Analyzer** is a Python-based tool that detects and classifies **static hand gestures** using **MediaPipe** and a **Random Forest classifier**. It allows users to collect gesture data, train machine learning models, and perform real-time gesture prediction via webcam — laying the foundation for scalable sign language interpretation systems.

## 📦 Features

- Real-time webcam-based gesture detection using MediaPipe  
- Landmark extraction and labeled dataset generation  
- Model training using `RandomForestClassifier`  
- Accuracy evaluation and real-time prediction  
- Save/load model functionality via `pickle`  
- Modular codebase with YAML-based config  

## 📁 Project Structure

```text
sign-analyzer/
├── data/
│   └── gestures.pkl             # Saved training data
├── models/
│   └── rf_model.pkl             # Trained ML model
├── config/
│   └── config.yaml              # Settings and hyperparameters
├── images/                      # Optional visual assets or sample frames
├── src/
│   ├── collect_data.py          # Capture and label gesture data
│   ├── train_model.py           # Train and evaluate the model
│   └── predict.py               # Real-time prediction script
├── requirements.txt
└── README.md
```

## 🔧 Installation

1. **Clone the repository**
```
git clone https://github.com/your-username/sign-analyzer.git
cd sign-analyzer
``` 
2. **Create a virtual environment**
```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

3. **Install the dependencies**
```
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Run this to capture hand gesture data and label them using keystrokes:
```
python src/collect_data.py
```

### 2. Train the model using the collected gestures:
```
python src/train_model.py
```

- Splits data into train/test sets
- Outputs accuracy
- Saves model to models/rf_model.pkl

### 3. Run the realtime guesture prediction script
```
python src/predict.py
```
---
#### Pro tip: Make sure that the config file is correct 