Trusted Multi-View Classification
---

## 🧐 About

An easy-to-run implementation using PyTorch for the paper
"
Trust EEG Epileptic Seizure Detection via Evidential Multi-view Learning
".

## 🎈 Usage

**Requirements**

+ python 3.8
+ numpy 1.23
+ pytorch 1.12
+ scikit-learn 1.2

**Running**

Extracting domain feature:
```bash
cd dataset/eeg/preprocessing/ && matlab  -nodesktop -nosplash -r preprocessing_data.m
```

Processing domain feature:
```bash
python preprocess.py
```

Training and validating:
```bash
python main.py
```
