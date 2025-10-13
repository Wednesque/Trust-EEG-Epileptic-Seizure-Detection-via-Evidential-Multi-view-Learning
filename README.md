Trust EEG Epileptic Seizure Detection via Evidential Multi-view Learning
---

## 🧐 About
---

An easy-to-run implementation using PyTorch for the paper
**"Trust EEG Epileptic Seizure Detection via Evidential Multi-view Learning"**  

If you have any questions, feel free to contact **zqwenn@stu.xidian.edu.cn** — happy to discuss and exchange ideas!  

If you find this work useful, please kindly cite our paper:  
```bibtex
@article{liu2025trust,
  title={Trust EEG epileptic seizure detection via evidential multi-view learning},
  author={Liu, Ying and Xu, Cai and Wen, Ziqi and Dong, Yansong},
  journal={Information Sciences},
  volume={694},
  pages={121699},
  year={2025},
  publisher={Elsevier}
}
```

## 🎈 Usage
---

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

## ⭐ Preprocessing Procedure for EML using the CHB-MIT Dataset
---

## Dataset Download
https://physionet.org/content/chbmit/1.0.0/
Place it in the \dataset\eeg\data\raw_data folder

## Data Processing Instructions

**Requirements**
- python 3.8
- numpy 1.23
- pytorch 1.12
- scikit-learn 1.2
- mne 0.23.4
- scipy 1.1.0


**Running**
- Channel Selection
```bash
python record.py
```

After running, you will obtain a document of seizure time slices from the dataset
Note: The extracted seizure time slices may include segments with different channels from other segments, you can choose according to your needs
It is recommended to directly use the pre-selected document record.txt

- Channel Extraction
```bash
python channel.py
```
Note: Select the required patient seizure segments from record.txt and set them in the segments section of channel.py
If using the provided preset record.txt, the selected channels are:
```bash
'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8-1'
```
For patient number 6:
```bash
'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'T8-P8-1', 'FC1-Ref', 'FC2-Ref', 'FC5-Ref', 'FC6-Ref'
```
Please set according to your actual patient, channels, and seizure segment selection as needed


- Feature Extraction
```bash
cd dataset/eeg/preprocessing/ && matlab  -nodesktop -nosplash -r preprocessing_data.m
```
This step extracts features, resulting in the corresponding MATLAB files

- Obtain Multiview Data
```bash
python preprocess.py
```
After running, you will obtain train.pkl and valid.pkl
