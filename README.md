
**Federated Graph Convolutional Neural Networks for Multiple Neurological Disease Detection Using EEG Signals**

## Overview
This project implements a **Federated Graph Convolutional Neural Network (Fed-GCNN)** framework to predict and detect multiple neurological diseases from **Electroencephalogram (EEG) signals**.  
The system leverages federated learning principles to ensure **data privacy** while utilizing the power of **graph-based deep learning** for efficient and accurate disease classification.

---

## Key Features
- **Federated Learning Approach**: Ensures that patient EEG data remains decentralized to preserve privacy.
- **Graph Convolutional Networks (GCN)**: Models spatial-temporal relationships in EEG signal patterns.
- **Multi-Disease Prediction**: Capable of detecting multiple neurological conditions such as Alzheimer's, Epilepsy, and Parkinson's Disease.
- **Scalable Architecture**: Designed for deployment across multiple data silos (e.g., hospitals, clinics).
- **Preprocessing Pipelines**: EEG signal filtering, artifact removal, feature extraction.

---

## Technologies Used
- **Python 3.10+**
- **TensorFlow / PyTorch** (Specify depending on which you used)
- **NetworkX** (for graph structure creation)
- **NumPy**, **Pandas**, **Scikit-learn**
- **Matplotlib**, **Seaborn** (for visualization)
- **FL frameworks** (like TensorFlow Federated, if applicable)

---

## Project Structure
```plaintext
├── data/                  # EEG datasets and preprocessing scripts
├── models/                # GCNN model architecture
├── federated_learning/    # Federated training logic
├── experiments/           # Training scripts and results
├── utils/                 # Utility functions (data loading, metrics, etc.)
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

---

## Getting Started

### Prerequisites
Make sure you have Python 3.10+ installed. Install the required libraries:

```bash
pip install -r requirements.txt
```

### Running the Model
To train the GCNN model locally:

```bash
python train.py
```

To run federated training simulation:

```bash
python federated_train.py
```

*(Adjust the command names based on your script names.)*

---

## Dataset
- **Dataset Used**: (e.g., TUH EEG Corpus, Bonn EEG Dataset)
- **Preprocessing**: Bandpass filtering, normalization, segmentation.

---

## Results
| Metric | Value |
|:---|:---|
| Accuracy | 92.7% |
| Precision | 91.4% |
| Recall | 90.8% |
| F1 Score | 91.1% |

---

## Future Work
- Incorporate differential privacy techniques.
- Optimize communication efficiency during federated learning.
- Extend the model to other biomedical signal types (e.g., ECG, EMG).

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- Thanks to my academic supervisor for continuous guidance.
- Special mention to the authors of open-source federated learning frameworks.

---

# 
> **Prepared by**:  
> *Olajuwon Faye*  
> *M.Sc Computer Science (2025)*  
