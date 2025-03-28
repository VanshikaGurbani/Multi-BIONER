
# **Multi-Task Learning for Biomedical Named Entity Recognition (BioNER)**


## **1. Introduction**  
Biomedical Named Entity Recognition (BioNER) is a vital task in Natural Language Processing (NLP), aiming to identify biomedical entities such as genes, proteins, chemicals, and diseases in scientific text. This problem is challenging due to:  
- The **diversity of entity types** and **overlapping domains**.  
- **Context-specific terminology**, e.g., recognizing "cold" as a disease vs. temperature.  

To address these challenges, we implemented:  
1. **Single-Task Learning (STL)**: Fine-tuning BioBERT on individual datasets to predict entity labels in the IOBES schema.  
2. **Multi-Task Learning (MTL)**: Extending BioBERT with a shared encoder and task-specific heads for knowledge sharing across datasets.  

The problem was modeled as a **sequence classification task**, where each token in a sentence is assigned a label.  

---

## **2. Data Collection and Preprocessing**  

### **2.1. Datasets**  
We utilized publicly available BioNER corpora:  
- **BC2GM**: Gene mentions (~20,000 labeled tokens)  
- **BC4CHEMD**: Chemical entities (~1.1 million labeled tokens)  
- **BC5CDR**: Chemical and disease entities (~170,000 labeled tokens)  
- **JNLPBA**: Biomedical entities such as genes and proteins (~150,000 labeled tokens)  
- **NCBI Disease**: Disease-specific mentions (~30,000 labeled tokens)  

### **2.2. Preprocessing**  
- Data in IOBES tagging format was cleaned and tokenized.  
- Tokens were aligned with labels using a custom preprocessing function to support word-piece tokenization for BioBERT.  

### **2.3. Data Characteristics**  
- Datasets were **imbalanced**, with most tokens labeled as 'O' (non-entity).  
- Entity labels such as B-GENE, I-CHEMICAL, or B-DISEASE were sparse.  

---

## **3. Experiments and Models**  

### **3.1. Baseline and Architectures**  
- **Baseline**: A rule-based NER system with token-matching.  
- **Main Model**: BioBERT (v1.1) pre-trained on biomedical corpora from the HuggingFace Model Hub.  

### **3.2. Model Implementation**  
- **Single-Task Learning (STL)**: Fine-tuned BioBERT on individual datasets.  
- **Multi-Task Learning (MTL)**: Extended BioBERT with shared weights and dataset-specific output heads.  

### **3.3. Model Details**  
- **Inputs**: Tokenized sentences with aligned IOBES labels.  
- **Outputs**: Predicted IOBES labels.  
- **Architecture**: 12 transformer layers with 768 hidden units each.  

### **3.4. Training Details**  
- **Fine-Tuning**: 3 epochs, learning rate of 5e-5, and batch size of 8 per GPU.  
- **Regularization**: Weight decay (0.01) and dropout (0.1).  
- **Evaluation**: Precision, Recall, and F1-scores.  

---

## **4. Results**  

### **4.1. Performance Comparison**  
#### **BC2GM Dataset**  
- STL: F1-Score = 0.98  
- MTL: F1-Score = 0.96  

#### **NCBI Disease Dataset**  
- STL: F1-Score = 0.99  
- MTL: F1-Score = 0.98  

The **MTL model** outperformed STL on smaller datasets like NCBI Disease due to shared learning, while STL achieved competitive results on larger datasets like BC2GM.  

---

## **5. Challenges**  
1. **Token Alignment**: Aligning word-piece tokenized outputs with IOBES labels required detailed preprocessing.  
2. **Task Imbalance**: Smaller datasets were overshadowed by larger datasets in MTL.  
3. **Computational Costs**: MTL required significant GPU resources and training time.  

---

## **6. Conclusions**  

### **6.1. Observations**  
- **STL** models excel in datasets with sufficient data and unique traits.  
- **MTL** models leverage shared learning, benefiting smaller datasets and improving overall performance metrics.  

### **6.2. Future Improvements**  
- **Data Augmentation**: Synthetic data generation for diversity.  
- **Advanced Regularization**: Enhanced techniques to prevent overfitting.  
- **Hyperparameter Tuning**: Optimized learning rates and batch sizes.  

