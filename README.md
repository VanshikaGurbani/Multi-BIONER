# Biomedical Named Entity Recognition (BioNER) Using Multi-Task Learning

## 1. Overview
Biomedical Named Entity Recognition (BioNER) plays a crucial role in Natural Language Processing (NLP) by identifying key biomedical entities—such as diseases, chemicals, genes, and proteins—from scientific literature. The task is particularly challenging due to:
- The variety of entity types and overlap between domains  
- Context-dependent terminology (e.g., “cold” may refer to illness or temperature)

To address these complexities, we implemented:
1. **Single-Task Learning (STL)** – Fine-tuned BioBERT separately on each dataset using the IOBES labeling scheme.  
2. **Multi-Task Learning (MTL)** – Extended BioBERT with a shared encoder and task-specific classification heads to allow cross-dataset knowledge sharing.

The problem was modeled as a **token-level classification task**, assigning entity labels to individual tokens within sentences.

---

## 2. Datasets and Preprocessing

### 2.1. Data Sources
We used publicly available biomedical corpora:
- **BC2GM** – Gene mentions (~20k labeled tokens)  
- **BC4CHEMD** – Chemical entities (~1.1M labeled tokens)  
- **BC5CDR** – Chemical and disease mentions (~170k labeled tokens)  
- **JNLPBA** – Biomedical terms like proteins and genes (~150k labeled tokens)  
- **NCBI Disease** – Disease-specific references (~30k labeled tokens)

### 2.2. Preprocessing Steps
- Data followed the IOBES format for entity labeling.  
- We used a custom script to align labels with tokens produced by BioBERT’s word-piece tokenizer.

### 2.3. Data Notes
- All datasets were heavily **imbalanced**, with a majority of tokens labeled as 'O'.  
- True entity tags like B-GENE or I-CHEMICAL were significantly rarer.

---

## 3. Model Setup and Experiments

### 3.1. Baseline and Architecture
- **Baseline**: A simple rule-based NER model using token matching.  
- **Main Model**: BioBERT v1.1, a transformer model pre-trained on biomedical text (sourced from HuggingFace).

### 3.2. Implementation
- **STL**: BioBERT fine-tuned individually per dataset.  
- **MTL**: BioBERT adapted to share encoder layers, with individual heads for each dataset.

### 3.3. Model Design
- **Input**: Tokenized sentences and aligned IOBES labels.  
- **Output**: IOBES-tagged predictions per token.  
- **Architecture**: 12 transformer layers, 768 hidden dimensions.

### 3.4. Training Configuration
- **Epochs**: 3  
- **Learning Rate**: 5e-5  
- **Batch Size**: 8  
- **Regularization**: Weight decay = 0.01, dropout = 0.1  
- **Evaluation Metrics**: Precision, Recall, F1-score

---

## 4. Key Results

### 4.1. STL vs MTL Performance
#### BC2GM
- STL: F1 = 0.98  
- MTL: F1 = 0.96  

#### NCBI Disease
- STL: F1 = 0.99  
- MTL: F1 = 0.98  

**STL** performed well on larger datasets like BC2GM, while **MTL** demonstrated better generalization and improved accuracy on smaller datasets such as NCBI.

---

## 5. Challenges Encountered
- Aligning word-piece tokens to original IOBES tags was non-trivial.  
- Larger datasets risked overpowering smaller ones in MTL training.  
- MTL incurred higher GPU and training time costs.

---

## 6. Takeaways and Future Work

### 6.1. Conclusions
- **STL** is effective for datasets with abundant, unique information.  
- **MTL** shows strong potential for improving accuracy across varied or limited datasets by enabling shared learning.

### 6.2. Suggested Enhancements
- Incorporate **data augmentation** for minority classes  
- Explore **stronger regularization** techniques  
- Conduct thorough **hyperparameter optimization**
