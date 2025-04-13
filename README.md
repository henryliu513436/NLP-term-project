# LLM Classification Finetuning

<div align="center">

![NLP](https://img.shields.io/badge/NLP-Classification-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-orange)

</div>

> A project focused on classifying Large Language Model (LLM) responses and predicting human preferences based on a Kaggle competition, combining multiple modeling approaches and feature engineering techniques.

## 📝 Project Overview

This project focuses on the Kaggle competition task of "Improving Large Language Model (LLM) Response Classification." The goal is to predict which response a user would prefer when presented with responses from two anonymous models. We combined various approaches including DebertaV3, XGBClassifier, and a Weighted Average Method, enhanced with rich feature engineering techniques, successfully elevating our model performance to the top ranks of the competition.

### 📺 Project Demonstration

[Watch Project Introduction Video](https://www.youtube.com/watch?v=1SF3cSyJm-w)

## 🎯 Project Objectives

This repository contains our implementation for predicting human preferences between pairs of LLM responses, as part of the "Improving LLM Response Classification" competition on Kaggle. Our goal was to analyze interaction data from Chatbot Arena, where users were presented with two anonymous model responses and asked to choose their preferred one.
We developed multiple approaches including DebertaV3, XGBClassifier, and a weighted average ensemble method combining LGBMClassifier with pre-trained LLMs (Gamma2-9b and Llama3-8b). Our best model achieved a competitive score of 0.83084, placing it near the top of the competition leaderboard.

## 📊 Data Analysis

This project uses interaction data from Chatbot Arena, including:

- **Training Data**: Approximately 55,000 interaction records
- **Testing Data**: Approximately 25,000 interaction records
- **Data Structure**: Each record includes prompts, responses from two models, and user selections


## 🔧 Implementation Methods

### 1. DebertaV3

- **Model Structure**: Using Microsoft's DeBERTaV3 pre-trained model, based on the improved Transformer architecture
- **Training Strategy**: Three-stage fine-tuning approach:
  - Stage 1: Freeze backbone layers, train only the classification head (learning rate 1e-3)
  - Stage 2: Unfreeze the last two backbone layers (learning rate 5e-6)
  - Stage 3: Completely unfreeze all backbone layers (learning rate 1e-6)
- **Feature Engineering**: Combined prompts and responses, set sequence length to 512, performed standardized tokenization and encoding processing

### 2. XGBClassifier

- **Model Selection**: Using XGBClassifier from XGBoost, based on the Gradient Boosting Tree algorithm
- **Parameter Tuning**: Employed RandomizedSearchCV to search for optimal hyperparameters, including tree depth, number, learning rate, etc.
- **Feature Extraction**: Used self-trained Word2Vec model to convert words into vector representations (vector dimension 300, window size 5)

### 3. Weighted Average Method

- **Model Ensemble**: Combining LGBMClassifier with pre-trained Gamma2-9b and Llama3-8b models
- **Training Process**: Used k-fold cross validation to train LGBMClassifier (10 folds)
- **Feature Engineering**: Used word-level and character-level TF-IDF and CountVectorizer to extract text features
- **Weight Distribution**: Based on each model's performance, the final weight ratio was Gamma2-9b:Llama3-8b:LGBMClassifier = 50:42:0.5

## 📈 Experimental Results

### DebertaV3
- Training accuracy reached approximately 48%
- Validation accuracy reached approximately 47%
- Leaderboard score: 1.02899

### XGBClassifier
- Average Log Loss: 1.0351
- OOF Log Loss: 1.0351
- Leaderboard score: 1.02907

### Weighted Average Method
- First stage (LGBMClassifier only): Leaderboard score 1.01347
- Second stage (combined with LLMs): Leaderboard score 0.83084, ranked second

### Key Findings

1. **Model Performance and Preferences**: Pre-trained LLMs can capture more complex semantic relationships
2. **Feature Impact Analysis**: Response length, vocabulary richness, and entity count significantly influence prediction results
3. **Weight Distribution**: In the weighted average method, deep LLM models (Gamma2-9b and Llama3-8b) contributed more than shallow models

## 🔮 Future Directions

1. **Feature Engineering Optimization**:
   - Add Sentence Embeddings to obtain sentence-level semantic features
   - Consider semantic similarity metrics in feature extraction
   - Increase sentiment analysis-related features

2. **Model and Training Improvements**:
   - Try padding all responses to the same length to reduce length bias
   - Use larger LLMs if hardware performance allows
   - Design more targeted data augmentation strategies

3. **Bias Handling**:
   - Design specialized mechanisms for handling length and position biases
   - Explore methods like swapping response positions to reduce the impact of specific biases


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
