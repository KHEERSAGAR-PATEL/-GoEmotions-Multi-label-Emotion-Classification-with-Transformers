# -GoEmotions-Multi-label-Emotion-Classification-with-Transformers
This project builds a custom Transformer from scratch to classify text into one or more of 28 human emotions using the **GoEmotions** dataset from Google Research. It handles multi-label classification with threshold optimization, custom callbacks, and real-world predictions. Built for **Applied Scientist roles**, with strong evaluation metrics and interpretable outputs.

---

### 🚀 Features

- 🔤 **Multi-label emotion classification** on human conversations
- 🤗 **custom Transformer from scratch** custom Transformer from scratch

- ### 🔧 Vanilla Transformer Architecture (From Scratch)

This model is a clean reimplementation of the original "Attention is All You Need" Transformer using TensorFlow/Keras. Here's a breakdown of the components:

#### 📐 Positional Encoding
Since transformers lack recurrence, we inject positional information using sinusoidal encodings to give the model a sense of word order.

#### 🔁 Multi-Head Self-Attention
The model captures different types of relationships between tokens using 8 parallel attention heads. This enables nuanced understanding of emotional cues from various perspectives.

#### 🧱 Transformer Block
Each block includes:
- Multi-head attention
- Add & Norm
- Feed Forward Network (2-layer MLP)
- Dropout for regularization

Stacking multiple blocks improves depth and learning capacity.

#### 📤 Output Projection
The final encoder output is mean-pooled and passed through a dense layer with a sigmoid activation for multi-label classification across 28 emotion categories (including "neutral").

#### 🛠 Custom Training Details
- Binary cross-entropy loss
- Macro F1 evaluation during training
- Threshold tuning for multi-label decision boundaries

- 📈 Evaluation using **F1-score**, **AUC**, and class-wise metrics
- 🧠 Handles **28 emotions** + "neutral" from real Reddit data
- 📊 Clean visualizations & structured output for inference use-cases

---

### 🧬 Dataset: GoEmotions

- Source: [GoEmotions (Google Research)](https://github.com/google-research/goemotions)
- Labels: 27 fine-grained emotions + neutral (total 28)
- Format: Multi-label (each sample can have multiple emotions)
- Real Reddit comments for rich emotional diversity

---



### 📊 Model Training & Evaluation

We trained our scratch-built Transformer model for up to 20 epochs with early stopping and validation monitoring. A custom callback continuously tracked the optimal threshold for binary prediction (initialized at 0.10). Below are some key highlights:

#### 🧠 Training Logs
- Training stabilized after ~10 epochs
- Macro F1 gradually improved, peaking at `0.2391`
- Micro AUC rose to `0.8553`, indicating good class separation
- Training Time: `722.98 seconds`

#### 📌 Final Evaluation on Test Set
- 🏁 **Best Threshold**: `0.10`
- 🎯 **Macro F1-score**: `0.2382`
- 🎯 **Micro AUC-score**: `0.8553`

These metrics reflect the challenge of multi-label emotion classification on imbalanced data, where nuanced expressions are harder to capture with high recall.



### 📝 Example Inference

```python
📝 Text: "I'm honestly so proud of how far I’ve come."
🔍 Predicted Emotions: ['pride', 'joy']

📝 Text: "I really don't care anymore."
🔍 Predicted Emotions: ['disappointment', 'disapproval']

📝 Text: "It was okay, I guess."
🔍 Predicted Emotions: ['neutral']
```

---


### 💡 Future Work

- ✅ Streamlit-based inference UI
- 🔍 Attention visualization on input tokens
- 📚 Explainability with LIME/SHAP
- 🧪 Zero-shot emotion detection extension

---

### 🤝 Contributing

PRs welcome! For feature requests or feedback, please open an issue.

---

### 🧑‍🔬 Author

Built with ❤️ for M.Tech Applied Scientist and Machine Learning roles by Kheer Sagar Patel  
M.Tech in AI & ML @ IIITDM Jabalpur

---
