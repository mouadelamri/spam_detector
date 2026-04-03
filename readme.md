
# 🛡️ AI SMS Spam Detector
A professional Machine Learning application to classify SMS messages as **Spam** or **Ham** using Natural Language Processing (NLP).

## 🚀 Live Demo
| Spam Detection (🚨) | Ham Detection (✅) |
| :---: | :---: |
| ![Spam Example](screenshots/spam_result.png) | ![Ham Example](screenshots/ham_result.png) |

## 🧠 How it Works
This project follows a complete Data Science lifecycle:
1. **Data Cleaning:** Removing punctuation and converting text to lowercase.
2. **Vectorization:** Using **TF-IDF** (Term Frequency-Inverse Document Frequency) to convert text into numerical features.
3. **Modeling:** Training a **Multinomial Naive Bayes** classifier.
4. **Deployment:** Building an interactive UI with **Streamlit**.



## 📊 Performance Metrics
The model was trained on the **UCI SMS Spam Dataset** and achieved:
- **Accuracy:** 96.86%
- **Precision:** 1.00 (Perfect precision for spam detection).

## 🛠️ Setup & Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/spam-detector.git](https://github.com/yourusername/spam-detector.git)