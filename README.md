## Task Categorization:https video://drive.google.com/file/d/1v8Bs0YeQ_7q8F239sdFZFWc64exFGz3L/view?usp=drive_link
## Sentiment Analysis video:https://drive.google.com/file/d/1Fw6rjDVODtUrBbCUSZAZS2Jw6xzVxomS/view?usp=drive_link
"""
## Task Catagorization
Task Categorization is a project that classifies tasks into predefined categories based on their descriptions. It utilizes machine learning techniques to automate task classification, improving efficiency and accuracy.

## Features
- Automatic classification of tasks into categories
- Preprocessing of textual task descriptions
- Model training and evaluation
## Installation of Task Catagorization app
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```
# IMDB Sentiment Analysis Dataset

The dataset used in this project is the **IMDB Movie Reviews Dataset**, which contains **50,000 movie reviews** labeled as **positive (pos) or negative (neg)**. It is commonly used for sentiment analysis tasks.

## 📌 Dataset Details:
- **Source:** Stanford AI Lab
- **Size:** 50,000 reviews (25,000 for training, 25,000 for testing)
- **Labels:** 
  - **pos** → Positive Sentiment
  - **neg** → Negative Sentiment
- **Format:** Text files organized in "pos" and "neg" subdirectories

## 📥 Download Dataset:
You can download the dataset from the following link:

🔗 [IMDB Dataset from Stanford](https://ai.stanford.edu/~amaas/data/sentiment/)

## 📂 Dataset Structure:

In this project, we opted for **TF-IDF** (Term Frequency-Inverse Document Frequency) over the traditional **Bag-of-Words (BoW)** approach for several important reasons:

1. **Word Importance**  
   - **BoW:** Simply counts the frequency of each word in the document without any weighting.
   - **TF-IDF:** Weighs each word by its importance—words that occur frequently in a document but rarely across all documents are given higher weights. This helps emphasize words that are more meaningful to the context.

2. **Handling Common Words**  
   - **BoW:** Treats all words equally, which means common stopwords (like "the", "and", "is") can dominate the representation.
   - **TF-IDF:** Downweights these common words, reducing noise and ensuring that more significant words have a greater impact on the feature representation.

3. **Better Differentiation for Classification**  
   - **BoW:** Provides a basic count of words, which may not capture the nuances necessary for tasks like sentiment analysis.
   - **TF-IDF:** Enhances model performance by distinguishing documents based on the relative importance of words, thereby improving the accuracy of classification models.

4. **Dimensionality and Sparsity**  
   - **BoW:** Often results in a high-dimensional, sparse matrix where every word in the vocabulary is a feature.
   - **TF-IDF:** Helps in reducing the impact of high-dimensionality by prioritizing unique and informative words, which can lead to a more compact and discriminative feature set.

5. **Improved Classification Performance**  
   - By providing a more meaningful representation of the text, **TF-IDF** generally leads to better performance in tasks such as sentiment analysis, where capturing the subtle differences in word usage is crucial.

In summary, **TF-IDF** is preferred over **Bag-of-Words** because it not only captures the frequency of words but also their relevance within the context of the entire dataset, leading to more effective and accurate text classification.
"""
