# Movie-Genre-Classification

# Project Description

The goal of this project is to develop an algorithm for determining a movie's genre based on its description. Natural language processing techniques were used to analyze textual descriptions, and movies were classified into appropriate genres.

# Algorithms
1. **K-Nearest Neighbors**
2. **Naive Bayes**
3. **Recurrent Neural Network**

# Evaluation

The accuracy of the algorithms was evaluated using a confusion matrix. Based on the confusion matrix, precision, recall, and the F-measure (Micro and Macro F-measure) were calculated for each algorithm.

## Installation

1. **Clone the Repository**:
    ```
    git@github.com:matijap59/Movie-Genre-Classification.git
    ```
2. **Create Virtual Environment**:
    ```
    python -m venv venv
    ```
3. **Activate Virtual Environment**:
    ```
    venv\Scripts\activate# On Windows
    # or
    source venv/bin/activate  # On macOS/Linux
    ```
4. **Install Dependencies**:
    ```
    pip install -r requirements.txt
    ```
5. **Running**:
   
   Depending on which algorithm you want to use, you should start the corresponding script:
   
   For K-Nearest Neighbors algorithm
   ```
   python3 Knn-algorithm.py
   ```
   For Naive Bayes algorithm
   ```
   python3 Naive-Bayes-algorithm.py
   ```
   For Recurrent Neural Network
    ```
   python3 RNN-Classification.py
   ```  


