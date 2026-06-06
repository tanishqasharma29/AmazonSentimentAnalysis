# Machine Learning & Deep Learning Projects

## Amazon Sentiment Analysis

Customer reviews play a crucial role in helping businesses understand customer satisfaction and improve their products. With thousands of reviews being generated daily on Amazon, manually analyzing customer feedback becomes difficult and time-consuming. This project uses Natural Language Processing (NLP) and Machine Learning techniques to automatically classify Amazon product reviews as either positive or negative.

The review text is first cleaned by removing stopwords and unnecessary words. After preprocessing, TF-IDF Vectorization is used to convert text into numerical features that can be understood by machine learning algorithms. A Logistic Regression model is then trained to predict the sentiment of new reviews. The project also includes Word Cloud visualizations and a Confusion Matrix to better understand model performance.

**Technologies Used:** Python, NLTK, Scikit-learn, Pandas, Matplotlib, WordCloud

---

## Detecting Spam Emails

Spam emails often contain advertisements, phishing attempts, scams, or malicious content that can affect users and organizations. The goal of this project is to automatically identify whether an email is spam or legitimate (ham) using Deep Learning techniques.

The email text undergoes preprocessing steps such as punctuation removal, stopword removal, tokenization, and sequence padding. An LSTM (Long Short-Term Memory) neural network is trained using TensorFlow to understand patterns in email content and classify messages accurately. The project also includes Word Cloud visualizations and training performance graphs to analyze model effectiveness.

**Technologies Used:** TensorFlow, Keras, NLTK, Pandas, NumPy, WordCloud

---

## SMS Spam Detection

SMS spam messages are commonly used for promotions, scams, and fraudulent activities. This project develops an intelligent SMS classification system capable of distinguishing spam messages from genuine messages.

The dataset is preprocessed using TensorFlow’s TextVectorization layer, which converts raw text into numerical sequences. Three different models are implemented and compared: a Dense Embedding Model, a Bidirectional LSTM Model, and a Transfer Learning model using the Universal Sentence Encoder (USE). Their performance is evaluated using Accuracy, Precision, Recall, and F1-Score, allowing comparison of different deep learning approaches for text classification.

**Technologies Used:** TensorFlow, TensorFlow Hub, Keras, Pandas, Scikit-learn

---

## Classify Text Documents

Text classification is one of the most common applications of Natural Language Processing. This project focuses on automatically categorizing documents into predefined categories such as Technology, Sports, Politics, and Entertainment.

The text documents are transformed into numerical features using Count Vectorization, which represents the frequency of words within each document. A Multinomial Naive Bayes classifier is trained on these features to learn category-specific patterns. Once trained, the model can predict the category of unseen text documents with good accuracy. This project demonstrates how probabilistic machine learning algorithms can be effectively used for multi-class text classification.

**Technologies Used:** Python, Scikit-learn, Pandas, NumPy, Matplotlib

---

## IPL Score Prediction Using Deep Learning

Predicting cricket scores is an exciting application of Machine Learning and Sports Analytics. This project uses historical IPL match data to estimate the final score of an innings based on current match conditions.

Features such as batting team, bowling team, venue, current runs, wickets, overs, batsman, and bowler are used as inputs. The categorical data is encoded, numerical features are scaled, and a Deep Neural Network is trained using TensorFlow and Keras. The project also includes an interactive widget that allows users to enter live match information and instantly receive a predicted final score.

**Technologies Used:** TensorFlow, Keras, Pandas, NumPy, Scikit-learn, Matplotlib

---

## Calories Burnt Prediction using Machine Learning

Monitoring calories burned during physical activity is important for maintaining fitness and tracking health goals. This project predicts the number of calories burned during exercise using various biological and workout-related features.

The dataset contains information such as age, gender, height, weight, workout duration, heart rate, and body temperature. Multiple machine learning regression models including Linear Regression, Random Forest Regressor, Ridge Regression, Lasso Regression, and XGBoost are trained and compared. The project identifies the best-performing model based on Mean Absolute Error (MAE) and demonstrates how machine learning can be applied in fitness and healthcare applications.

**Technologies Used:** Scikit-learn, XGBoost, Pandas, NumPy, Matplotlib, Seaborn

---

## Titanic Survival Prediction

The Titanic Survival Prediction project is one of the most popular beginner machine learning problems. It focuses on predicting whether a passenger survived the Titanic disaster based on demographic and travel-related information.

The dataset includes passenger details such as age, gender, passenger class, fare, embarkation point, and family information. Extensive feature engineering is performed, including age grouping, title extraction from names, missing value handling, and categorical encoding. A Random Forest Classifier is then trained to predict survival outcomes. The project demonstrates the complete machine learning pipeline, from preprocessing and feature engineering to model evaluation and prediction.

**Technologies Used:** Python, Scikit-learn, Pandas, NumPy, Seaborn, Matplotlib

---

## Predict Fuel Efficiency

Fuel efficiency is a critical factor in automobile design because it directly impacts fuel consumption, operating costs, and environmental sustainability. This project predicts vehicle fuel efficiency, measured in Miles Per Gallon (MPG), using automobile specifications.

The dataset contains features such as cylinders, horsepower, weight, acceleration, model year, and origin. After cleaning the data and handling missing values, a Deep Learning regression model is built using TensorFlow. Batch Normalization and Dropout layers are incorporated to improve training stability and reduce overfitting. The trained model learns the relationship between vehicle characteristics and fuel consumption, enabling accurate MPG predictions.

**Technologies Used:** TensorFlow, Keras, Pandas, NumPy, Matplotlib, Seaborn

---

## License Plate Recognition

License Plate Recognition is a practical Computer Vision application used in traffic management systems, parking automation, law enforcement, and surveillance. This project automatically detects and extracts vehicle registration numbers from images.

The system uses YOLOv8 for object detection and EasyOCR for Optical Character Recognition (OCR). When a vehicle image is uploaded, YOLO identifies regions containing potential license plates. These regions are then cropped and passed to EasyOCR, which extracts the text from the image. The detected license plate number is displayed as output. The project demonstrates the integration of object detection and OCR technologies for building an Automatic Number Plate Recognition (ANPR) system.

**Technologies Used:** YOLOv8, EasyOCR, OpenCV, Python, Google Colab
