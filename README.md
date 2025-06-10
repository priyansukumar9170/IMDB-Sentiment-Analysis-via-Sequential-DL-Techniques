# IMDB-Sentiment-Analysis-via-Sequential-DL-Techniques

🧠 Objective
To classify movie reviews from the IMDB dataset as positive or negative using various sequential deep learning architectures (RNN, LSTM, GRU) and optimize model performance through advanced preprocessing and regularization techniques.

🔄 Data Preprocessing
Text Cleaning: Applied tokenization, stop word removal, stemming, and lemmatization to clean and normalize the input text data.

Word Representation: Utilized word embedding techniques (e.g., Word2Vec or embedding layers) to convert text into dense vector representations that capture semantic relationships.

🏗️ Model Development
Implemented and compared Recurrent Neural Network (RNN), Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU) architectures using Keras/TensorFlow.

Used Binary Cross-Entropy as the loss function for binary classification.

Trained models using the Stochastic Gradient Descent (SGD) optimizer with tuned learning rates and momentum.

🎯 Key Achievements
Achieved 91% accuracy on the test dataset using the LSTM model, outperforming other recurrent models.

Demonstrated improved sequence handling and long-term dependency learning with LSTM architecture.

🛡️ Regularization & Overfitting Control
Applied Dropout layers to prevent co-adaptation of neurons.

Used gradient clipping to stabilize training and prevent exploding gradients.

Incorporated L1 and L2 regularization to penalize overly complex models, ensuring robust generalization to unseen data.

📈 Evaluation
Visualized training vs. validation accuracy and loss to monitor overfitting.

Evaluated models using confusion matrix, accuracy, precision, recall, and F1 score.
