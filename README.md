# NLP-Sentiment-Analysis
[![](https://img.shields.io/badge/Python-blue?style=for-the-badge)](https://github.com/hamzamohdzubair/redant)
[![](https://img.shields.io/badge/NLP-blueviolet?style=for-the-badge)](https://hamzamohdzubair.github.io/redant/)
[![](https://img.shields.io/badge/Library-NLTK-yellow?style=for-the-badge)](https://docs.rs/crate/redant/latest)
[![](https://img.shields.io/badge/Library-TensorFlow-purple?style=for-the-badge)](https://docs.rs/crate/redant/latest)
[![](https://img.shields.io/badge/Library-Keras-blue?style=for-the-badge)](https://docs.rs/crate/redant/latest)


[![](https://img.shields.io/badge/Package-Tokenize-orange?style=for-the-badge)](https://crates.io/crates/redant)
[![](https://img.shields.io/badge/Package-Corpus-green?style=for-the-badge)](https://crates.io/crates/redant)
[![](https://img.shields.io/badge/Package-WordNetLemmatizer-yellow?style=for-the-badge)](https://crates.io/crates/redant)

![](https://img.shields.io/static/v1?label=&message=WordCloud&color=green)

## Research Question 
Can we analyze customer sentiments from the reviews using neural networks and NLP techniques so that our company can take appropriate actions to address any issues?

## The goal of the data analysis
The objective is to predict how a user feels about a product based on their word choices. By using this sentiment analysis based on past reviews, we can predict customer sentiment so the company can take appropriate actions for maximum customer satisfaction. We can also use a successful model to evaluate customer sentiment and make recommendations.

## Technique Justification
Artificial neural networks, designed with biological neural network inspiration, can comprehend unstructured data and make general observations without explicit training.

Recurrent neural Networks (RNN) are a class of artificial neural networks where the output of recurrent neural networks depends on the previous elements within the sequence. RNN leverage backpropagation which aids the network to learn continuously by using corrective feedback loops to improve their predictive analytics. Simply there is one correct path to the output. To find this path neural network uses a feedback loop, checking each node's guess about the next node, assigning weight values according to correct/incorrect guesses, making new predictions using the higher weight paths, then repeating the process.

RNN algorithms are commonly used in language translation, natural language processing (NLP), speech recognition, and image captioning. Some examples of applications that incorporate RNN are Siri and Google Translate.

![image](https://github.com/secil-carver/NLP-Sentiment-Analysis/assets/77639654/79ed8f8b-1583-465f-831f-040ad16e03c7)

## Vocabulary Size and Tokenization
The vocabulary size is the number of unique words used in the dataset. Keras Tokenizer method creates a dictionary based on word frequency, where low numbers represent higher frequency words. The length of this dictionary represents the number of unique words in the corpus or data frame.

## Word embedding length
Embedding is a dense vector floating point representation of the words in NLP, specified by weight. The representation vectors encode words closer in the vector space with similar meaning. Word embedding length is the positional distance of the word from the beginning of the vector. Word embedding length is determined by taking the fourth root of the vocabulary size.

The words are vectorized in NLP. The sequence length is the length of the longest sentence in the input data. The maximum sequence can be used to preserve most of the information during data input. The shorter sequences are then padded with 0s at the beginning or at the end of the sequence to make all sequences an even length. I used the longest sentence as the maximum sentence length and pre-padded the shorter sentences.

![image](https://github.com/secil-carver/NLP-Sentiment-Analysis/assets/77639654/fbcda0b6-5d96-4ecf-83ba-10918d0f4e82)

## Goal Of Tokenization Process
Tokenization is the process of separating strings. These tokens can be processed further by being replaced, formatted, and lemmatized. Tokenization is also used to count words, documents, and characters, add indexes, as well as prepare sequences to be padded, truncated, or masked to unify sequences.

## Padding Process
Padding is a necessary technique in Neural Networks to preserve the matrix dimensions for the input data. It is implemented before or after sequences, adding 0s to fill the necessary gaps to reach the maximum sequence length so that all sequences have equal lengths in the tensor.

## Number of Layers in Network Architecture
Artificial neural networks have two main hyperparameters that control the architecture or topology of the network: the number of layers and the number of nodes in each hidden layer. The input layer of input variables will eventually lead to the output layer which produces the output variables with one or more hidden layers in between. The number of nodes in layers determines the width, and the number of layers determines the depth of the neural network.

**There are 5 layers in the model:**

- 1st layer- Core type: Embedding / Layer type: Input
- 2nd layer- Core type: Pooling / Layer type: GlobalAveragePooling1D 
- 3rd layer- Core type: Dense / Layer type: Hidden
- 4th layer- Core type: Dense / Layer type: Hidden
- 5th layer- Core type: Dense / Layer type: Output

# Model Evaluation

## Justification of hyperparameters

- **Activation functions:** Activation functions are the function through which we pass our weighted sums. Activation functions can be used as many times as desired at any point. I used Relu for the first two layers, then Sigmoid for the last layer. Relu is a fast and efficient activation function. I used the Sigmoid function for the output function, which is used in binary probability outcomes between 0 and 1. However, since it is a slower function I saved it only for the last layer.   

- **Number of nodes per layer:** 
input layer = 118 nodes, one per feature<br>
output layer = 1 node for class since using sigmoid activation function<br>
hidden layer nodes = 100 and 50 (gradual decrease from input layer)

- **Loss function:** I used binary cross-entropy as a loss function since the output is binary positive/negative. Entropy is a measure of loss or uncertainty. Binary cross-entropy is the computation of cross-entropy between both distributions. 

- **Optimizer:** I used the 'Adam' optimizer because it runs fast. An optimizer is a function that modifies the attributes such as weights and learning rates of each epoch in a neural network algorithm. Adam optimizer is an extension of stochastic gradient descent, but Adam updates the learning rate for each network weight individually rather than a single learning rate through training. Since an optimizer helps reduce the loss and improve accuracy it is a crucial step in deep learning models. Adam has many benefits making it widely used and recommended as a default optimizer.

- **Stopping criteria:** I used EarlyStopping for validation_loss when it reaches a minimum level for 3 consecutive epochs. I set epochs to 50 so the dataset would run 50 times all the way through, but will stop if/when the validation_loss reaches a minimum level and doesn't improve from that for 3 epochs. 

- **Evaluation metric:** I used 'accuracy' to evaluate how often predictions equal labels.

