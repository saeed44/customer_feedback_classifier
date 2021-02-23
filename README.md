# Customer Feedback Classifier

This is an object-oriented-programming implementation of my solution to the problem.

The goal is to classify customer feedback messages into three categories: compliment, complaint and suggestion.

The three main scripts are: 1-eda: for exploratory data analysis, 2-preprocessing: for different transformations on the given data to prepare for eda and training; and 3-models: for training a model and testing and/or making predictions.

The desired script is run by: 
`python run.py process path1 path2 path3`. 

"process" takes the following strings: "eda", "train", "test". The arguments "path"s refer to the input data frames that correspond to three sectors of businesses that the feedback messages are coming from.

Challenge
======

One challenging aspect of this data was that the data set was kind of small, `~1800` samples. Here I have implemented the logistic regression model and the performance of the model is good, `recall~88%`. 

To improve the predictions, I used a pre-trained deep learning model (BERT) which improved the recall by `~7%`. 

Results of the model are stored in the "results" folder. 
