# SNLI
Stanford Natural Language Inference

This notebook provides various strategies for tackling the natural language inference given a premise and a hypothesis.
The task is to understand if the relationship between the two sentences is entailment, contradiction or neutral.
It's essentially a multi-class classification problem. I explore several modelling strategies including some simple baselines.
The strategies explored are:
1) Simple word matching between premise and hypothesis. Softmax classifier used
2) All word combinations between premise and hypothesis. Softmax classifier used
3) Glove pre-trained averaged embeddings with softmax
4) Glove pre-trained averaged embeddings with neural network
5) Sentence encoding RNNs with Fasttext embeddings
6) Chained model RNN with Random embeddings
7) Chained model RNN with Fasttext embeddings


