<h1 align="center">strong-language-detector</h1>
<p align="center">just another ML-based profanity checker</p>

---
## Introduction
The initial idea for this project came from the [Hack the North 2020++ project](https://github.com/j985chen/Purity) I helped to build, which strove to make the internet, in all of its profanity and distraction-filled glory, easier to navigate for young children. 
The original project censored strong and/or profane language by matching it to an array of swear words we compiled; however, given the wide variety of inventive swear words, insults, and slurs found on the internet today, that solution is clearly insufficient beyond a two-day hackathon project.

## Process
Using scikit-learn, I built a simple logical regression model for a dataset compiled from multiple sources (see the citations), using a Bag of Words representation of each sentence in the dataset and classifying each sentence using a label of 1 (strong language) or 0 (not strong language).
Some of the datasets used had more specific classifications: one had the classifications "hate speech" and "offensive language", while another had the classifications "toxic", "severe toxic", "obscene", "threat", "insult", and "identity hate". 
In these cases, I classified a sentence classified as any of the above as being a 1 (strong language), with everything else as a 0 (not strong language).

A summary of my dataset:

| Dataset   | Strong language | Not strong language | 
| --------- | --------------- | ------------------- | 
| Twitter   | 20620 (83.2%)   | 4163 (16.8%)        |
| Fox News  | 435 (28.5%)     | 1093 (71.5%)        |
| Wikipedia | 16225 (10.2%)   | 143346 (89.8%)      |

## How to run
Use pip to install the necessary dependencies:

`pip install pandas scikit-learn joblib flask`

For a full list of dependencies, see _dependencies.txt_

To run the api, navigate to the apis folder and run `python api.py <port>`, where you can (optionally) specify a port. The api endpoint is `/is_strong`.

## Citations
### Fox News dataset:
Gao, L., &amp; Huang, R. (2018, May 22). Detecting Online Hate Speech Using Context Aware Models. arXiv.org. https://arxiv.org/pdf/1710.07395.pdf. 

The dataset can be found [here](https://github.com/sjtuprog/fox-news-comments)

### Twitter dataset: 
Ribeiro, M. H., Calais, P. H., Santos, Y. A., Almeida, V. A. F., &amp; Meira, W. (2018, March 23). Characterizing and Detecting Hateful Users on Twitter. arXiv.org. https://arxiv.org/pdf/1803.08977.pdf. 

The dataset can be found [here](https://github.com/manoelhortaribeiro/HatefulUsersTwitter)

### Wikipedia dataset:
The dataset can be found [here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)

## Resources used
- [Building a Better Profanity Detection Library with scikit-learn](https://towardsdatascience.com/building-a-better-profanity-detection-library-with-scikit-learn-3638b2f2c4c2) by Victor Zhou
- [Turning Machine Learning Models into APIs in Python](https://www.datacamp.com/community/tutorials/machine-learning-models-api-python) by Sayak Paul
