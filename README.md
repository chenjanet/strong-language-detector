<h1 align="center">strong-language-detector</h1>
<p align="center">just another ML-based profanity checker</p>

---
## Introduction
The initial idea for this project came from the [Hack the North 2020++ project](https://github.com/j985chen/Purity) I helped build, which strove to make the internet, in all of its profanity and distraction-filled glory, easier to navigate for young children. 
The original project censored strong and/or profane language by matching it to an array of swear words we compiled; however, given the wide variety of inventive swear words, insults, and slurs found on the internet today, that solution is clearly insufficient beyond a two-day hackathon project.

## Process
Using scikit-learn, I built a simple logical regression model for a dataset compiled from multiple sources (see the citations), classifying each sentence using a label of 1 (strong language) or 0 (not strong language). Based on 
Some of the datasets used had more specific classifications: one had the classifications "hate speech" and "offensive language", while another had the classifications "toxic", "severe toxic", "obscene", "threat", "insult", and "identity hate". The bag-of-words model is used, and the sentences can be vectorized based on count occurrence, normalized count occurrence, and TF-IDF, given the command-line arguments 'count', 'tf', and 'tfidf' respectively.
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

To create a model, navigate to the models folder and run `python model.py <vectorize method>`, where the vectorize method is one of 'count', 'tf', or 'tf-idf'.

To run the api, navigate to the apis folder and run `python api.py <port>`, where you can (optionally) specify a port. If no port is specified, the api runs on the port `5000`. POST the texts you want to classify as an array of strings assigned to the "text" key:
```
{
    "text": ["foo", "<insert bad word here>", "bar"]
}
```
The api will return an array of 0s and 1s, with a 0 meaning that the string with that index does not contain strong language, and a 1 meaning that it does:
```
{
    "is_strong": [0, 1, 0]
}
```
The api endpoint is `/is_strong`.

## Citations
### Fox News dataset:
Gao, L., &amp; Huang, R. (2018, May 22). Detecting Online Hate Speech Using Context Aware Models. arXiv.org. https://arxiv.org/pdf/1710.07395.pdf. 

The dataset can be found [here](https://github.com/sjtuprog/fox-news-comments)

### Twitter dataset: 
Ribeiro, M. H., Calais, P. H., Santos, Y. A., Almeida, V. A. F., &amp; Meira, W. (2018, March 23). Characterizing and Detecting Hateful Users on Twitter. arXiv.org. https://arxiv.org/pdf/1803.08977.pdf. 

The dataset can be found [here](https://github.com/manoelhortaribeiro/HatefulUsersTwitter)

### Wikipedia dataset:
The dataset can be found [here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
