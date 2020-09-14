# QuickRecommender
A quick, unsupervised, content-based recommendation system.

## Introduction
I was looking into recommendation systems a while ago and just came up with this quick way of creating a recommendation system. It'll create a nearest-neighbors graph and use it to assign probability values to what each user might like or not and recommend based on those probabilities. Some of the ideas such as diversification using K-Means++ came from Yum-Me <a href="#yangetalyumme">[1]</a>.

## Dependencies
- Python (>= 3.6)
- NumPy (>= 1.13.3)
- SciPy (>= 0.19.1)
- Scikit-Learn (>= 0.23)

## References
<div id="yangetalyumme">
[1] Yang, Longqi, Cheng-Kang Hsieh, Hongjian Yang, John P. Pollak, Nicola Dell, Serge Belongie, Curtis Cole, and Deborah Estrin. "Yum-me: a personalized nutrient-based meal recommender system." ACM Transactions on Information Systems (TOIS) 36, no. 1 (2017): 1-31. (<a href="https://arxiv.org/abs/1605.07722">arXiv</a>)
</div>
