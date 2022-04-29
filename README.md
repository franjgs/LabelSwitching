#Imbalance datasets 
High imbalance occurs in many real-world application areas where the decision system is aimed to detect rare but important cases. Simply by listing some of them, data imbalance is found in the Information Technology area, in biomedical data, industrial applications, or in the financial area.

The imbalance implies a difficulty for learning algorithms, since they will be biased towards the most frequent cases. To overcome such bias to majority samples, specific machine learning algorithms must be applied. This domain is known as learning from imbalanced data [[he2009learning](https://doi.org/10.1109/TKDE.2008.239)], [[fernandez2018learning](https://link.springer.com/book/10.1007/978-3-319-98074-4)].

 The approaches to dealing with imbalanced datasets are usually sorted
 into three categories: 

1.   Data-level methods concentrate on modifying the training set to make it 
 suitable for a standard learning algorithm; balancing distributions by 
 creating new objects for minority classes (oversampling and variations 
 such as SMOTE [[chawla2002smote](https://doi.org/10.1613/jair.953)] and its different modifications, or removing examples from majority classes 
 (undersampling belong to this category).
2. Algorithm-level methods modify existing learning algorithms to 
 alleviate the bias towards majority examples; cost-sensitive approaches 
 [[lopez2012analysis](https://doi.org/10.1016/j.eswa.2011.12.043)], which assume higher misclassification
 costs for samples in the minority class fall in this category. 
3. Ensemble learning [[rokach2010ensemble](https://link.springer.com/article/10.1007/s10462-009-9124-7)] [[polikar2012ensemble](https://link.springer.com/chapter/10.1007/978-1-4419-9326-7_1)]
is a multi-learner paradigm in which several models, or base learners,
are trained using diverse examples, and their complementary (or
uncorrelated) predictions are fused to yield a final decision.

Class-label switching improves the classification accuracy of neural network ensembles outperforming bagging and boosting in some common imbalanced scenarios. 

An obvious question that may arise is whether the diversity on majority and minority class should be equal or different. Our answer is to use asymmetric label-switching [[gutierrez2020asymmetric](https://doi.org/10.1016/j.inffus.2020.02.004)].
