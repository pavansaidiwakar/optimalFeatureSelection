# optimalFeatureSelection
Selection of optimal features using genetic algorithm to improve classification accuracy(rate)  


In the real world, data collection and storage technology have enabled to accumulate an
enormaous amount of data. The high dimensional datasets may be obtained from a single or
multiple sources, where each source may be considered as a view of the data. Diversity of the
multiple sourced of the data provide the intuition to learn from multiple sources. The high
dimensional data also lack adequate number of sample to reduce an effective model especially
for the classification task. 

The high dimensionality poses problems challenges of curse of
dimensionality. The feature selection is the most common approach for handling the high
dimensionality.Among the various feature selection algorithms we explore Genetic Algorithms
to obtain a reduced feature subset. 
Ferrer diagram based partitioning methods along with
majority voting is used to classify the features.
Given the large number of features, it is difficult to find the subset of features that isuseful for a given task. 

Genetic Algorithms (GA) can be used to alleviate this problem, b
searching the entire feature set, for those features that are not only essential but improve
performance as well. In this project, we explore the approaches to use GA to select features and
pass the reduced feature subset to Decision Tree Classifier for improved classification rate.
