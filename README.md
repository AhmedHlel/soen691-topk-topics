## Abstract

With the advent of large sets of PDF files in several scientific applications, such as bioinformatics and neuroscience, there is a growing need to automatically identify a list of top-K topics in each set. This list could be used later to predict descriptions, keywords, or even cluster datasets.

In this project, I will develop a machine learning (ML) model for identifying a list of top-K topics in a set of papers. The model will be implemented using Apache Spark and MLlib. My project will use a dataset containing 7241 articles.

In this project, I will study TPMS and reuse some of the features used by its ML model. To the best of my knowledge, there is no Spark-based implementation for these features or TPMS.

## Introduction

####      Context

The continuously increasing  amount of data we generate coupled with the various formats pose many challenges  in how we store and read it. For this reason, it is inevitable nowadays in the field of big data to use the adequate technologies such as Apache Spark to process this voluminous and varying data. Data processing is an essential step to extract knowledge which will impact many fields such as Neuroscience and bioinformatics. These fields heavily rely on human intervention which requires a lot of time to read reports and understand them.

####       Objective and Presentation of the problem

For this reason, in this project I will work on creating a scalable system to identify the top-k topics (word sets) covered by a set of PDF files. The list of the top-k topics will be handy in generating a set of categories labelling a given set. It will be a milestone towards predicting descriptions or keywords for a given dataset or even caption an image related to a given set of PDF files.

####         Related Work

The Toronto Paper Matching System (TPMS)[1] has been implemented to find the suitable reviewer for a paper. The system was widely adopted by several top conferences such VLDP and NeurIPS. In fact, TPMS, using the publications of a reviewer can assess their expertise by assigning them a score. This score is the metric the TPMS relies on to predict the affinity between a given paper under review and a potential reviewer. Even though the TPMS has different end than my project's, we still can witnesss a common ground. In fact, the TPMS uses the publication to score the expertise of a reviewer. I will alter this part in order to reflect the top-k topics in a given set of PDF files. In addition, the TPMS has a limitation which I will be addressing in my project which scalability by using Apache Spark.



## Materials and Methods

####         Dataset

The dataset I will be using to implement my project is a list of papers published in the Neural Information Processing Systems conference (NeurIPS) between 1987 until 2016[2]. The dataset contains 7241 articles covering a wide range of topics including Artificial Intelligence, statistics, and neuroscience. As this is an unsupervised machine learning problem, the data does not contain any labels. In order to feed my data to the model, I will need to pre-process it:  data cleaning, text tokenization, and removing stopwords.

####         Technologies

Given its widespread usage in Machine learning and data processing, the project will be mainly implemented in python. However an important feature of my project is scalability. For this reason I will additionally be using pyspark, the python API of Apache Spark, Machine Learning lirbary (MLlib). Apache Spark has been used in multiple applications such as finance and science. It has been adopted by more than 1000 companies[3]. In fact, Apache Spark is one of the most famous tools used to process a large amount of data thanks to its high performance compared to other tools such as Apache Hadoop. It enables iterative computation making its MLlib faster and more accurate. The performance of Spark is derived from distributed in-memory computation, lazy evaluation that helps in internally optimize the execution of transformations, and abstraction which improves usability.

####         Algorithms

My project will revolve around implementing an unsupervised algorithm capable of listing a number of topics covered by the set of papers I will be processing. Hence the best tools provided by Apache Spark MLlib are the topic Modeling algorithms, specifically the Latent Dirichlet Allocation (LDA) a generative probabilistic model of a corpus. Its basic idea is that documents are represented as random mixtures over latent topics, where each topic is characterised by a distribution over words[4]. The model will be able to return a list of words per topic. The number of expected topics and the number of words per topic will be passed as parameters to the model. As we lack the ground truth, I will evaluate the performance of the LDA model using Topic Coherence Measures. The coherence is defined as the median of pairwise word semantic similarities formed by top words of a given topic. Hence using these metrics we can help distinguish between the good and bad topics based on the interpretability of the words each topic contains[5]. Based on the coherence measures, I will work on improving my model by further pre-processing my data through eliminating the words that appeared in the result but semantically are not meaningful. 

------

## References

[1] Charlin, L., & Zemel, R.S. (2013). The Toronto Paper Matching System: An automated paper-reviewer assignment system.

[2] https://nips.cc/

[3] Matei Zaharia, Reynold S. Xin, Patrick Wendell, Tathagata Das, Michael Armbrust, Ankur Dave, Xiangrui Meng, Josh Rosen, Shivaram Venkataraman, Michael J. Franklin, Ali Ghodsi, Joseph Gonzalez, Scott Shenker, and Ion Stoica. 2016. Apache Spark: a unified engine for big data processing. Commun. ACM 59, 11 (October 2016), 56–65. DOI:https://doi.org/10.1145/2934664

[4] David M. Blei, Andrew Y. Ng, and Michael I. Jordan. 2003. Latent dirichlet allocation. J. Mach. Learn. Res. 3, null (March 2003), 993–1022.

[5] Michael Röder, Andreas Both, and Alexander Hinneburg. 2015. Exploring the Space of Topic Coherence Measures. In Proceedings of the Eighth ACM International Conference on Web Search and Data Mining (WSDM ’15). Association for Computing Machinery, New York, NY, USA, 399–408. DOI:https://doi.org/10.1145/2684822.2685324







