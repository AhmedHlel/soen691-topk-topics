# soen691-topk-topics
**A Scalable System for Identifying Top-K Topics  in a Set of Papers**

With the advent of large sets of PDF files in several scientific applications, such as bioinformatics and neuroscience, there is a growing need to automatically identify a list of top-K topics in each set. This list could be used later to predict descriptions, keywords, or even cluster datasets. 

In this project, I will develop a machine learning (ML) model for identifying a list of top-K topics in a set of papers. The model will be implemented using Apache Spark and MLlib. My project will use a dataset containing 473 articles with their corresponding labels. More details about this dataset are available at [1].

The Toronto Paper Matching System (TPMS) finds a suitable reviewer for a paper [2]. TPMS is widely adopted by top conferences, such as VLDB and NeurIPS, and used by Microsoft’s Conference Management Toolkit (CMT). TPMS gets a set of papers published by each reviewer and uses an ML model to predict the affinity between the paper under review and the sets of papers. 

In this project, I will study TPMS and reuse some of the features used by its ML model. To the best of my knowledge, there is no Spark-based implementation for these features or TPMS. 

------

#### References

1-Paper-To-Reviewer-Matching-System https://github.com/rahulguptakota/paper-To-Reviewer-Matching-System 

2- The Toronto Paper Matching System: An automated paper-reviewer assignment system http://www.cs.toronto.edu/~lcharlin/papers/tpms.pdf 