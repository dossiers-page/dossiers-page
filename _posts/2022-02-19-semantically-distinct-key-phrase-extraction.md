---
title: Semantically Distinct Key Phrase Extraction
---

Working with Hilbert Space is fascinating. It is a hash function with prefix matching properties and two hashes can be compared just like a ZIP Code or a PIN Code.  This implies that a vector in positive space can be hashed. Combining this with a typical vector transformation in NLP (say word2vec) can generate one dimensional embeddings. 

To give an example,  consider the words in months (april, july, january) and mathematics (mathematics, deep learning, machine learning).  If we transform those words using a word vector embedding and do a hilbert hash,  it would look like the following figure 

![Hilbert Hash for various words from word2vec )](/assets/images/hilbert_hash_example_months_maths.png)

What it implies are
* If the topics are different, the hash prefix would differ 
* We can reduce all the words in an article to set of hashes and find most common prefixes they belong
* The look up table can be saved as a dictionary for easy re-use. The original word vectors are no longer needed
* A trie can do fast look up under each subtree and can summarize/rank keywords 

Using this approach, the first thing we could do is to generate distinct key phrases from a given article text. The problem is caleed **Key phrase extraction**.  The uniqueness of this approach is that generated key phrases will be semantically distnict. 

## Output from the package 

![Distinct Keywords Sample Output )](/assets/images/distinct_keywords_sample_output.png)
## Generalization and benchmarks
The approach can be generalized to any vector embedding technique and can do semantic sentence comparison or document comparison in an unsupervised setting.  The current implementation used Trie and SortedDict for making it one of the fastest implementation.  The approach does not require any training and shown a 31% recall score while doing benchmark with KPTimes Test Data Set (20000 articles) with manual keywords 

![KP Times Test Data Recall Score )](/assets/images/kptimes_test_data_recall_score.png)


#### Github Link 
[DistinctKeywords](https://github.com/sahyagiri/DistinctKeywords)
