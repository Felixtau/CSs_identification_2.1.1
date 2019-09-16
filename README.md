# CSs_identification_2.1.1
Identification of comparative sentences in movie reviews with CSR based machine learning classifier
## Schematic Diagram
![image](https://github.com/Felixtau/CSs_identification_2.1.1/raw/master/Picture2.png)
## Source code description
data_pr.py is the source code to process train data (review_0408.txt) into required format of CSR rules, and give each sentence a tag (YES \ NO) in front of this sentence.
This result of this script py is as follow:
before:
![image](https://github.com/Felixtau/CSs_identification_2.1.1/raw/master/pr1.PNG)
required format:
![image](https://github.com/Felixtau/CSs_identification_2.1.1/raw/master/pr2.PNG)

CSR_rules_based_machine_learning_classifier.py is the source code to train the CSR rules and SVM/NB classifier to identify comparative sentences in movie reviews. there are some main functions in this code:
### tokenization and POS tagging 
token frequency:
![image](https://github.com/Felixtau/CSs_identification_2.1.1/raw/master/tokenfrequency.png)
example of tagged POS tupples.PNG
![image](https://github.com/Felixtau/CSs_identification_2.1.1/raw/master/example%20of%20tagged%20POS%20tupples.PNG)
### keywords strategy
screen shot of using keywords strategy and pos tags to generate sequences
![image](https://github.com/Felixtau/CSs_identification_2.1.1/raw/master/generated%20sequences.PNG)
### CSR rules with PrefixSpan 
the results was stored into review_0408_handled0.txt_CSR_rules.csv, here is an examle:
![image](https://github.com/Felixtau/CSs_identification_2.1.1/raw/master/CSR_Prefix.PNG)
then, select the generated rules here as features to train machine learning classifier
### The SVM/NB Classifier result
![image](https://github.com/Felixtau/CSs_identification_2.1.1/raw/master/output%20results%20of%20classfier.PNG)
the overall perfermance of these two classifers on identifing CSs in movie reviews is not as good as them in Jindal's results.
We analyzed the possible reasons in dissertation in detail: 1. Imbalanced categories. 2. Small training data size. 3. This research does not add any manual rules into a classifier. 
