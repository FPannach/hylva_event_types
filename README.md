# Annotating events in Mythological Sequences
* Hylemes_extract.csv: data excerpt for the annotation study on event types in mythological sequences 
* Statements are derived according to the hylistic theory (Zgoll, 2019)
* For access to the full data, contact: franziska.pannach@uni-goettingen.de
* Full data: 93 durative-initial, 1338 durative-constant, 438 durative-resultative, 4443 single-point (= punktuell)
* [submitted as a paper@LAW-XVII 2023](https://sigann.github.io/LAW-XVII-2023/cfp.html) 



## Performance
### Binary Classifer
|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| durativ      | 0.83      | 0.75   | 0.79     | 430     |
| punktuell    | 0.91      | 0.94   | 0.93     | 1148    |
| accuracy     |           |        | 0.89     | 1578    |
| macro avg    | 0.87      | 0.85   | 0.86     | 1578    |
| weighted avg | 0.89      | 0.89   | 0.89     | 1578    |


### Durative Classifier
|                   | precision | recall | f1-score | support |
|-------------------|-----------|--------|----------|---------|
| durativ-initial   | 0.62      | 0.37   | 0.47     | 27      |
| durativ-konstant  | 0.80      | 0.87   | 0.84     | 326     |
| durativ-resultativ| 0.58      | 0.49   | 0.53     | 116     |
| accuracy          |           |        | 0.75     | 469     |
| macro avg         | 0.67      | 0.58   | 0.61     | 469     |
| weighted avg      | 0.74      | 0.75   | 0.74     | 469     |



### Fine-grained Classifer
|                   | precision | recall | f1-score | support |
|-------------------|-----------|--------|----------|---------|
| durativ-initial   | 0.50      | 0.17   | 0.25     | 30      |
| durativ-konstant  | 0.73      | 0.67   | 0.70     | 297     |
| durativ-resultativ| 0.55      | 0.46   | 0.50     | 103     |
| punktuell         | 0.90      | 0.95   | 0.93     | 1148    |
| accuracy          |           |        | 0.85     | 1578    |
| macro avg         | 0.67      | 0.56   | 0.59     | 1578    |
| weighted avg      | 0.84      | 0.85   | 0.84     | 1578    |




## Running the classifiers 
You can run the pre-trained classifiers like this: 

from joblib import load

path = 'my_path'
filenames = ['binary_classifier.joblib', 'classifier_all_classes.joblib', 'classifier_durative.joblib']

for filename in filenames:
    loaded_model = joblib.load(path+filename)
    pred = loaded_model.predict(['Die Götter wohnen auf dem Olymp.']) 
    print(pred)

## Literature
Zgoll, Christian. Tractatus mythologicus: Theorie und Methodik zur Erforschung von Mythen als Grundlegung einer allgemeinen, transmedialen und komparatistischen Stoffwissenschaft, Berlin, Boston: De Gruyter, 2019. https://doi.org/10.1515/9783110541588