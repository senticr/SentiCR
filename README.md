# SentiCR

SentiCR is an automated sentiment analysis tool for code review comments. SentiCR uses supervised learning algorithms to train 
models based on 1600 manually label code review comments (https://github.com/senticr/SentiCR/blob/master/SentiCR/oracle.xlsx). Features of SentiCR include:

- Special preprocessing steps to exclude URLs and code snippets
- Special preprocessing for emoticons
- Preprocessing steps for contractions
- Special handling of negation phrases through precise identification 
- Optimized for the SE domain 

## Performance
In our hundred ten-fold cross-validations, SentiCR achieved 83.03% accuracy (i.e., human level accuracy), 67.84% precision, 
58.35% recall, and 0.62 f-score on a Gradient Boosting Tree based model. Details cross validation results are included here: 
https://github.com/senticr/SentiCR/tree/master/cross-validation-results
