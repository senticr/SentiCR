
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

## Cite

Ahmed, T. , Bosu, A., Iqbal, A. and Rahimi, S., "SentiCR: A Customized Sentiment Analysis Tool for Code Review Interactions", In Proceedings of the 32nd IEEE/ACM International Conference on Automated Software Engineering (NIER track).

@INPROCEEDINGS{Ahmed-et-al-SentiCR,

 author = {Ahmed, Toufique and Bosu, Amiangshu and Iqbal, Anindya and Rahimi, Shahram},
 
 title = {{SentiCR: A Customized Sentiment Analysis Tool for Code Review Interactions}},
 
 year = {2017},
 
 series = {ASE '17},
 
 booktitle = {32nd IEEE/ACM International Conference on Automated Software Engineering (NIER track)}, 
}
