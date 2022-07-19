## TLMOTE: A Topic-based Language Modelling Approach for Text Oversampling

Training machine learning and deep learning models on unbalanced datasets can lead to a bias portrayed by the models towards the majority classes. To tackle the problem of bias towards majority classes, researchers have presented various techniques to oversample the minority class data points. Most of the available state-of-the-art oversampling techniques generate artificial data points which cannot be comprehensibly understood by the reader, despite the synthetic data points generated being similar to the original minority class data points. We present Topic-based Language Modelling Approach for Text Oversampling (TLMOTE), a novel text oversampling technique for supervised learning from unbalanced datasets. TLMOTE improves upon previous approaches by generating data points which can be intelligibly understood by the reader, can relate to the main topics of the minority class, and introduces more variations to the synthetic data generated.

We evaluated the efficacy of our approach on various tasks like Suggestion Mining SemEval 2019 Task 9 Subtasks A and B, SMS Spam Detection, and Sentiment Analysis. Experimental results verified that over-sampling unbalanced datasets using TLMOTE yields a higher macro F1 score than with other oversampling techniques.

### Instructions:
1. Create a python3 envrionment and install all the packages mentionsed in the requirements.txt file.
2. Download the tlmote.py file and the glove.6B.100d (https://nlp.stanford.edu/data/glove.6B.zip) file in the same directory.
2. Store the unbalanced dataset in a .csv file, with column names "Sentence" and "Tag" for the text and labels respectively.
3. Run the shell command:

```sh
python tlmote.py data_directory_and_file

```

## If you use our code, please do cite our paper. The citation is as follows:
Choudhry, A., Susan, S., Bansal, A., & Sharma, A. (2022). TLMOTE: A Topic-based Language Modelling Approach for Text Oversampling. The International FLAIRS Conference Proceedings, 35. https://doi.org/10.32473/flairs.v35i.130676
