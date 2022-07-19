# Imports ==========================================================================================================================

import collections
import gensim
import keras
import matplotlib.pyplot as plt
import nltk
import nltk.stem as stemmer
import numpy as np
import pandas as pd
import string
import sys

from collections import Counter
from gensim import corpora, models
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from pprint import pprint
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Embedding, Dense, Input, Bidirectional
from tensorflow.keras.models import Sequential
from tqdm import tqdm

nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))


# Imports End ======================================================================================================================

# Helper Functions =================================================================================================================

def lemmatize_stemming(text):
    stemmer = SnowballStemmer('english')
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def ngrams_with_topics(mc_sugg, topic_words):
    filtered = []
    for ng in mc_sugg:
        flag = 0
        for word in ng[0]:
            if word in topic_words:
                flag = 1
        if flag:
            filtered.append(ng)
    return filtered


def ngrams_lang_inp(arr, n, word2id):
    inp = []
    curr = []
    for s in arr:
        i=0
        while(i+n-1<len(s)):
            ngram = list(s[i:i+n])
            idngram = [word2id[w] for w in ngram]
            if i+n<len(s):
                nextword = word2id[s[i+n]]   
            else:
                break
            curr = [idngram, nextword]
            inp.append(curr)
            curr = []
            i=i+1
        
    return inp


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token!="https":
            result.append(lemmatize_stemming(token))
    return result


def preprocess_df(df):
    inp = df.values
    for i in range(len(inp)):
        s = inp[i]
        sl = s.lower()
        sl = result = sl.translate(sl.maketrans('','', string.punctuation))
        tokens = word_tokenize(sl)
        #tokens = [w for w in tokens if not w in stop_words]
        inp[i] = tokens
    return inp


def string2cnov(text,dict):
            
    val=0
    pres=""
    flag=False
    for curr in range(0,len(text)-1):
        
        #print (text[curr] +" ")
        #print(pres+" ")
        if text[curr]=='*':
          val=float(pres)
          pres=""
        elif text[curr]=='"' and flag==False :
          
          flag=True
          
          curr=curr+1
          ster=""
          while (text[curr]!='"') :
                ster=ster+text[curr]
                curr=curr+1
          curr=curr+3
          filler=0
          if (ster!="."):
              if ster in dict.keys():
                  filler=dict[ster]
              dict[ster]=filler+val
              if (curr>=len(text)):
                break
          #print(text[curr])
          
          pres=""
          val=0
        elif text[curr]=='"' and flag==True :
            flag=False
        elif text[curr]=='.' or (text[curr]>='0' and text[curr]<='9'):
            pres=pres+text[curr]
    return dict


# Helper Functions End =============================================================================================================

def main():
    file = sys.argv[1]
    folder = '/'.join(file.split('/')[:-1])
    train_df = pd.read_csv(file)
    train_df_gen = pd.read_csv(file)
    print(train_df.head())
    train_df.dropna(inplace=True)
    print(train_df.shape)
    
    tags = train_df['Tag'].unique()
    max_samples = 0
    print('Tag - Number of Samples')
    for tag in tags:
        samples = sum(train_df['Tag']==tag)
        print(tag,'-',samples)
        max_samples = max(max_samples, samples)
    
    for tag in tags:
        train_df = pd.read_csv(file)
        train_df.dropna(inplace=True)
        samples = sum(train_df['Tag']==tag)
        if samples == max_samples:
            continue

        print("Generating Synthetic Samples for Class",tag)
        
        sugg_samples = []
        print(train_df.size)
        for i in range(len(train_df)):
            if(train_df.iloc[i]['Tag']==tag):
                sugg_samples.append(train_df.loc[i]['Sentence'])
        num_of_topic_words = 100
        num_of_ngrams = 15000

        print("\tExtraction of all n-grams for n=5")

        textp = ' '.join(sugg_samples)
        textp = textp.lower()
        textp = textp.translate(textp.maketrans('','', string.punctuation))
        textp = word_tokenize(textp)

        ngrams = lambda a, n: zip(*[a[i:] for i in range(n)])
        mc_sugg = Counter(ngrams(textp, 5)).most_common(num_of_ngrams)

        print("\tFinished")
        print("\tTopic Modeling")

        np.random.seed(2018)
        tdictr = {}
        processed_docs = [preprocess(text) for text in sugg_samples]
        dictionary = gensim.corpora.Dictionary(processed_docs)
        count = 0
        for k, v in dictionary.iteritems():
            count += 1
            if count > 50:
                break

        bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
        bow_corpus[len(bow_corpus)-1]
        tfidf = models.TfidfModel(bow_corpus)
        corpus_tfidf = tfidf[bow_corpus]
        for doc in corpus_tfidf:
            pprint(doc)
            break
        
        lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=100, id2word=dictionary, passes=2, workers=2)
        
        dictr={}
        for idx, topic in lda_model.print_topics(-1):
            d=string2cnov(topic,dictr)
            t=string2cnov(topic,tdictr)
        
        newd = dict(Counter(dictr).most_common(num_of_topic_words))
        values = list(newd.values())
        topic_words = set(newd.keys())

        print("\tFnished")
        print("\tExtraction of N-Grams with atleast one word among the most common keywords")

        filtered_mc_sugg = ngrams_with_topics(mc_sugg, topic_words)
        filtered_mc_sugg = filtered_mc_sugg[:(max_samples-samples)]

        print("\tFinished")
        print("\tPreprocessing for Training Language Models")

        train_df = pd.read_csv(file)
        train_df.dropna(inplace=True)
        train_df = train_df[train_df['Tag']==tag]['Sentence']
        textall = preprocess_df(train_df)

        en_model = {}
        f = open('glove.6B.100d.txt', encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            en_model[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(en_model))

        for t in textall:
            for w in t:
                if w in string.punctuation:
                    t.remove(w)

        words=[]
        keys=[]
        i=1
        print('\t\tStarting')
        for k, tsent in enumerate(textall):
            if (k%1000 == 0):
                print ('k', k, k/textall.size)
            for j, word in enumerate(tsent):
                if word not in words:
                    words.append(word)
                    keys.append(i)
                    i=i+1
        print ('\t\tDone!')

        embed_len = len(words)
        print (embed_len)

        word2id = dict(zip(words, keys))
        id2word = dict(zip(keys, words))
        embeddings = np.zeros((embed_len+1,len(en_model['test'])))  # num_words * 100 (word vec len)
        print('\t\tStarting')
        for i, w in enumerate(words):
            try:
                vec = en_model[w]
            except KeyError:
                pass
            embeddings[word2id[w]]=vec
        print('\t\tDone')


        for v in range(4,7):
            sugg_text = []
            sugg_len = len(sugg_samples)
            print (sugg_len)
            print ('\t\tStarting')
            for i in range(sugg_len):
                s = sugg_samples[i]
                sl = s.lower()
                sl = result = sl.translate(sl.maketrans('','', string.punctuation))
                tokens = word_tokenize(sl)
                sugg_text.append(tokens)
            print ('\t\tFinishing')
            sugg_lang_seq = ngrams_lang_inp(sugg_text, v, word2id)
            train_xp = np.array([s[0] for s in sugg_lang_seq])
            train_yp = np.array([s[1] for s in sugg_lang_seq])

            print('\t\tDone')

            print("\tTraining Language Model - ",v)

            modelp = Sequential()
            modelp.add(Embedding(embeddings.shape[0], output_dim = 100, weights = [embeddings], input_length=v, trainable = True))
            modelp.add(Bidirectional(LSTM(256, activation = 'relu')))
            modelp.add(Dense(1024, activation = 'relu'))
            modelp.add(Dense(embeddings.shape[0], activation = 'softmax'))

            modelp.compile(optimizer = 'Adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

            modelp.fit(train_xp, train_yp, batch_size = 64, epochs=20, verbose=True)

            print("\tSaving Language Model - ",v)
            modelp.save(folder + "/tag_" + str(tag) + '/models/sugg_model'+str(v))
        
        print("\tLoading Language Models")

        model4 = keras.models.load_model(folder+"/tag_"+str(tag)+'/models/sugg_model4')
        model5 = keras.models.load_model(folder+"/tag_"+str(tag)+'/models/sugg_model5')
        model6 = keras.models.load_model(folder+"/tag_"+str(tag)+'/models/sugg_model6')

        print("\tGeneration of synthetic datapoints using the three language models")

        #Randomized output of 3 language models for generation of synthetic sample
        sugg_generated = []
        for sug in tqdm(filtered_mc_sugg, "Progress"):
            ngram_gen = sug[0] #('', '', '', '', '')
            idngram = []
            for p in ngram_gen:
                idngram.append(word2id[p])
            #[1, 2, 3, 4, 5]
            ran_len = np.random.randint(20, 30)
            sentence = [id2word[i] for i in idngram]
            curr5 = idngram[:5]
            curr4 = idngram[1:5]
            curr6 = idngram[:5]
            for l in range(ran_len):
                inp4 = np.array([curr4, [2, 3, 4, 5]])
                out4 = model4.predict(inp4)
                i4 = np.argmax(out4[0])
                max4 = out4[0][i4]
                inp5 = np.array([curr5, [2, 3, 4, 5, 6]])
                out5 = model5.predict(inp5)
                i5 = np.argmax(out5[0])
                max5 = out5[0][i5]
                if l != 0:
                    inp6 = np.array([curr6, [2, 3, 4, 5, 6, 7]])
                    out6 = model6.predict(inp6)
                    i6 = np.argmax(out6[0])
                    max6 = out6[0][i6]

                if l != 0:
                    z = np.random.randint(0, 2)

                else:
                    z = np.random.randint(0, 1)

                if z==0:
                    p = i4
                    wordp = id2word[p]
                    sentence.append(wordp)
                    curr4 = np.delete(curr4, 0)
                    curr4 = np.append(curr4, p)
                    curr5 = np.delete(curr5, 0)
                    curr5 = np.append(curr5, p)
                    if l != 0:
                        curr6 = np.delete(curr6, 0)
                        curr6 = np.append(curr6, p)

                    else:
                        curr6 = np.append(curr6, p)

                elif z==1:
                    p = i5
                    wordp = id2word[p]
                    sentence.append(wordp)
                    curr4 = np.delete(curr4, 0)
                    curr4 = np.append(curr4, p)
                    curr5 = np.delete(curr5, 0)
                    curr5 = np.append(curr5, p)
                    if l != 0:
                        curr6 = np.delete(curr6, 0)
                        curr6 = np.append(curr6, p)

                    else:
                        curr6 = np.append(curr6, p)

                elif z==2:
                    p = i6
                    wordp = id2word[p]
                    sentence.append(wordp)
                    curr4 = np.delete(curr4, 0)
                    curr4 = np.append(curr4, p)
                    curr5 = np.delete(curr5, 0)
                    curr5 = np.append(curr5, p)
                    curr6 = np.delete(curr6, 0)
                    curr6 = np.append(curr6, p)
            sugg_generated.append(sentence)
            
        final_list = []
        for l in sugg_generated:
            sentence = " ".join(l)
            final_list.append(sentence)
        
        print("\tFinished")
        print("\tAugmenting the dataset with the synthetic samples")

        df = pd.DataFrame(final_list, columns=["Sentence"])
        df['Tag'] = tag

        train_df_gen = train_df_gen.append(df)
        print("\tFinished")
    
    print("Saving Augmented Dataset")
    train_df_gen.to_csv(file[:-4]+'_augmented.csv', index = False)

if __name__ == "__main__":
    main()
