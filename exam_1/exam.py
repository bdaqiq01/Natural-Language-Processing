import pandas as pd
import re
from nltk.corpus import opinion_lexicon
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from collections import defaultdict, Counter
import math

positive_words = set(opinion_lexicon.positive())
negative_words = set(opinion_lexicon.negative())

#a

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def preprocess(filepath):
    df = pd.read_csv(filepath, encoding='latin-1')
    df['label'] = (df['Rating'] >= 4).astype(int)
    df['clean_review'] = df['Review'].apply(clean_text)
    return df

# (b) Feature extraction

def count_positive_words(text):
    return sum(1 for w in text.split() if w in positive_words)

def count_negative_words(text):
    return sum(1 for w in text.split() if w in negative_words)


def extract_features(df):
    df['exclamation_count'] = df['Review'].apply(lambda x: str(x).count('!'))
    df['pos_word_count'] = df['clean_review'].apply(count_positive_words)
    df['neg_word_count'] = df['clean_review'].apply(count_negative_words)
    df['has_positive'] = (df['pos_word_count'] > 0).astype(int)
    df['has_negative'] = (df['neg_word_count'] > 0).astype(int)
    
    return df

#c logistic regression

FEATURE_COLUMNS = ['exclamation_count', 'has_positive', 'has_negative',
                   'pos_word_count', 'neg_word_count']

def train_and_evaluate(X_train, X_test, y_train, y_test):

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

    return model


def build_ngram_counts(tokenized_sentences, n, min_freq=1):
    all_words = [w for sent in tokenized_sentences for w in sent] #all words inthe training 
    word_freq = Counter(all_words) #word frequency

    vocab = {w for w, c in word_freq.items() if c > min_freq} #list of unique words freq > 1 
    vocab.add('<UNK>') 

    sentences = [ 
        ['<START>'] + [w if w in vocab else '<UNK>' for w in sent] + ['<END>'] 
        for sent in tokenized_sentences
    ]  

    ngram_counts = {}
    for order in range(1, n + 1):
        counts = defaultdict(int) #count of unigram to ngram 
        for sent in sentences:
            for i in range(len(sent) - order + 1):
                gram = tuple(sent[i:i + order])
                counts[gram] += 1
        ngram_counts[order] = counts #each iten is a dic of unigram count to ngram count

    total_unigrams = sum(ngram_counts[1].values()) #total unigrams in the training set
    return ngram_counts, vocab, total_unigrams

def backoff_prob(word, context, ngram_counts, total_unigrams, alpha=0.4): #calculate the probability of a word given a context using backoff
    n = len(context) + 1 # length of the context
    ngram = context + (word,)

    if n == 1:
        return ngram_counts[1].get((word,), 0) / total_unigrams

    if ngram_counts[n].get(ngram, 0) > 0:
        return ngram_counts[n][ngram] / ngram_counts[n - 1].get(context, 0)
    else:
        return alpha * backoff_prob(word, context[1:], ngram_counts, total_unigrams, alpha) 

def sentence_perplexity(sent, ngram_counts, total_unigrams, n, alpha=0.4):
    log_prob_sum = 0
    word_count = 0
    padded = ['<START>'] + sent + ['<END>']
    for i in range(1, len(padded)):
        word = padded[i]
        context = tuple(padded[max(0, i - n + 1):i])
        p = backoff_prob(word, context, ngram_counts, total_unigrams, alpha)
        if p > 0:
            log_prob_sum += math.log2(p)
        else:
            log_prob_sum += -float('inf')
        word_count += 1
    return 2 ** (-log_prob_sum / word_count)

def perplexity(sentences, ngram_counts, total_unigrams, n, alpha=0.4):
    log_prob_sum = 0
    word_count = 0
    for sent in sentences:
        padded = ['<START>'] + sent + ['<END>']
        for i in range(1, len(padded)):
            word = padded[i]
            context = tuple(padded[max(0, i - n + 1):i])
            p = backoff_prob(word, context, ngram_counts, total_unigrams, alpha)
            if p > 0:
                log_prob_sum += math.log2(p)
            else:
                log_prob_sum += -float('inf')
            word_count += 1
    return 2 ** (-log_prob_sum / word_count)


if __name__ == '__main__':

    #regression 
    df = preprocess('trip_adivser_data.csv')
    df = extract_features(df)
    X = df[FEATURE_COLUMNS]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = train_and_evaluate(X_train, X_test, y_train, y_test)

    #traing and test sentecne same as regression
    train_sents = df.loc[X_train.index, 'clean_review'].apply(str.split).tolist() #tokenized sentences for training
    test_sents = df.loc[X_test.index, 'clean_review'].apply(str.split).tolist()

    train_labels = y_train.tolist()
    test_labels = y_test.tolist()

    #e 
    # if label is 1 then goes to pos else to negatice 
    pos_train = [s for s, l in zip(train_sents, train_labels) if l == 1]
    neg_train = [s for s, l in zip(train_sents, train_labels) if l == 0]

    #creating ngram counts for positive and negative reviews
    pos_counts, pos_vocab, pos_total = build_ngram_counts(pos_train, n=3, min_freq=1)
    neg_counts, neg_vocab, neg_total = build_ngram_counts(neg_train, n=3, min_freq=1)

    combined_vocab = pos_vocab | neg_vocab

    preds = []
    for sent in test_sents:
        prepared = [w if w in combined_vocab else '<UNK>' for w in sent] #replace all words in testing which are not in the training set with <UNK>
        pp_pos = perplexity([prepared], pos_counts, pos_total, n=3, alpha=0.4) #perplexity for positive reviews
        pp_neg = perplexity([prepared], neg_counts, neg_total, n=3, alpha=0.4) #perplexity for negative reviews
        preds.append(1 if pp_pos <= pp_neg else 0) #if positive pp is lower than negative pp then it is positive else negative

    print("==== Perplexity Classifier Results ====")
    print(f"Accuracy: {accuracy_score(test_labels, preds):.4f}")
    print(classification_report(test_labels, preds, target_names=['Negative', 'Positive']))

