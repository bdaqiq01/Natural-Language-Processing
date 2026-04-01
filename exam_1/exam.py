import numpy as np
import pandas as pd
import re
import math
import gensim.downloader as api
from nltk.corpus import opinion_lexicon
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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


def load_analogies(filepath):
    """Parse questions-words.txt into {category: [(a, b, c, d), ...]}."""
    categories = defaultdict(list)
    current = None
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith(':'):
                current = line[2:]
            else:
                parts = line.lower().split()
                if len(parts) == 4:
                    categories[current].append(tuple(parts))
    return categories


def evaluate_analogies(model, categories):
    """Part (a): evaluate using vector arithmetic b - a + c."""
    results = {}
    total_correct = 0
    total_count = 0

    for cat, analogies in categories.items():
        correct = 0
        count = 0
        for a, b, c, d in analogies:
            if not all(w in model for w in (a, b, c, d)):
                continue
            count += 1
            predicted = model.most_similar(positive=[b, c], negative=[a], topn=1)[0][0]
            if predicted == d:
                correct += 1
        acc = correct / count if count > 0 else 0.0
        results[cat] = (correct, count, acc)
        total_correct += correct
        total_count += count

    overall_acc = total_correct / total_count if total_count > 0 else 0.0
    results['Overall'] = (total_correct, total_count, overall_acc)
    return results


def print_table(results_50, results_100):
    df = pd.DataFrame({
        'Category': list(results_50.keys()),
        'glove-50': [results_50[cat][2] * 100 for cat in results_50.keys()],
        'glove-100': [results_100[cat][2] * 100 for cat in results_100.keys()]
    })
    print(df.to_string(index=False))


def prepare_splits(categories, glove): # the catagoes[] is a diction whire the key is the cat name and the value is a list of 4 words analogy 

    all_analogies = [] #each row will be a, b, c, d, cat_idx
    for cat_idx, (cat, analogies) in enumerate(categories.items()):
        for a, b, c, d in analogies: #throgh eah anal in the catagory
            if all(w in glove for w in (a, b, c, d)): #check if all the words are in the glove model
                all_analogies.append((a, b, c, d, cat_idx)) #append the analogy to the list

    cat_labels= [x[4] for x in all_analogies] #the label is the cat_idx
    train_data, temp_data = train_test_split(
        all_analogies, test_size=0.30, random_state=42, stratify=cat_labels #make sure the split has same distribution for each catagory
    )
    temp_cat_labels = [x[4] for x in temp_data]

    val_data, test_data = train_test_split(
        temp_data, test_size=0.50, random_state=42, stratify=temp_cat_labels
    )
    return train_data, val_data, test_data #return the train, val, and test data


def build_tensors(glove, data):

    X = np.array([np.concatenate([glove[a], glove[b], glove[c]]) for a, b, c, d, _ in data]) #feature arrray of a, b , c
    Y = np.array([glove[d] for _, _, _, d, _ in data])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)


def train_model(model_nn, X_train, Y_train, epochs=20, lr=0.001, batch_size=256):
    """Train the neural network with MSE loss."""
    optimizer = torch.optim.Adam(model_nn.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(
        TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True
    )

    for epoch in range(epochs):
        model_nn.train()
        epoch_loss = 0
        for X_batch, Y_batch in train_loader:
            pred = model_nn(X_batch)
            loss = loss_fn(pred, Y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)
        avg_loss = epoch_loss / len(X_train)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}  loss={avg_loss:.6f}")

    return model_nn


def compute_accuracy(model_nn, X, data, glove, cat_names=None):
    """Predict d vectors, find nearest vocab word, compare to true d."""
    model_nn.eval()
    all_words = glove.index_to_key
    all_vecs = glove.vectors
    norms = np.linalg.norm(all_vecs, axis=1, keepdims=True)
    normed_vecs = all_vecs / norms

    with torch.no_grad():
        preds = model_nn(X).numpy()

    pred_norms = np.linalg.norm(preds, axis=1, keepdims=True)
    normed_preds = preds / pred_norms
    similarities = normed_preds @ normed_vecs.T

    cat_correct = defaultdict(int)
    cat_count = defaultdict(int)

    for i, (a, b, c, d, cat_idx) in enumerate(data):
        exclude = {glove.key_to_index[w] for w in (a, b, c)}
        sims = similarities[i].copy()
        for idx in exclude:
            sims[idx] = -1
        best_idx = np.argmax(sims)
        cat_count[cat_idx] += 1
        if all_words[best_idx] == d:
            cat_correct[cat_idx] += 1

    results = {}
    for cat_idx in sorted(cat_count.keys()):
        name = cat_names[cat_idx] if cat_names else str(cat_idx)
        acc = cat_correct[cat_idx] / cat_count[cat_idx] * 100
        results[name] = acc

    total_correct = sum(cat_correct.values())
    total_count = sum(cat_count.values())
    results['Overall'] = total_correct / total_count * 100
    return results


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

    categories = load_analogies('questions-words.txt')

    model_glove50 = api.load('glove-wiki-gigaword-50')
    model_glove100 = api.load('glove-wiki-gigaword-100')

    results_50 = evaluate_analogies(model_glove50, categories)
    results_100 = evaluate_analogies(model_glove100, categories)
    #a
    print_table(results_50, results_100)

    #b - feedforward neural network for analogy task
    cat_names = list(categories.keys())

    #100 
    dim100 = model_glove100.vector_size
    train_data100, val_data100, test_data100 = prepare_splits(categories, model_glove100)

    X_train100, Y_train100 = build_tensors(model_glove100, train_data100)
    X_val100, Y_val100 = build_tensors(model_glove100, val_data100)

    linear100 = nn.Sequential(nn.Linear(3 * dim100, dim100))
    linear100 = train_model(linear100, X_train100, Y_train100)

    deep100 = nn.Sequential(
        nn.Linear(3 * dim100, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, dim100)
    )
    deep100 = train_model(deep100, X_train100, Y_train100)

    #50 
    dim50 = model_glove50.vector_size
    train_data50, val_data50, test_data50 = prepare_splits(categories, model_glove50)

    X_train50, Y_train50 = build_tensors(model_glove50, train_data50)
    X_val50, Y_val50 = build_tensors(model_glove50, val_data50)

    linear50 = nn.Sequential(nn.Linear(3 * dim50, dim50))
    linear50 = train_model(linear50, X_train50, Y_train50)

    deep50 = nn.Sequential(
        nn.Linear(3 * dim50, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, dim50)
    )
    deep50 = train_model(deep50, X_train50, Y_train50)

    #eval
    val_linear100 = compute_accuracy(linear100, X_val100, val_data100, model_glove100, cat_names)
    val_linear50 = compute_accuracy(linear50, X_val50, val_data50, model_glove50, cat_names)

    X_test100, Y_test100 = build_tensors(model_glove100, test_data100)
    X_test50, Y_test50 = build_tensors(model_glove50, test_data50)

    test_deep100 = compute_accuracy(deep100, X_test100, test_data100, model_glove100, cat_names)
    test_deep50 = compute_accuracy(deep50, X_test50, test_data50, model_glove50, cat_names)

    print("\nNN Validation Accuracy (per category):")
    df = pd.DataFrame({
        'Category': list(val_linear100.keys()),
        'Linear-50': [f"{val_linear50[cat]:.2f}" for cat in val_linear50],
        'Linear-100': [f"{val_linear100[cat]:.2f}" for cat in val_linear100],
    })
    print(df.to_string(index=False))

    print("\n Deep NN testing Accuracy (per category):")
    df = pd.DataFrame({
        'Category': list(test_deep100.keys()),
        'Deep-50': [f"{test_deep50[cat]:.2f}" for cat in test_deep50],
        'Deep-100': [f"{test_deep100[cat]:.2f}" for cat in test_deep100],
    })
    print(df.to_string(index=False))
