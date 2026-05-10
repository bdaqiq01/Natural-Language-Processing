# CSCI 682
# Assignment 3 Question 1 Skeleton
# Named Entity Recognition with HMMs and Independent Token Classifiers
# Developed with AI assistance from OpenAI Codex

import math

from gensim.downloader import load as loadKeyedVectors

# Special tokens for sequence boundaries and out-of-vocabulary words.
SOS_TOKEN = "<S>"
EOS_TOKEN = "</S>"
UNK_TOKEN = "<UNK>"


#################
### Embeddings ##
#################


# Load the embeddings from GloVe wiki + Gigaword 100-dimensional vectors.
def loadGloveWikiGigaword100():
  return loadKeyedVectors("glove-wiki-gigaword-100")


#################
### Load Data ###
#################
# Parse one raw CoNLL-2003 file into token and BIO label sequences
# filePath - path to one split file
# labelToId - mapping from BIO label names to integers
# return - a list of sentence dictionaries
def parseConllFile(filePath, labelToId):
  examples = []
  with open(filePath, encoding="utf-8") as textStream:
    tokens = []
    labels = []
    for rawLine in textStream:
      line = rawLine.strip()
      if not line or line.startswith("-DOCSTART-"):
        if tokens:
          examples.append({"tokens": tokens, "ner_tags": labels})
          tokens = []
          labels = []
        continue

      fields = line.split()
      tokens.append(fields[0])
      labels.append(labelToId[fields[-1]])

    if tokens:
      examples.append({"tokens": tokens, "ner_tags": labels})
  return examples


# Load the local CoNLL-2003 train, validation, and test splits
# datasetDirectory - directory containing train.txt, valid.txt, and test.txt
# return - dataset dictionary with splits and BIO label names
def loadConll2003Dataset(datasetDirectory):
  labelNames = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
  labelToId = {labelName: labelIndex for labelIndex, labelName in enumerate(labelNames)}
  return {
    "train": parseConllFile(f"{datasetDirectory}/train.txt", labelToId),
    "validation": parseConllFile(f"{datasetDirectory}/valid.txt", labelToId),
    "test": parseConllFile(f"{datasetDirectory}/test.txt", labelToId),
    "labelNames": labelNames,
  }


# Build a short summary of the dataset splits
# dataset - dictionary returned by loadConll2003Dataset
# return - a printable summary string
def formatDatasetSummary(dataset):
  return "\n".join([
    f"train sentences: {len(dataset['train'])}",
    f"validation sentences: {len(dataset['validation'])}",
    f"test sentences: {len(dataset['test'])}",
    f"BIO labels: {', '.join(dataset['labelNames'])}",
  ])


# CoNLL token -> GloVe key if present, else UNK.
def mapTokenForGlove(token, embedding):
  token = token.lower()
  return token if token in embedding else UNK_TOKEN


# For every split: GloVe OOV -> UNK, wrap tokens with <S> / </S>
def preprocessDatasetForGlove(dataset, embedding):
  
  labelToId = {name: i for i, name in enumerate(dataset["labelNames"])} #BIO label to class Id 
  oId = labelToId["O"]  #outside the name span 

  processed = {"labelNames": dataset["labelNames"]} #will hold the processed dataset

  for split in ("train", "validation", "test"):
    processed_sentences = []
    for sent in dataset[split]:
      mapped = [mapTokenForGlove(token, embedding) for token in sent["tokens"]]
      processed_sentences.append({
        "tokens": [SOS_TOKEN] + mapped + [EOS_TOKEN],
        "ner_tags": [oId] + list(sent["ner_tags"]) + [oId],
      })
    processed[split] = processed_sentences
  return processed


##################
### HMM tagging ###
##################
# A[i][j] = P(next tag j | previous tag i). B[s][w_ix] = P(word w | tag s). pi[s] = P(first tag is s).


def _stripSentenceBoundaries(example):
  """Drop SOS/EOS tokens and matching boundary tags if present."""
  tokens, tags = example["tokens"], example["ner_tags"]
  if tokens and tokens[0] == SOS_TOKEN and tokens[-1] == EOS_TOKEN:
    return tokens[1:-1], tags[1:-1]
  return tokens, tags


def trainHmm(train_examples, label_names, smoothing=1.0):
  """Return pi, A, B with add-`smoothing` (Laplace): plain probabilities, not logs."""
  alpha = smoothing
  n_tags = len(label_names)

  stripped = [_stripSentenceBoundaries(ex) for ex in train_examples]
  stripped = [(t, y) for t, y in stripped if t]

  vocab = {UNK_TOKEN}
  for tokens, _ in stripped:
    vocab.update(tokens)
  vocab_list = sorted(vocab)
  vocab_size = len(vocab_list)
  word_to_ix = {w: i for i, w in enumerate(vocab_list)}

  pi_counts = [alpha] * n_tags
  trans_counts = [[alpha] * n_tags for _ in range(n_tags)]
  emit_counts = [[alpha] * vocab_size for _ in range(n_tags)]

  for tokens, tags in stripped:
    pi_counts[tags[0]] += 1
    for y, w in zip(tags, tokens):
      emit_counts[y][word_to_ix[w]] += 1
    for y_prev, y_next in zip(tags, tags[1:]):
      trans_counts[y_prev][y_next] += 1

  pi_den = sum(pi_counts)
  pi = [c / pi_den for c in pi_counts]

  A = []
  for row in trans_counts:
    den = sum(row)
    A.append([c / den for c in row])

  B = []
  for row in emit_counts:
    den = sum(row)
    B.append([c / den for c in row])

  return {
    "label_names": label_names,
    "n_tags": n_tags,
    "pi": pi,
    "A": A,
    "B": B,
    "word_to_ix": word_to_ix,
    "vocab_size": vocab_size,
  }


def viterbiDecode(words, model):
  """Most likely tag sequence; uses log internally so products of tiny probs do not underflow."""
  if not words:
    return []

  n_tags = model["n_tags"]
  pi = model["pi"]
  A = model["A"]
  B = model["B"]
  word_to_ix = model["word_to_ix"]
  unk_ix = word_to_ix[UNK_TOKEN]

  log_pi = [math.log(p) for p in pi]
  log_A = [[math.log(a) for a in row] for row in A]
  log_B = [[math.log(b) for b in row] for row in B]

  obs_ix = [word_to_ix[w] if w in word_to_ix else unk_ix for w in words]
  T = len(obs_ix)

  dp = [[-math.inf] * n_tags for _ in range(T)]
  back = [[0] * n_tags for _ in range(T)]

  for s in range(n_tags):
    dp[0][s] = log_pi[s] + log_B[s][obs_ix[0]]

  for t in range(1, T):
    for s in range(n_tags):
      best_val = -math.inf
      best_prev = 0
      emit = log_B[s][obs_ix[t]]
      for sp in range(n_tags):
        cand = dp[t - 1][sp] + log_A[sp][s] + emit
        if cand > best_val:
          best_val = cand
          best_prev = sp
      dp[t][s] = best_val
      back[t][s] = best_prev

  best_last = max(range(n_tags), key=lambda s: dp[T - 1][s])
  path = [0] * T
  path[T - 1] = best_last
  for t in range(T - 2, -1, -1):
    path[t] = back[t + 1][path[t + 1]]
  return path


def tokenAccuracy(predicted_tags, gold_tags):
  if not predicted_tags:
    return 0.0
  correct = sum(int(p == g) for p, g in zip(predicted_tags, gold_tags))
  return correct / len(predicted_tags)


def evaluateHmmTagger(model, examples):
  """Mean token accuracy over sentences (after stripping SOS/EOS)."""
  total_correct = 0
  total_tokens = 0
  for ex in examples:
    tokens, gold = _stripSentenceBoundaries(ex)
    if not tokens:
      continue
    pred = viterbiDecode(tokens, model)
    total_correct += sum(int(p == g) for p, g in zip(pred, gold))
    total_tokens += len(gold)
  return total_correct / total_tokens if total_tokens else 0.0


############
### Main ###
############
# Load the dataset and print a short summary
def main():
  datasetDirectory = "data/conll2003"

  dataset = loadConll2003Dataset(datasetDirectory) #conll2003 dic training, validation, test, labeNames (O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC)
  print(formatDatasetSummary(dataset))

  GloVE = loadGloveWikiGigaword100()  #embdedding of words

  preprocessed = preprocessDatasetForGlove(dataset, GloVE)

  hmm = trainHmm(preprocessed["train"], preprocessed["labelNames"])
  val_acc = evaluateHmmTagger(hmm, preprocessed["validation"])
  test_acc = evaluateHmmTagger(hmm, preprocessed["test"])
  print(f"HMM token accuracy (validation): {val_acc:.4f}")
  print(f"HMM token accuracy (test): {test_acc:.4f}")


if __name__ == "__main__":
  main()
