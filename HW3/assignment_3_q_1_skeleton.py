# CSCI 682
# Assignment 3 Question 1 Skeleton
# Named Entity Recognition with HMMs and Independent Token Classifiers
# Developed with AI assistance from OpenAI Codex

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


############
### Main ###
############
# Load the dataset and print a short summary
def main():
  datasetDirectory = "data/conll2003"
  dataset = loadConll2003Dataset(datasetDirectory)
  print(formatDatasetSummary(dataset))


if __name__ == "__main__":
  main()
