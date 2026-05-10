# CSCI 682
# Assignment 3 Question 2 Skeleton
# Transition-Based Dependency Parsing
# Developed with AI assistance from OpenAI Codex

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

#################
### Load Data ###
#################

# Parse one CoNLL-U file into sentence records
# datasetDirectory - directory containing the local UD files
# fileName - one split file name
# lowercaseWords - whether to lowercase tokens during parsing
# return - list of parsed sentence dictionaries
def parseConlluFile(datasetDirectory, fileName, lowercaseWords=True):
  filePath = f"{datasetDirectory}/{fileName}"
  sentences = []
  comments = {}
  words = ["<ROOT>"]
  posTags = ["<ROOT_POS>"]
  heads = [-1]
  relations = ["<ROOT_REL>"]

  with open(filePath, encoding="utf-8") as textStream:
    for rawLine in textStream:
      line = rawLine.rstrip("\n")
      if not line:
        if len(words) > 1:
          sentences.append(buildSentenceRecord(words, posTags, heads, relations, comments))
        comments = {}
        words = ["<ROOT>"]
        posTags = ["<ROOT_POS>"]
        heads = [-1]
        relations = ["<ROOT_REL>"]
        continue

      if line.startswith("#"):
        if " = " in line:
          commentKey, commentValue = line[1:].split(" = ", 1)
          comments[commentKey.strip()] = commentValue.strip()
        continue

      fields = line.split("\t")
      tokenId = fields[0]
      if "-" in tokenId or "." in tokenId:
        continue

      words.append(fields[1].lower() if lowercaseWords else fields[1])
      posTags.append(fields[3])
      heads.append(int(fields[6]))
      relations.append(fields[7])

  if len(words) > 1:
    sentences.append(buildSentenceRecord(words, posTags, heads, relations, comments))
  return sentences


# Build one normalized sentence record
# words - token list including the root token
# posTags - POS tags including the root tag
# heads - gold head indices including the root placeholder
# relations - gold dependency relations including the root relation
# comments - metadata collected from CoNLL-U comments
# return - one sentence dictionary
def buildSentenceRecord(words, posTags, heads, relations, comments):
  return {
    "words": list(words),
    "posTags": list(posTags),
    "heads": list(heads),
    "relations": list(relations),
    "length": len(words) - 1,
    "text": comments.get("text", " ".join(words[1:])),
    "sentId": comments.get("sent_id", ""),
  }


# Load the local UD train, dev, and test splits
# datasetDirectory - directory containing the local UD files
# trainFile - training split filename
# devFile - development split filename
# testFile - test split filename
# lowercaseWords - whether to lowercase tokens during parsing
# return - dataset dictionary with three splits
def loadUdDataset(datasetDirectory, trainFile, devFile, testFile, lowercaseWords=True):
  return {
    "training": parseConlluFile(datasetDirectory, trainFile, lowercaseWords),
    "dev": parseConlluFile(datasetDirectory, devFile, lowercaseWords),
    "test": parseConlluFile(datasetDirectory, testFile, lowercaseWords),
  }


#########################
### Projective Filter ###
#########################

# Check whether one gold dependency tree is projective
# sentence - one parsed UD sentence
# return - True if the tree is projective, else False
def isProjective(sentence):
  arcs = []
  for dependentId in range(1, sentence["length"] + 1):
    headId = sentence["heads"][dependentId]
    left = min(headId, dependentId)
    right = max(headId, dependentId)
    arcs.append((left, right, headId, dependentId))

  for arcIndex, firstArc in enumerate(arcs):
    firstLeft, firstRight, firstHead, firstDependent = firstArc
    for secondArc in arcs[arcIndex + 1:]:
      secondLeft, secondRight, secondHead, secondDependent = secondArc
      if len({firstHead, firstDependent, secondHead, secondDependent}) < 4:
        continue
      if firstLeft < secondLeft < firstRight < secondRight:
        return False
      if secondLeft < firstLeft < secondRight < firstRight:
        return False
  return True


# Keep only the projective sentences in one split
# split - list of parsed sentences
# return - kept sentences and counts
def filterProjectiveSentences(split):
  projectiveSentences = [sentence for sentence in split if isProjective(sentence)]
  return {
    "sentences": projectiveSentences,
    "keptCount": len(projectiveSentences),
    "droppedCount": len(split) - len(projectiveSentences),
  }


####################
### Parser State ###
####################

# Build the initial parser state for one sentence
# sentence - one parsed UD sentence
# return - initial arc-standard parser state
def buildInitialParserState(sentence):
  return {
    "stack": [0],
    "buffer": list(range(1, sentence["length"] + 1)),
    "predictedHeads": [-1] + [None] * sentence["length"],
    "arcs": [],
  }


# Check whether parsing is complete
# state - current parser state
# return - True if the parse is complete, else False
def isTerminalState(state):
  return not state["buffer"] and state["stack"] == [0]


# List the valid transitions for the current state
# state - current parser state
# return - list of valid transition names
def getValidTransitions(state):
  transitions = []
  if state["buffer"]:
    transitions.append("SHIFT")
  if len(state["stack"]) >= 2:
    if state["stack"][-2] != 0:
      transitions.append("LEFT_ARC")
    transitions.append("RIGHT_ARC")
  return transitions


# Copy a parser state
# state - current parser state
# return - independent copy of the state
def copyParserState(state):
  return {
    "stack": list(state["stack"]),
    "buffer": list(state["buffer"]),
    "predictedHeads": list(state["predictedHeads"]),
    "arcs": list(state["arcs"]),
  }


# Apply one transition to a parser state
# state - current parser state
# transition - transition to apply
# return - next parser state
def applyTransition(state, transition):
  nextState = copyParserState(state)

  if transition == "SHIFT":
    nextState["stack"].append(nextState["buffer"].pop(0))
    return nextState

  stackTop = nextState["stack"][-1]
  stackSecond = nextState["stack"][-2]

  if transition == "LEFT_ARC":
    nextState["predictedHeads"][stackSecond] = stackTop
    nextState["arcs"].append((stackTop, stackSecond))
    del nextState["stack"][-2]
    return nextState

  if transition == "RIGHT_ARC":
    nextState["predictedHeads"][stackTop] = stackSecond
    nextState["arcs"].append((stackSecond, stackTop))
    nextState["stack"].pop()
    return nextState

  raise ValueError(f"Unknown transition: {transition}")


########################
### Gold Transitions ###
########################

# Build the gold child list for each head token
# sentence - one parsed UD sentence
# return - list of child-id lists
def buildGoldChildren(sentence):
  childLists = [[] for _ in range(sentence["length"] + 1)]
  for dependentId in range(1, sentence["length"] + 1):
    childLists[sentence["heads"][dependentId]].append(dependentId)
  return childLists


# Check whether all gold children of a token have been attached
# tokenId - token whose children we are checking
# state - current parser state
# goldChildren - gold child lists for the sentence
# return - True if all gold children are attached, else False
def allChildrenAttached(tokenId, state, goldChildren):
  return all(state["predictedHeads"][childId] is not None for childId in goldChildren[tokenId])


# Choose the gold next transition under a static oracle
# state - current parser state
# sentence - one parsed UD sentence
# goldChildren - gold child lists for the sentence
# return - gold transition label or None
def getGoldTransition(state, sentence, goldChildren):
  if len(state["stack"]) >= 2:
    stackTop = state["stack"][-1]
    stackSecond = state["stack"][-2]

    if stackSecond != 0 and sentence["heads"][stackSecond] == stackTop:
      if allChildrenAttached(stackSecond, state, goldChildren):
        return "LEFT_ARC"

    if sentence["heads"][stackTop] == stackSecond:
      if allChildrenAttached(stackTop, state, goldChildren):
        return "RIGHT_ARC"

  if state["buffer"]:
    return "SHIFT"
  return None


# Simulate the gold transition sequence for one sentence
# sentence - one parsed UD sentence
# return - parser states and gold transitions, or None if the oracle fails
def simulateGoldTransitions(sentence):
  state = buildInitialParserState(sentence)
  goldChildren = buildGoldChildren(sentence)
  stateExamples = []
  transitions = []

  while not isTerminalState(state):
    transition = getGoldTransition(state, sentence, goldChildren)
    if transition is None:
      return None
    stateExamples.append(copyParserState(state))
    transitions.append(transition)
    state = applyTransition(state, transition)

  return {
    "states": stateExamples,
    "transitions": transitions,
  }


##########################
### Feature Extraction ###
##########################

# Get the word for one token id, with a null fallback
# sentence - one parsed UD sentence
# tokenId - token index or None
# return - token word or a null marker
def getWord(sentence, tokenId):
  if tokenId is None:
    return "<NULL>"
  return sentence["words"][tokenId]


# Get the POS tag for one token id, with a null fallback
# sentence - one parsed UD sentence
# tokenId - token index or None
# return - token POS tag or a null marker
def getPosTag(sentence, tokenId):
  if tokenId is None:
    return "<NULL_POS>"
  return sentence["posTags"][tokenId]


# Read a token id from the top of the stack
# state - current parser state
# relativeIndex - 0 for the top item, 1 for the next item, and so on
# return - token id or None
def getStackToken(state, relativeIndex):
  if len(state["stack"]) > relativeIndex:
    return state["stack"][-1 - relativeIndex]
  return None


# Read a token id from the buffer
# state - current parser state
# bufferIndex - index into the buffer
# return - token id or None
def getBufferToken(state, bufferIndex):
  if len(state["buffer"]) > bufferIndex:
    return state["buffer"][bufferIndex]
  return None


# Build features for one parser state
# state - current parser state
# sentence - one parsed UD sentence
# return - sparse feature dictionary
def extractParserFeatures(state, sentence):
  # TODO: complete this function for Question 2 Part 1.
  # At a minimum, include:
  # 1. the word and POS tag for the top two items on the stack
  # 2. the word and POS tag for the first two items in the buffer
  # You may add any other parser-state features you think are useful.
  pass


# Collect supervised transition examples from one split
# split - list of parsed projective sentences
# return - feature dictionaries and labels
def collectTransitionExamples(split):
  # Each call to simulateGoldTransitions converts one gold dependency tree
  # into parser states paired with gold next-transition labels.
  featureRows = []
  labels = []

  for sentence in split:
    oracleResult = simulateGoldTransitions(sentence)
    if oracleResult is None:
      continue
    for state, transition in zip(oracleResult["states"], oracleResult["transitions"]):
      featureRows.append(extractParserFeatures(state, sentence))
      labels.append(transition)

  return {
    "features": featureRows,
    "labels": labels,
  }


########################
### Train Classifier ###
########################

# Fit the transition classifier
# trainingFeatures - parser-state feature dictionaries
# trainingLabels - gold transition labels
# cValue - inverse regularization strength for logistic regression
# config - configuration dictionary
# return - fitted vectorizer and classifier
def fitTransitionClassifier(trainingFeatures, trainingLabels, cValue, config):
  vectorizer = DictVectorizer(sparse=True)
  trainingMatrix = vectorizer.fit_transform(trainingFeatures)
  classifier = LogisticRegression(
    C=cValue,
    max_iter=config["classifierMaxIter"],
    tol=1e-3,
    random_state=config["randomState"],
  )
  classifier.fit(trainingMatrix, trainingLabels)
  return {
    "vectorizer": vectorizer,
    "classifier": classifier,
  }


# Choose the best C value on the dev set
# trainingSplit - projective training sentences
# devSplit - projective dev sentences
# config - configuration dictionary
# return - best model bundle and dev-set results
def selectTransitionClassifier(trainingSplit, devSplit, config):
  # TODO: complete this function for Question 2 Part 2.
  # Recommended steps:
  # 1. call collectTransitionExamples on the training and dev splits
  # 2. train one classifier for each C value in config["cCandidates"]
  # 3. use the fitted DictVectorizer to transform the dev examples
  # 4. evaluate each model on the dev transition examples
  # 5. keep the best-performing model and return the full set of results
  pass


#######################
### Greedy Decoding ###
#######################

# Predict the best valid transition for the current state
# state - current parser state
# sentence - one parsed UD sentence
# modelBundle - fitted vectorizer and classifier
# return - predicted transition label
def predictBestTransition(state, sentence, modelBundle):
  # TODO: complete this function for Question 2 Part 3.
  # Recommended steps:
  # 1. extract features for the current parser state
  # 2. transform those features with the trained DictVectorizer
  # 3. score all transitions with the classifier
  # 4. call getValidTransitions(state) to find the legal transitions
  # 5. return the highest-scoring transition that is valid in the current state
  pass


# Greedily decode one sentence
# sentence - one parsed UD sentence
# modelBundle - fitted vectorizer and classifier
# return - predicted heads and transition sequence
def greedyDecodeSentence(sentence, modelBundle):
  state = buildInitialParserState(sentence)
  transitions = []

  while not isTerminalState(state):
    transition = predictBestTransition(state, sentence, modelBundle)
    transitions.append(transition)
    state = applyTransition(state, transition)

  return {
    "predictedHeads": state["predictedHeads"],
    "transitions": transitions,
  }


# Greedily decode one full split
# split - list of parsed sentences
# modelBundle - fitted vectorizer and classifier
# return - list of predicted parses
def parseSentenceSplit(split, modelBundle):
  return [greedyDecodeSentence(sentence, modelBundle) for sentence in split]


##################
### Evaluation ###
##################

# Evaluate predicted parses against the gold trees
# sentences - gold parsed sentences
# predictedParses - predicted parse records
# return - UAS, exact match, and token count
def evaluateParses(sentences, predictedParses):
  totalTokens = 0
  correctHeads = 0
  exactMatches = 0

  for sentence, predictedParse in zip(sentences, predictedParses):
    sentenceCorrect = True
    for tokenId in range(1, sentence["length"] + 1):
      totalTokens += 1
      if predictedParse["predictedHeads"][tokenId] == sentence["heads"][tokenId]:
        correctHeads += 1
      else:
        sentenceCorrect = False
    if sentenceCorrect:
      exactMatches += 1

  return {
    "uas": correctHeads / totalTokens if totalTokens else 0.0,
    "exactMatch": exactMatches / len(sentences) if sentences else 0.0,
    "totalTokens": totalTokens,
  }


# Format one projective-filter summary line
# splitName - name of the split
# summary - output of filterProjectiveSentences
# return - printable summary line
def formatProjectiveSummary(splitName, summary):
  return (
    f"{splitName}: kept {summary['keptCount']} projective sentences, "
    f"dropped {summary['droppedCount']} non-projective sentences"
  )


# Format one parse-evaluation line
# splitName - name of the split
# metrics - parse evaluation metrics
# return - printable summary line
def formatParseMetrics(splitName, metrics):
  return (
    f"{splitName}: "
    f"UAS={metrics['uas']:.4f}, "
    f"exactMatch={metrics['exactMatch']:.4f}, "
    f"tokens={metrics['totalTokens']}"
  )


############
### Main ###
############

# Load the dataset, train the parser, and print the results
# return - none
def main():
  datasetDirectory = "data/ud_english_ewt"
  config = {
    "cCandidates": [0.25, 1.0, 4.0],
    "classifierMaxIter": 500,
    "randomState": 42,
  }
  dataset = loadUdDataset(
    datasetDirectory,
    "en_ewt-ud-train.conllu",
    "en_ewt-ud-dev.conllu",
    "en_ewt-ud-test.conllu",
    True,
  )

  trainingSummary = filterProjectiveSentences(dataset["training"])
  devSummary = filterProjectiveSentences(dataset["dev"])
  testSummary = filterProjectiveSentences(dataset["test"])

  print("Projective filtering summary:")
  print(formatProjectiveSummary("train", trainingSummary))
  print(formatProjectiveSummary("dev", devSummary))
  print(formatProjectiveSummary("test", testSummary))
  print()

  selectionResults = selectTransitionClassifier(trainingSummary["sentences"], devSummary["sentences"], config)
  bestModel = selectionResults["bestResult"]["modelBundle"]
  testParses = parseSentenceSplit(testSummary["sentences"], bestModel)
  testMetrics = evaluateParses(testSummary["sentences"], testParses)

  print(f"Selected C: {selectionResults['bestResult']['cValue']}")
  print(formatParseMetrics("test", testMetrics))


if __name__ == "__main__":
  main()
