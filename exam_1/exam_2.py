import numpy as np
import pandas as pd
import gensim.downloader as api
from collections import defaultdict
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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


def prepare_splits(categories, glove):
    """Flatten analogies, filter to vocab, stratified split 70/15/15."""
    all_analogies = []
    for cat_idx, (cat, analogies) in enumerate(categories.items()):
        for a, b, c, d in analogies:
            if all(w in glove for w in (a, b, c, d)):
                all_analogies.append((a, b, c, d, cat_idx))

    labels = [x[4] for x in all_analogies]
    train_data, temp_data = train_test_split(
        all_analogies, test_size=0.30, random_state=42, stratify=labels
    )
    temp_labels = [x[4] for x in temp_data]
    val_data, test_data = train_test_split(
        temp_data, test_size=0.50, random_state=42, stratify=temp_labels
    )
    return train_data, val_data, test_data


def build_tensors(glove, data):
    """Convert (a,b,c,d,cat) tuples into X=concat(a,b,c) and Y=d tensors."""
    X = np.array([np.concatenate([glove[a], glove[b], glove[c]]) for a, b, c, d, _ in data])
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


def compute_accuracy(model_nn, X, data, glove):
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

    correct = 0
    for i, (a, b, c, d, _) in enumerate(data):
        exclude = {glove.key_to_index[w] for w in (a, b, c)}
        sims = similarities[i].copy()
        for idx in exclude:
            sims[idx] = -1
        best_idx = np.argmax(sims)
        if all_words[best_idx] == d:
            correct += 1

    return correct / len(data) * 100


if __name__ == '__main__':
    categories = load_analogies('questions-words.txt')

    model_glove50 = api.load('glove-wiki-gigaword-50')
    model_glove100 = api.load('glove-wiki-gigaword-100')

    results_50 = evaluate_analogies(model_glove50, categories)
    results_100 = evaluate_analogies(model_glove100, categories)
    #a
    print_table(results_50, results_100)

    #b - feedforward neural network for analogy task
    glove = model_glove100
    dim = glove.vector_size

    train_data, val_data, test_data = prepare_splits(categories, glove)
    print(f"\nSplit sizes: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    X_train, Y_train = build_tensors(glove, train_data)
    X_val, Y_val = build_tensors(glove, val_data)
    X_test, Y_test = build_tensors(glove, test_data)

    # No hidden layer — simple linear baseline
    model_nn = nn.Sequential(nn.Linear(3 * dim, dim))
    model_nn = train_model(model_nn, X_train, Y_train)

    val_acc = compute_accuracy(model_nn, X_val, val_data, glove)
    print(f"\nNeural network validation accuracy: {val_acc:.2f}%")
