"""Word and sentence attention with a regression head for essay scoring."""

import re
import math
import random
from collections import Counter
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from essay_scoring.data import Essay
from essay_scoring.metrics import evaluate_scores

MAX_SENTENCES = 48
MAX_WORDS = 64
PAD_ID = 0
UNKNOWN_ID = 1
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")
WORD = re.compile(r"@\w+|[^\W_]+(?:['’][^\W_]+)*")


def training_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def truncation_statistics(essays: list[Essay]) -> dict:
    affected = lost = total = chunked = 0
    for essay in essays:
        full = _sentences(essay.text, max_sentences=10**9, max_words=MAX_WORDS)
        lengths = [len(sentence) for sentence in full]
        total += sum(lengths)
        removed = sum(lengths[MAX_SENTENCES:])
        lost += removed
        affected += removed > 0
        chunked += any(len(WORD.findall(s.casefold())) > MAX_WORDS for s in SENTENCE_BREAK.split(essay.text))
    return {"samples": len(essays), "truncated_essays": affected,
            "truncated_fraction": affected / len(essays), "discarded_tokens": lost,
            "total_tokens": total, "discarded_fraction": lost / max(1, total),
            "essays_with_long_sentence_chunking": chunked}


def _sentences(text: str, max_sentences: int, max_words: int) -> list[list[str]]:
    sentences = []
    for sentence in SENTENCE_BREAK.split(text):
        words = WORD.findall(sentence.casefold())
        for start in range(0, len(words), max_words):
            sentences.append(words[start:start + max_words])
            if len(sentences) == max_sentences:
                return sentences
    return sentences


def _encode(sentences: list[list[str]], vocabulary: dict[str, int]) -> list[list[int]]:
    return [
        [vocabulary.get(word, UNKNOWN_ID) for word in sentence]
        for sentence in sentences
    ] or [[UNKNOWN_ID]]


def _pad_batch(essays: list[list[list[int]]]) -> Tensor:
    sentence_count = max(len(essay) for essay in essays)
    word_count = max(len(sentence) for essay in essays for sentence in essay)
    tokens = torch.full((len(essays), sentence_count, word_count), PAD_ID, dtype=torch.long)
    for essay_index, essay in enumerate(essays):
        for sentence_index, sentence in enumerate(essay):
            tokens[essay_index, sentence_index, :len(sentence)] = torch.tensor(sentence)
    return tokens


def _collate_training(batch):
    essays, targets = zip(*batch)
    return _pad_batch(essays), torch.tensor(targets, dtype=torch.float32)


class AttentionEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, inputs: Tensor, mask: Tensor) -> Tensor:
        lengths = mask.sum(dim=1).cpu()
        packed = pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
        packed_states, _ = self.gru(packed)
        states, _ = pad_packed_sequence(
            packed_states, batch_first=True, total_length=mask.size(1)
        )
        logits = self.attention(states).squeeze(-1).masked_fill(~mask, -torch.inf)
        weights = logits.softmax(dim=1).unsqueeze(-1)
        return (states * weights).sum(dim=1)


class HANRegressor(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_size: int = 64, hidden_size: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_size, padding_idx=PAD_ID)
        self.words = AttentionEncoder(embedding_size, hidden_size)
        self.sentences = AttentionEncoder(2 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.score = nn.Linear(2 * hidden_size, 1)

    def encode_document(self, tokens: Tensor) -> Tensor:
        word_mask = tokens.ne(PAD_ID)
        sentence_mask = word_mask.any(dim=-1)

        # Remove empty sentences before the word GRU so each sequence has a valid token.
        sentence_tokens = tokens[sentence_mask]
        sentence_states = self.words(self.embedding(sentence_tokens), word_mask[sentence_mask])
        documents = sentence_states.new_zeros((*sentence_mask.shape, sentence_states.size(-1)))
        documents[sentence_mask] = sentence_states
        return self.sentences(documents, sentence_mask)

    def forward(self, tokens: Tensor) -> Tensor:
        documents = self.encode_document(tokens)
        return torch.sigmoid(self.score(self.dropout(documents))).squeeze(-1)


def _prediction_loader(texts, vocabulary, max_sentences, max_words, batch_size):
    encoded = [_encode(_sentences(text, max_sentences, max_words), vocabulary) for text in texts]
    return DataLoader(encoded, batch_size=batch_size, collate_fn=_pad_batch)


@torch.inference_mode()
def _predict(model, loader, device, bounds) -> list[float]:
    model.eval()
    low, high = bounds
    predictions = []
    for tokens in loader:
        normalized = model(tokens.to(device))
        predictions.extend((low + normalized * (high - low)).cpu().tolist())
    return predictions


def fit_han(train: list[Essay], validation: list[Essay], bounds, *, epochs, batch_size, seed):
    if not train or not validation or epochs < 1 or batch_size < 1:
        raise ValueError("Training needs nonempty data and positive epochs/batch size.")
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    torch.manual_seed(seed)
    device = training_device()
    train_sentences = [_sentences(essay.text, MAX_SENTENCES, MAX_WORDS) for essay in train]
    counts = Counter(word for essay in train_sentences for sentence in essay for word in sentence)
    vocabulary = {word: index for index, (word, _) in enumerate(counts.most_common(30_000), start=2)}
    low, high = bounds
    train_data = [
        (_encode(sentences, vocabulary), (essay.score - low) / (high - low))
        for sentences, essay in zip(train_sentences, train, strict=True)
    ]
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=_collate_training
    )
    validation_loader = _prediction_loader(
        [essay.text for essay in validation], vocabulary, MAX_SENTENCES, MAX_WORDS, batch_size
    )
    actual = [essay.score for essay in validation]

    model = HANRegressor(len(vocabulary) + 2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    best_mse = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for tokens, targets in train_loader:
            optimizer.zero_grad()
            predictions = model(tokens.to(device))
            loss = loss_fn(predictions, targets.to(device))
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite training loss at epoch {epoch}.")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError(f"Nonfinite gradient at epoch {epoch}.")
            optimizer.step()
            total_loss += loss.item() * targets.numel()

        predicted = _predict(model, validation_loader, device, bounds)
        metrics = evaluate_scores(actual, predicted, bounds)
        validation_mse = metrics["mse"]
        if not math.isfinite(validation_mse):
            raise ValueError(f"Nonfinite validation MSE at epoch {epoch}.")
        history.append({"epoch": epoch, "train_loss": total_loss / len(train),
                        "validation_mse": validation_mse, "validation_qwk": metrics["qwk"]})
        print(
            f"Epoch {epoch}/{epochs} | train normalized MSE {total_loss / len(train):.4f} "
            f"| validation MSE {validation_mse:.4f} | validation QWK {metrics['qwk']}"
        )
        if validation_mse < best_mse:
            best_mse = validation_mse
            best_metrics = metrics
            best_epoch = epoch
            best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    checkpoint = {
        "weights": best_weights,
        "vocabulary": vocabulary,
        "max_sentences": MAX_SENTENCES,
        "max_words": MAX_WORDS,
        "model_config": {"embedding_size": 64, "hidden_size": 32},
        "score_bounds": list(bounds),
        "training": {"seed": seed, "epochs": epochs, "batch_size": batch_size,
                     "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 0.01,
                     "loss": "normalized MSE", "device": str(device),
                     "torch_num_threads": torch.get_num_threads(),
                     "parameter_count": sum(p.numel() for p in model.parameters()),
                     "vocabulary_size": len(vocabulary), "dropout": 0.2,
                     "best_epoch": best_epoch, "selection_metric": "minimum validation MSE",
                     "history": history, "train_truncation": truncation_statistics(train)},
    }
    return checkpoint, best_metrics


def predict_han(path: Path, texts: list[str], bounds) -> list[float]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return predict_checkpoint(checkpoint, texts, bounds)


def predict_checkpoint(checkpoint: dict, texts: list[str], bounds) -> list[float]:
    device = training_device()
    if "score_bounds" in checkpoint and list(bounds) != checkpoint["score_bounds"]:
        raise ValueError("Score bounds differ from the checkpoint.")
    vocabulary = checkpoint["vocabulary"]
    model = HANRegressor(len(vocabulary) + 2, **checkpoint.get("model_config", {})).to(device)
    model.load_state_dict(checkpoint["weights"])
    batch_size = checkpoint.get("training", {}).get("batch_size", 16)
    loader = _prediction_loader(
        texts, vocabulary, checkpoint["max_sentences"], checkpoint["max_words"], batch_size=batch_size
    )
    return _predict(model, loader, device, bounds)
