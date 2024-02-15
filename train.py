# Created by Bryce Shurts on January 29th, 2024
# Purpose: Create classifer models
# Note: When I made this I did not realize torch.Tensor (and the other constructors like IntTensor) were deprecated in
# favor of torch.tensor (note the case). torch.Tensor is technically legacy and unsupported but it works so I'm not
# going to bother refactoring everything over to torch.tensor & torch.empty.
import math
import os
from types import FunctionType

import pandas as pd
import torch
import torchtext
from gensim.models import KeyedVectors

BASE_PATH: str = os.path.dirname(os.path.abspath(__file__))
CLASSIFIERS_PATH: str = BASE_PATH + "/classifiers/"
DATASET_PATH: str = BASE_PATH + "/datasets/"
EMBEDDINGS_PATH: str = BASE_PATH + "/embeddings/"
NUM_EMOTIONS: int = 28
EMBED_SIZE: int = 0

if not os.path.exists(CLASSIFIERS_PATH):
    raise FileNotFoundError("Could not find folder for classifier models.")
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError("Could not find folder with GoEmotion dataset.")
if not os.path.exists(EMBEDDINGS_PATH):
    raise FileNotFoundError("Could not find folder with word embeddings sets.")

if not torch.cuda.is_available():
    print("Warning: Using CPU for Pytorch.")
device: device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# First, let's define our basic BDRNN architecture
class BDRNN(torch.nn.Module):
    def __init__(self, vocab_word_count: int, vectors: torch.Tensor, output_size: int, num_layers: int, dropout: float,
                 *args: tuple[any],
                 **kwargs: dict[str, any]) -> None:
        super().__init__(*args, **kwargs)

        self.num_layers = num_layers if num_layers > 1 else 2
        self.hidden_size = NUM_EMOTIONS // num_layers

        self.embeddings = torch.nn.Embedding.from_pretrained(vectors, padding_idx=EMBED_SIZE)

        self.rnn_layers = torch.nn.RNN(input_size=vocab_word_count, hidden_size=self.hidden_size, num_layers=num_layers,
                                       bidirectional=True, dropout=dropout, batch_first=True)

        self.output_layer = torch.nn.Linear(self.hidden_size, output_size)

    def forward(self, input_data) -> torch.Tensor:
        embedded: torch.Tensor = self.embeddings(input_data)

        output: torch.Tensor
        hidden: torch.Tensor
        output, hidden = self.rnn_layers(embedded)

        return self.output_layer(hidden[-1, :])


class pandas_dataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, index: int) -> (str, str):
        return self.df["text"].iloc[index], self.df["emotion_ids"].iloc[index]


def parse_word2vec(word2vec_embeddings, embedding_components) -> tuple[dict[str, int], torch.Tensor]:
        word_labels: dict[str, int] = {}
        tensor: torch.Tensor = torch.empty((EMBED_SIZE + 1, embedding_components), dtype=torch.float32, device=device)
        
        # Clean up the file and load the embeddings into a tensor
        loop_idx = 0
        for word, idx in word2vec_embeddings.key_to_index.items():
            word_labels[word] = idx
            tensor[idx] = torch.tensor(word2vec_embeddings.get_vector(word), dtype=torch.float32,
                                         device=device)
            # Output our progress every 100,000 words
            if (loop_idx + 1) % 100000 == 0:
                print("Processed {}/{}".format(loop_idx + 1, EMBED_SIZE))
            loop_idx += 1
        tensor[-1] = torch.zeros(embedding_components, dtype=torch.float32, device=device)

        # Adding a padding token
        word_labels["<PAD>"] = EMBED_SIZE
        tensor.to(device)
        return word_labels,



# wrinkled_skin -0.1185 0.0853
# /en/c/n/wrinkled_skin -0.1185 0.0853


# "Hello my name is bob."
# ["hello", "my", "name", "is", "bob", "."]
# [word_labels[n], word_labels[x], word_labels[a],  word_labels[b],  word_labels[c]]
# [ n, x, a, b, c] --> nn.Embedding() --> [[...], [...], [...], [...], [...], [...]]


def get_vectors(embedding: str) -> tuple[dict[str, int], torch.Tensor]:
    skip_first_line: bool = False
    global EMBED_SIZE  # Sorry
    match embedding:
        case "glove":
            embedding_path: str = EMBEDDINGS_PATH + "glove.840B.300d.txt"
            EMBED_SIZE = 2196018
            embedding_components: int = 300
        case "word2vec":
            embedding_path: str = EMBEDDINGS_PATH + "GoogleNews-vectors-negative300.bin"
            gn_model = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
            EMBED_SIZE = 3000000
            return parse_word2vec(gn_model, embedding_components)
        case "numberbatch":
            embedding_path: str = EMBEDDINGS_PATH + "numberbatch-19.08-en.txt"
            EMBED_SIZE = 516782
            embedding_components: int = 300
            skip_first_line = True
        case default:
            raise RuntimeError("Invalid embedding chosen.")

    if not os.path.exists(embedding_path):
        raise FileNotFoundError("Could not find embedding file: {}".format(embedding_path))
    with (open(embedding_path, encoding="utf_8") as embeddings_file):
        word_labels: dict[str, int] = {}
        tensor: torch.Tensor = torch.empty((EMBED_SIZE + 1, embedding_components), dtype=torch.float32, device=device)
        if skip_first_line:
            _ = embeddings_file.readline()
        for index, embedding in enumerate(embeddings_file):
            embedding_split: list[str] = embedding.rstrip().split(" ")
            word_labels[embedding_split[0]] = index
            tensor[index] = torch.tensor([float(val) for val in embedding_split[1:]], dtype=torch.float32,
                                         device=device)
            if (index + 1) % 100000 == 0:
                print("Processed {}/{}".format(index + 1, EMBED_SIZE))
        tensor[-1] = torch.zeros(embedding_components, dtype=torch.float32, device=device)
        word_labels["<PAD>"] = EMBED_SIZE
        tensor.to(device)  # Unneeded?
        return word_labels, tensor


def tokenize(text: str, labels: dict, tokenizer: FunctionType) -> list[int]:
    return [labels[word] if word in labels.keys() else labels["something"] for word in tokenizer(text)]


def resolve_emotions(id: str) -> str:
    return [emotions[int(emotion)] for emotion in id.split(",")]


def train(model: BDRNN, batches, num_epochs: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    losses = []

    model.train()
    for epoch_num, epochs in enumerate(range(num_epochs)):
        correct: int = 0
        total: int = 0
        for num_batch, batch in enumerate(batches):
            for sentence, emotions in batch:

                optimizer.zero_grad()

                predictions = model(sentence)
                # Rounding is naive, we should base this off a confidence threshold
                guesses = torch.round(torch.sigmoid(predictions))
                if torch.equal(guesses, emotions): correct += 1
                total += 1

                loss = criterion(predictions, emotions)
                losses.append(float(loss))

                loss.backward()

                optimizer.step()
        print("Epoch: {} | Loss: {} | Accuracy: {}%".format(epoch_num + 1, sum(losses) / len(losses), (correct /
                                                                                                      total) * 100))


def collate(batch: list[tuple[list[int], list[str]]]) -> list[tuple[torch.IntTensor, torch.Tensor]]:
    final_batch = []
    max_tokens = len(max(batch, key=lambda tuple: len(tuple[0]))[0])
    for sentence, emotions in batch:
        sentence.extend([EMBED_SIZE] * (max_tokens - len(sentence)))
        sentence = torch.IntTensor([int(value) for value in sentence]).to(device)
        # There's definitely a way to do a list comprehension here but I'm too stupid to figure it out
        _emotions = torch.zeros(NUM_EMOTIONS, dtype=torch.float32, device=device)
        emotions = emotions.split(",")
        for emotion in emotions:
            _emotions[int(emotion)] = 1.0
        final_batch.append((sentence, _emotions))
    return final_batch  # Can we modify in-place instead?

def main():
    # Now we need to handle our dataset
    with open(DATASET_PATH + "emotions.txt") as emotions_file:
        emotions = [emotion.strip() for emotion in emotions_file]
    if len(emotions) != NUM_EMOTIONS or emotions[4] != "approval":
        raise RuntimeError("Failed to load emotion mappings.")

    training_set = pd.read_csv(DATASET_PATH + "train.tsv", delimiter="\t", names=["text", "emotion_ids"],
                               usecols=[0, 1])
    testing_set = pd.read_csv(DATASET_PATH + "test.tsv", delimiter="\t", usecols=[0, 1])
    print(training_set.head())
    print(testing_set.head())

    max_words: int = max(training_set["text"].map(len).max(), testing_set["text"].map(len).max())
    input_dim: int = 2 ** math.ceil(math.log2(max_words)) if max_words >= 2 else 2

    # Time to do some training!
    labels, vectors = get_vectors("numberbatch")
    tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
    training_set["text"] = training_set["text"].apply(tokenize, labels=labels, tokenizer=tokenizer)
    testing_set["text"] = testing_set["text"].apply(tokenize, labels=labels, tokenizer=tokenizer)
    print(training_set.head())
    print(testing_set.head())
    numberbatch_model = BDRNN(vectors.shape[1], vectors, NUM_EMOTIONS, 4, 0.5).to(device)
    train_dataset = pandas_dataset(training_set)
    test_dataset = pandas_dataset(testing_set)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, collate_fn=collate)
    print('Created `training dataloader` with %d batches!' % len(train_dataloader))
    print('Created `testing dataloader` with %d batches!' % len(test_dataloader))
    train(numberbatch_model, train_dataloader, 10)

if __name__ == '__main__':
    main()