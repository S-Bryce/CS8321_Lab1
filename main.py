# Created by Bryce Shurts on January 29th, 2024
# Purpose: Load classifer model & related embeddings, construct example cases for definitional distance comparison
# Download link for word2vec: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
# Download link for Glove: https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from sklearn.manifold import TSNE
from torch import Tensor
from torch.cuda import is_available as cuda_is_available
from train import get_vectors


# We should look at comparing vectors in different embeddings and see how well ambigious words center around common
# synonyms for each meaning. We could probably do some sort of visualization for this as well.

def euclidean_distance(vector1: Tensor, vector2: Tensor) -> float:
    return np.linalg.norm(vector1 - vector2)

def cosine_similarity(vector1: Tensor, vector2: Tensor) -> float:
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

def manhattan_distance(v1, v2):
    return np.sum(np.abs(v1 - v2))

def convert_tensors_to_numpy(embeddings_list: list[Tensor]) -> ndarray[float]:
    if cuda_is_available():
        numpy_vectors = np.array([vector.cpu().numpy() for vector in embeddings_list])
    else:
        numpy_vectors = np.array([vector.numpy() for vector in embeddings_list])
    return numpy_vectors

def reduce_embeddings(numpy_vectors: ndarray[float]) -> ndarray[float]:
    tsne = TSNE(n_components=3, random_state=0, perplexity=2, init='pca', n_iter=6000)
    return tsne.fit_transform(numpy_vectors)

def get_distances(base_word: ndarray[float], similar_word_1: ndarray[float], similar_word_2: ndarray[float]) -> dict[str, list[float]]:
    return {
        "euclidean": [euclidean_distance(similar_word_1, base_word), euclidean_distance(similar_word_2, base_word)],
        "cosine": [cosine_similarity(similar_word_1, base_word), cosine_similarity(similar_word_2, base_word)],
        "manhattan": [manhattan_distance(similar_word_1, base_word), manhattan_distance(similar_word_2, base_word)],
    }

def get_word_vectors_to_compare(numpy_vectors: ndarray[float]) -> (ndarray[float], ndarray[float], ndarray[float]):
    return numpy_vectors[0], numpy_vectors[1], numpy_vectors[2]

def show_plot_comparison(reduced_vectors: ndarray[float], word_list: list[str], norm_distances: dict[str, list[float]]):
    for distance_type, calculated_distances in norm_distances.items():
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(reduced_vectors[0][0], reduced_vectors[0][1], reduced_vectors[0][2], c='b', label=word_list[0])
        ax.scatter(reduced_vectors[1][0], reduced_vectors[1][1], reduced_vectors[1][2], c='g', label=word_list[1])
        ax.scatter(reduced_vectors[2][0], reduced_vectors[2][1], reduced_vectors[2][2], c='r', label=word_list[2])
        ax.legend()
        plt.show()
        print("{} distance between 'tire' and 'tired' is {}\nDistance between 'tire' and 'tyre' is {}".format(
            distance_type, calculated_distances[0], calculated_distances[1]))

def visualize_embedding(embeddings_list: list[Tensor], word_list: list[str]) -> dict[str, list[float]]:
    numpy_vectors = convert_tensors_to_numpy(embeddings_list)
    reduced_vectors = reduce_embeddings(numpy_vectors)
    base_word, similar_word_1, similar_word_2 = get_word_vectors_to_compare(numpy_vectors)
    norm_distances = get_distances(base_word, similar_word_1, similar_word_2)
    show_plot_comparison(reduced_vectors, word_list, norm_distances)
    return norm_distances


numberbatch_vocab: dict[str, int]
numberbatch_vectors: Tensor
numberbatch_vocab, numberbatch_vectors = get_vectors("numberbatch")
glove_vocab: dict[str, int]
glove_vectors: Tensor
glove_vocab, glove_vectors = get_vectors("glove")

numberbatch_comparisons: list[Tensor] = [numberbatch_vectors[numberbatch_vocab["tire"]],
                                         numberbatch_vectors[numberbatch_vocab["tired"]],
                                         numberbatch_vectors[numberbatch_vocab["tyre"]]]
glove_comparisons: list[Tensor] = [glove_vectors[glove_vocab["tire"]],
                                   glove_vectors[glove_vocab["tired"]],
                                   glove_vectors[glove_vocab["tyre"]]]


# Hardcoding Euclidean for now.
print("=== Numberbatch distances ===")
numberbatch_distances: list[float] = visualize_embedding(numberbatch_comparisons, ["tire", "tired", "tyre"])["euclidean"]
# Hardcoding Euclidean for now.
print("=== Glove distances ===")
glove_distances: list[float] = visualize_embedding(glove_comparisons, ["tire", "tired", "tyre"])["euclidean"]

print("=== Overall distances ===")
numberbatch_dist_ratio: float = (max(numberbatch_distances[0], numberbatch_distances[1])
                                 / min(numberbatch_distances[0], numberbatch_distances[1]))
glove_dist_ratio: float = (max(glove_distances[0], glove_distances[1])
                           / min(glove_distances[0], glove_distances[1]))
if numberbatch_dist_ratio > glove_dist_ratio:
    print("Glove distances are closer by a ratio of {} vs Numberbatch's ratio of {} for a difference of {}.".format(
        glove_dist_ratio, numberbatch_dist_ratio, numberbatch_dist_ratio - glove_dist_ratio
    ))
elif numberbatch_dist_ratio < glove_dist_ratio:
    print("Numberbatch distances are closer by a ratio of {} vs Glove's ratio of {} for a difference of {}.".format(
        numberbatch_dist_ratio, glove_dist_ratio, glove_dist_ratio - numberbatch_dist_ratio
    ))
else:
    print("Numberbatch & Glove distance ratios are identical with a value of {}".format(numberbatch_dist_ratio))
