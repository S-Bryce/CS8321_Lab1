# Created by Bryce Shurts on January 29th, 2024
# Purpose: Load classifer model & related embeddings, construct example cases for definitional distance comparison

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from torch import Tensor
from torch.cuda import is_available as cuda_is_available
from train import get_vectors


# We should look at comparing vectors in different embeddings and see how well ambigious words center around common
# synonyms for each meaning. We could probably do some sort of visualization for this as well.


def visualize_embedding(embeddings_list: list[Tensor], word_list: list[str]) -> list[float]:
    tsne = TSNE(n_components=3, random_state=0, perplexity=2, init='pca', n_iter=6000)
    if cuda_is_available():
        numpy_vectors = np.array([vector.cpu().numpy() for vector in embeddings_list])
    else:
        numpy_vectors = np.array([vector.numpy() for vector in embeddings_list])
    reduced_vectors = tsne.fit_transform(numpy_vectors)
    norm_distances: list[float] = [np.linalg.norm(numpy_vectors[1] - numpy_vectors[0]),
                                   np.linalg.norm(numpy_vectors[2] - numpy_vectors[0])]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(reduced_vectors[0][0], reduced_vectors[0][1], reduced_vectors[0][2], c='b', label=word_list[0])
    ax.scatter(reduced_vectors[1][0], reduced_vectors[1][1], reduced_vectors[1][2], c='g', label=word_list[1])
    ax.scatter(reduced_vectors[2][0], reduced_vectors[2][1], reduced_vectors[2][2], c='r', label=word_list[2])
    ax.legend()
    plt.show()
    print("Distance between 'tire' and 'tired' is {}\nDistance between 'tire' and 'tyre' is {}".format(
        norm_distances[0], norm_distances[1]))
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

print("=== Numberbatch distances ===")
numberbatch_distances: list[float] = visualize_embedding(numberbatch_comparisons, ["tire", "tired", "tyre"])
print("=== Glove distances ===")
glove_distances: list[float] = visualize_embedding(glove_comparisons, ["tire", "tired", "tyre"])

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
