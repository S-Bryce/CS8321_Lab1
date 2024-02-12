# Created by Bryce Shurts on January 29th, 2024
# Purpose: Load classifer model & related embeddings, construct example cases for definitional distance comparison
# Download link for word2vec: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
# Download link for Glove: https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from sklearn.manifold import TSNE
from torch import Tensor, cat
from torch.cuda import is_available as cuda_is_available
from train import get_vectors


# EDITABLE VARIABLES
embeddings = ["numberbatch", "glove"] # Embeddings definition. Add word2vec once deserialization is done.

# Word comparison groups. Format: [base_word, similar_word_1, similar_word_2]
word_comparison_groups = [
    ["tire", "tired", "tyre"],
]

# Add a new distance function here if you want.
def calculate_distances(base_word: ndarray[float], similar_word_1: ndarray[float], similar_word_2: ndarray[float]) -> dict[str, list[float]]:
    return {
        "euclidean": [euclidean_distance(similar_word_1, base_word), euclidean_distance(similar_word_2, base_word)],
        "cosine": [cosine_similarity(similar_word_1, base_word), cosine_similarity(similar_word_2, base_word)],
        "manhattan": [manhattan_distance(similar_word_1, base_word), manhattan_distance(similar_word_2, base_word)],
    }


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

def get_word_vectors_to_compare(numpy_vectors: ndarray[float]) -> tuple[ndarray[float], ndarray[float], ndarray[float]]:
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

def get_distances(embeddings_list: list[Tensor]) -> dict[str, list[float]]:
    numpy_vectors = convert_tensors_to_numpy(embeddings_list)
    base_word, similar_word1, similar_word2 = get_word_vectors_to_compare(numpy_vectors)
    return calculate_distances(base_word, similar_word1, similar_word2)

def get_reduced_vectors(embeddings_list: list[Tensor]) -> ndarray[float, float]:
        numpy_vectors = convert_tensors_to_numpy(embeddings_list)
        return reduce_embeddings(numpy_vectors)

def visualize_embedding(embeddings_list: list[Tensor], norm_distances: dict[str, list[float]], word_list: list[str]) -> dict[str, list[float]]:
    reduced_vectors: ndarray[float, float] = get_reduced_vectors(embeddings_list)
    show_plot_comparison(reduced_vectors, word_list, norm_distances)

def get_embeddings(embeddings: list[str]) -> dict[str, (dict[str, int], Tensor)]:
    if len(embeddings) == 0:
        raise ValueError("No embeddings were selected to load.")
    
    result = {}
    for embedding in embeddings:
        vocab, vectors = get_vectors(embedding)
        result[embedding] = (vocab, vectors)
    return result

# Bc comparison groups added in sequence, we can query word_comparison_groups by index from where the group appears in the result. 
# (result[embedding_name][0] for the first word group)
def get_comparison_embeddings(embeddings: dict[str, (dict[str, int], Tensor)]) -> dict[str, list[list[Tensor]]]:
    # Dictionary of embedding name to list of comparison groups
    result: dict[str, list[list[Tensor]]] = {}
    for embed_name, (vocab, vectors) in embeddings.items():
        comparisons = []
        if embed_name not in result:
            result[embed_name] = []
        for idx, group in enumerate(word_comparison_groups):
            comparisons = []
            for word in group:
                comparisons.append(vectors[vocab[word]])
            result[embed_name].append(comparisons)
    return result

"""Returns a dictionary of embedding names to a list of dictionaries of distance types to their calculated distances"""
def compare_embeddings(comparison_embeddings: dict[str, list[list[Tensor]]]) -> dict[str, list[dict[str, list[float]]]]:
    result = {}
    for embedding, word_groups in comparison_embeddings.items():
        for idx, group_vectors in enumerate(word_groups):
            if result.get(embedding) is None:
                result[embedding] = []
            distances: dict[str, list[float]] = get_distances(group_vectors)
            result[embedding].append(distances)
    return result

def compare_distances(embedding_distances: dict[str, list[dict[str, list[float]]]]):
    """Returns a dictionary of embedding names to a dictionary of distance types to their calculated distance ratios."""
    def calculate_distance_ratios():
        # Dict of embedding name to distance type and all of that distance type's calculated distance ratios
        ratios: dict[str, dict[str, list[float]]] = {}
        # For every distance type for embeddings
        for embedding_name, distances in embedding_distances.items():
            ratios[embedding_name] = {}
            for distance in distances:
                for distance_type, values in distance.items():
                    if ratios[embedding_name].get(distance_type) is None:
                        ratios[embedding_name][distance_type] = []
                    ratio = (max(values[0], values[1]) / min(values[0], values[1]))
                    ratios[embedding_name][distance_type].append(ratio)
        return ratios

    """
    Plots boxplots of the distance ratios for each embedding and distance type.
    Args: ratios: dict[str, dict[str, list[float]]] - Dictionary of embedding names to a dictionary of distance types to a list of all their calculated distance ratios.
    
    Ex:
    ratios = {
        "glove": {
            "euclidean": [1.0, 2.55, 1.380, 4.44, 4.4, 4983],
            "cosine": [1.0, 2.55, 1.380, 4.44, 4.4, 4983],
            ...
        },
        "numberbatch": {
            "euclidean": [1.0, 2.55, 1.380, 4.44, 4.4, 4983],
            "cosine": [1.0, 2.55, 1.380, 4.44, 4.4, 4983],
            ...        
        }
    }
    """
    def show_boxplots(ratios: dict[str, dict[str, list[float]]]):
        distances: list[dict[str, list[float]]] = list(ratios.values())
        # Get all the values per distance type for every embedding
        to_plot = {}
        for i in range(0, len(distances)):
            for embedding_name, distances2 in ratios.items():
                for distance_type, values in distances2.items():
                    print(embedding_name, distance_type, values)
                    if to_plot.get(distance_type) is None:
                        to_plot[distance_type] = {}
                    to_plot[distance_type][embedding_name] = values
                    
        for distance_type, embedding_data in to_plot.items():
            fig, axs = plt.subplots(figsize=(10, 8))
            boxplots_data = []
            labels = []
            for embedding_name, values in embedding_data.items():
                boxplots_data.append(values)
                labels.append(embedding_name)
            axs.boxplot(boxplots_data)
            axs.set_xticklabels(labels)
            axs.set_title(distance_type)
            plt.tight_layout()
            plt.show()
    
    # Dict of embedding name to distance type and its calculated distance ratio
    ratios = calculate_distance_ratios()
    show_boxplots(ratios)

orig_embeddings = get_embeddings(embeddings)
comp_embeddings = get_comparison_embeddings(orig_embeddings)
comp_distances = compare_embeddings(comp_embeddings)
compare_distances(comp_distances)
