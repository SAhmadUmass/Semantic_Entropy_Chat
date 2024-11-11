from openai import OpenAI
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
import math
import os

client = OpenAI(
    api_key=os.environ.get("openaikey"),
)

# Set the model to use
LLM_MODEL = 'gpt-3.5-turbo'

# Function to generate multiple answers
def generate_answers(question, num_answers=5, temperature=1.0):
    responses = []
    for _ in range(num_answers):
        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": question}
            ],
            model=LLM_MODEL,
            temperature=temperature,
            max_tokens=50,
            n=1,
            stop=None,
        )
        answer = response.choices[0].message.content.strip()
        responses.append(answer)
    return responses

# Function to compute semantic similarity using embeddings
def get_embeddings(responses):
    embeddings = []
    for response in responses:
        embedding = client.embeddings.create(
            input=response,
            model='text-embedding-ada-002'
        )
        embeddings.append(embedding.data[0].embedding)
    return embeddings

# Function to cluster responses based on semantic similarity
def cluster_responses(responses, embeddings, threshold=0.1):
    # Compute pairwise cosine distances
    distances = pairwise_distances(embeddings, metric='cosine')
    # Agglomerative clustering based on distance threshold
    clustering_model = AgglomerativeClustering(
        n_clusters=None,
        metric='precomputed',
        linkage='average',
        distance_threshold=threshold
    )
    clustering_model.fit(distances)
    cluster_labels = clustering_model.labels_
    clusters = {}
    for idx, label in enumerate(cluster_labels):
        clusters.setdefault(label, []).append(responses[idx])
    return clusters

# Function to compute probabilities of clusters
def compute_cluster_probabilities(responses, clusters):
    # Assuming equal probability for each response (since token probabilities are not available)
    total_responses = len(responses)
    cluster_probs = {}
    for cluster_id, cluster_items in clusters.items():
        cluster_size = len(cluster_items)
        cluster_probs[cluster_id] = cluster_size / total_responses
    return cluster_probs

# Function to compute semantic entropy
def compute_semantic_entropy(cluster_probs):
    entropy = -sum(
        prob * math.log(prob, 2) for prob in cluster_probs.values() if prob > 0
    )
    return entropy

# Function to predict hallucination based on entropy
def predict_hallucination(entropy, threshold):
    return entropy > threshold

# Main function to tie everything together
def main(question, entropy_threshold=1.0):
    print(f"Question: {question}")
    # Step 1: Generate multiple answers
    responses = generate_answers(question, num_answers=5, temperature=1.0)
    print("\nGenerated Answers:")
    for idx, resp in enumerate(responses, 1):
        print(f"Answer {idx}: {resp}")

    # Step 2: Get embeddings for responses
    embeddings = get_embeddings(responses)

    # Step 3: Cluster responses based on semantic similarity
    clusters = cluster_responses(responses, embeddings, threshold=0.1)
    print("\nClusters:")
    for cluster_id, cluster_answers in clusters.items():
        print(f"Cluster {cluster_id}: {cluster_answers}")


    # Step 4: Compute cluster probabilities
    cluster_probs = compute_cluster_probabilities(responses, clusters)
    print("\nCluster Probabilities:")
    for cluster_id, prob in cluster_probs.items():
        print(f"Cluster {cluster_id}: {prob}")

    # Step 5: Compute semantic entropy
    entropy = compute_semantic_entropy(cluster_probs)
    print(f"\nSemantic Entropy: {entropy}")

    # Step 6: Predict hallucination
    is_hallucination = predict_hallucination(entropy, entropy_threshold)
    if is_hallucination:
        print("\nThe model's answer is likely a hallucination.")
    else:
        print("\nThe model's answer is likely reliable.")

if __name__ == "__main__":
    # Example question
    example_question = "What is the PMID of the paper by Hoenen T, Groseth A, on the role of VP24 in Ebola virus disease? Only print the PMID"
    main(example_question, entropy_threshold=0.8)
