from openai import OpenAI
import os
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
import math

client = OpenAI(
    api_key=os.environ.get("openaikey"),
)
# Set the model to use
LLM_MODEL = 'gpt-3.5-turbo'

# Function to generate multiple answers with token log probabilities
def generate_answers_with_probs(question, num_answers=5, temperature=1.0):
    responses = []
    for _ in range(num_answers):
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "user", "content": question}
            ],
            temperature=temperature,
            max_tokens=50,
            n=1,
            stop=None,
            logprobs=True,
        )
        choice = response.choices[0]
        text = choice.message.content.strip()
        logprobs = choice.logprobs
        tokens = logprobs.tokens
        token_logprobs = logprobs.token_logprobs
        responses.append({
            'text': text,
            'tokens': tokens,
            'token_logprobs': token_logprobs,
        })
    return responses

# Function to compute sequence probability from token log probabilities
def calculate_sequence_probability(token_logprobs):
    # Sum the log probabilities of all tokens in the sequence
    total_logprob = sum(token_logprobs)
    # Convert log probability to probability
    sequence_prob = math.exp(total_logprob)
    return sequence_prob

# Function to compute semantic similarity using embeddings
def get_embeddings(responses):
    embeddings = []
    for response in responses:
        embedding = client.embeddings.create(
            input=response,
            model='text-embedding-ada-002'
        )
        embeddings.append(embedding['data'][0]['embedding'])
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

# Function to compute probabilities of clusters using sequence probabilities
def compute_cluster_probabilities_with_sequence_probs(responses, clusters):
    cluster_probs = {}
    total_probability = sum(response['sequence_prob'] for response in responses)
    for cluster_id, cluster_items in clusters.items():
        cluster_prob = sum(
            response['sequence_prob'] for response in responses if response['text'] in cluster_items
        )
        cluster_probs[cluster_id] = cluster_prob / total_probability
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
    # Step 1: Generate multiple answers with token probabilities
    responses = generate_answers_with_probs(question, num_answers=5, temperature=1.0)
    print("\nGenerated Answers:")
    for idx, resp in enumerate(responses, 1):
        print(f"Answer {idx}: {resp['text']}")
    
    # Step 2: Get embeddings for responses
    response_texts = [resp['text'] for resp in responses]
    embeddings = get_embeddings(response_texts)
    
    # Step 3: Cluster responses based on semantic similarity
    clusters = cluster_responses(response_texts, embeddings, threshold=0.1)
    print("\nClusters:")
    for cluster_id, cluster_answers in clusters.items():
        print(f"Cluster {cluster_id}: {cluster_answers}")
    
    # Step 4: Compute sequence probabilities
    for response in responses:
        response['sequence_prob'] = calculate_sequence_probability(response['token_logprobs'])
    
    # Step 5: Compute cluster probabilities using sequence probabilities
    cluster_probs = compute_cluster_probabilities_with_sequence_probs(responses, clusters)
    print("\nCluster Probabilities:")
    for cluster_id, prob in cluster_probs.items():
        print(f"Cluster {cluster_id}: {prob}")
    
    # Step 6: Compute semantic entropy
    entropy = compute_semantic_entropy(cluster_probs)
    print(f"\nSemantic Entropy: {entropy}")
    
    # Step 7: Predict hallucination
    is_hallucination = predict_hallucination(entropy, entropy_threshold)
    if is_hallucination:
        print("\nThe model's answer is likely a hallucination.")
    else:
        print("\nThe model's answer is likely reliable.")

if __name__ == "__main__":
    # Example question
    example_question = "Who wrote the play Hamlet?"
    main(example_question, entropy_threshold=0.8)
