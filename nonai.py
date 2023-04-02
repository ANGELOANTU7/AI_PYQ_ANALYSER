import os
import re
import chardet
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize
from collections import Counter

def extract_questions_from_file(filepath):
    with open(filepath, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']
    with open(filepath, encoding=encoding) as f:
        content = f.read()
        pattern = r'((?:[IVX]+|\([a-z]\))\. .*(?:\n\s+\(\w\)\. .*)*)'
        matches = re.findall(pattern, content)
        questions = [re.sub(r'\n\s+\(\w\)\. ', ' ', match.strip()) for match in matches]
    return questions

def extract_questions_from_directory(directory):
    questions = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            questions += extract_questions_from_file(filepath)
    return questions

def tokenize_questions(questions, syllabus_file):
    with open(syllabus_file, 'r') as f:
        syllabus = set(f.read().splitlines())
    tokens = []
    for question in questions:
        words = word_tokenize(question)
        for word in words:
            if word.lower() in syllabus:
                tokens.append(word.lower())
    return tokens

def cluster_questions(questions, num_clusters, syllabus_file):
    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    embed = hub.load(module_url)
    embeddings = embed(questions).numpy()
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(embeddings)
    y_kmeans = kmeans.predict(embeddings)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(embeddings)
    plt.figure(figsize=(10, 10))
    plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    for i, txt in enumerate(questions):
        plt.annotate(txt, (principalComponents[i, 0], principalComponents[i, 1]))
    plt.show()
    tokens = tokenize_questions(questions, syllabus_file)
    token_counts = Counter(tokens)
    print("Token\tFrequency")
    for token, count in token_counts.most_common():
        print(f"{token}\t{count}")
    return y_kmeans

questions = extract_questions_from_directory('texts')
num_clusters = 5
num_clusters = int(input("To how many cluster:"))
syllabus_file = 'syllabus_txt/syllabus.txt'
labels = cluster_questions(questions, num_clusters, syllabus_file)
for i in range(num_clusters):
    cluster_questions = np.array(questions)[np.where(labels == i)[0]]
    print(f"Cluster {i+1}:")
    for question in cluster_questions:
        print(f" - {question}")
    print()
