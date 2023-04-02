
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
import openai

openai.api_key = "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"

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

def extract_important_topics(questions):
    text = '\n'.join(questions)
    print(text)
    response = openai.ChatCompletion.create(
    
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": f"create a table (extract atleast 15 important topic names and give all utube link to study those important topic) and give extra similar 3 questions from each topic:\n\n{text}\n\n"}
    ]
    )


    print("hello")
    #print(response)
    #print(response.choices[0].message)
    important_topics = response.choices[0].message.content
    return important_topics


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
    return y_kmeans


questions = extract_questions_from_directory('texts')
num_clusters = int(input("To how many clusters do you want to cluster the questions: "))
syllabus_file = 'syllabus_txt/syllabus.txt'
labels = cluster_questions(questions, num_clusters, syllabus_file)
for i in range(num_clusters):
    cluster_questions = np.array(questions)[np.where(labels == i)[0]]
    print(f"Module {i+1}:")
    for question in cluster_questions:
        print(f" - {question}")
    print()

important_topics = extract_important_topics(questions)
print("Most important topics are:")
print(important_topics)

# Save cluster questions to file
with open('cluster_questions.txt', 'w') as f:
    for i in range(num_clusters):
        cluster_questions = np.array(questions)[np.where(labels == i)[0]]
        f.write(f"Module {i+1}:\n")
        for question in cluster_questions:
            f.write(f" - {question}\n")
        f.write("\n")

# Save OpenAI reply to file
with open('reply.txt', mode='w',encoding='utf-8') as f:
    f.write(important_topics)