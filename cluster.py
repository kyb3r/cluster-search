import json
import pickle
from collections import defaultdict
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


class Cluster:
    def __init__(self, corpus):
        self.corpus = corpus
        self.model = KMeans(n_clusters=3, max_iter=1000)
        self.responses = self.load_json('responses.json')
        self.vectorizer = TfidfVectorizer(
            stop_words=self.load_json('stop_words.json')
            )
    
    @staticmethod
    def load_json(path):
        with open(path) as f:
            return json.load(f)
    
    @classmethod
    def from_saved(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def from_json(cls, path):
        corpus = cls.load_json(path)
        return cls(corpus)

    def show_cluster_info(self, number):
        order_centroids = self.model.cluster_centers_.argsort()[:, ::-1]
        terms = self.vectorizer.get_feature_names()
        print(f'Cluster {number}: ', end='')
        print(' '.join(terms[ind] for ind in order_centroids[number-1, :10]))
    
    def show_info(self):
        print("Top terms per cluster:")
        order_centroids = self.model.cluster_centers_.argsort()[:, ::-1]
        terms = self.vectorizer.get_feature_names()

        for i in range(self.model.n_clusters):
            print(f'Cluster {i+1}: ', end='')
            print(' '.join(terms[ind] for ind in order_centroids[i, :10]))


    def train(self):
        vectors = self.vectorizer.fit_transform(self.corpus)
        self.model.fit(vectors)

    def respond(self, text):
        prediction = self.predict(text)
    
    def predict(self, text):
        vector = self.vectorizer.transform([text])
        prediction = self.model.predict(vector)
        return int(prediction + 1)
    
    def save(self, fp=None):
        serialized = pickle.dumps(self)
        if fp:
            with open(fp, 'wb') as f:
                f.write(serialized)
        return serialized


model = Cluster.from_json('messages.json')
model.train()
model.save('models/model2')

clusters = defaultdict(list)

for message in model.load_json('messages.json'):
    prediction = model.predict(message)
    clusters[prediction].append(message)

for cluster in clusters:
    model.show_cluster_info(cluster)
    print('-'*10)
    for message in clusters[cluster][:20]:
        message = message.replace('\n', ' ')
        print(message[:100] if len(message) > 100 else message)
    print('-'*10)


while True:
    msg = input('Enter message: ')
    prediction = model.predict(msg)
    print('CLUSTER', prediction)



