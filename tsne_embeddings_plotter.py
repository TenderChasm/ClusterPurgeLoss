import pickle
import matplotlib.pyplot as plt
import numpy
import pandas as pd
from sklearn.manifold import TSNE

dbfile = open('embeddings_original_model_norm', 'rb')    
db = pickle.load(dbfile)

lines = []
for center,mutants in db.items():
    for url, embedding in mutants[0]:
        lines.append([center,0,url,embedding])
    for url, embedding in mutants[1]:
        lines.append([center,1,url,embedding])

data = pd.DataFrame(data = lines, columns=['center', 'sign', 'url', 'embedding'])

tsne = TSNE(random_state = 0, n_iter = 1000, metric = 'cosine')
embeddings_array = numpy.stack(data['embedding'].to_numpy())
embeddings2d = tsne.fit_transform(embeddings_array)
data['embeddings2d'] = list(embeddings2d)

plt.figure(figsize=(100,100))
cmap = plt.cm.get_cmap('Spectral', 53)
colors = cmap(numpy.arange(0,53,1))
i = -1
for center,mutants in db.items():
    i+=1
    #if i not in [1,8,15]:
    positives = data[(data['center'] == center) & (data['sign'] == 1)]
    if len(positives) > 0:
        plt.scatter(x = numpy.stack(positives['embeddings2d'].to_numpy())[:, 0],
                    y = numpy.stack(positives['embeddings2d'].to_numpy())[:, 1],
                    color='blue')
    negatives = data[(data['center'] == center) & (data['sign'] == 0)]
    if len(negatives) > 0:
        plt.scatter(x = numpy.stack(negatives['embeddings2d'].to_numpy())[:, 0],
                    y = numpy.stack(negatives['embeddings2d'].to_numpy())[:, 1],
                    color='red')

#plt.show()
plt.savefig('map.png')
a = 1

