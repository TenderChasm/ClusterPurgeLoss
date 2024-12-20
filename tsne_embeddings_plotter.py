import pickle
import matplotlib.pyplot as plt
import numpy
import pandas as pd
from sklearn.manifold import TSNE

def produce_df(db):

    lines = []
    for center,mutants in db.items():
        for url, embedding in mutants[0]:
            lines.append([center,0,url,embedding])
        for url, embedding in mutants[1]:
            lines.append([center,1,url,embedding])

    data = pd.DataFrame(data = lines, columns=['center', 'sign', 'url', 'embedding'])
    tsne = TSNE(random_state = 0, n_iter = 5000, metric = 'cosine')
    embeddings_array = numpy.stack(data['embedding'].to_numpy())
    embeddings2d = tsne.fit_transform(embeddings_array)
    data['embeddings2d'] = list(embeddings2d)

    return data

def draw(data,db,name):
    cmap = plt.cm.get_cmap('Spectral', 53)
    colors = cmap(numpy.arange(0,53,1))
    i = -1
    for center,mutants in db.items():
        fig.clear()
        plt.xlim(min_x - margin, max_x + margin)
        plt.ylim(min_y- margin,max_y + margin)
        plt.title(f"Mutant class with origin {center} : {name}")
        i+=1
        #if i not in [1,8,15]:
        negatives = data[(data['center'] == center) & (data['sign'] == 0)]
        if len(negatives) > 0:
            plt.scatter(x = numpy.stack(negatives['embeddings2d'].to_numpy())[:, 0],
                        y = numpy.stack(negatives['embeddings2d'].to_numpy())[:, 1],
                        color='red',s = 80,label = 'non-equivalent')
        positives = data[(data['center'] == center) & (data['sign'] == 1)]
        if len(positives) > 0:
            plt.scatter(x = numpy.stack(positives['embeddings2d'].to_numpy())[:, 0],
                        y = numpy.stack(positives['embeddings2d'].to_numpy())[:, 1],
                        color='blue',s = 80,label = 'equivalent')
        center_emb = data[(data['center'] == center) & (data['url'] == center)]
        plt.scatter(x = numpy.stack(center_emb['embeddings2d'].to_numpy())[:, 0],
                    y = numpy.stack(center_emb['embeddings2d'].to_numpy())[:, 1],
                    color='black',s = 80, label = 'origin')
        plt.legend()
        plt.savefig(f'images\map_{center}_{name}.png')

    #plt.savefig(f'map_{name}_auto.png')
    #fig.clear()



dbfile_orig = open('embeddings_original_model_norm', 'rb')    
db_orig = pickle.load(dbfile_orig)

dbfile_new = open('embeddings_new_model_norm', 'rb')    
db_new = pickle.load(dbfile_new)

data_orig = produce_df(db_orig)
data_new = produce_df(db_new)

plt.rcParams.update({'font.size': 30})
fig = plt.figure(figsize=(20,20))
min_x = min(numpy.stack(data_orig['embeddings2d'].to_numpy())[:,0].min(), numpy.stack(data_new['embeddings2d'].to_numpy())[:,0].min())
max_x = max(numpy.stack(data_orig['embeddings2d'].to_numpy())[:,0].max(), numpy.stack(data_new['embeddings2d'].to_numpy())[:,0].max())
min_y = min(numpy.stack(data_orig['embeddings2d'].to_numpy())[:,1].min(), numpy.stack(data_new['embeddings2d'].to_numpy())[:,1].min())
max_y = max(numpy.stack(data_orig['embeddings2d'].to_numpy())[:,1].max(), numpy.stack(data_new['embeddings2d'].to_numpy())[:,1].max())
margin = 5

draw(data_orig, db_orig,'baseline')
draw(data_new, db_new, 'CPL')


#plt.show()
#plt.savefig('map_new.png')
#a = 1

