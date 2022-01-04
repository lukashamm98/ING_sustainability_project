import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import h5py
import argparse

from sklearn.manifold import TSNE

def cmdparse():
    parser = argparse.ArgumentParser(description="Plot processed vectors")
    parser.add_argument("-p", type=str, help="Path to h5 vector file (for positive examples)")
    parser.add_argument("-n", type=str, help="Path to h5 vector file (for negative examples)")
    args = parser.parse_args()
    return args

def load_vector(path):
    if path is not None and os.path.isfile(path):
        print(f'Loading vectors from : {path}')
        file = h5py.File(path, 'r')
        print(f'... {len(file.keys())} vectors found')
        return file
    else:
        print(f'Unable to load vectors from : {path}')
        return None

def tsne(file):
    if file is not None:
        tsne = TSNE(n_components=2, n_iter=500, metric='cosine', init='random')
        vectors = None
        for key in file.keys():
            # Convert HDFS Dataset to numpy array
            vecs = np.array(file.get(key))
            if vectors is not None:
                vectors = np.vstack((vectors, vecs))
            else:
                vectors = vecs
            print(vectors.shape)
        print(f'Input vector size : {vectors.shape}')
        print('Running T-SNE ...')
        file.close()
        return tsne.fit_transform(vectors)


if __name__=='__main__':
    args = cmdparse()

    if args.p is None:
        raise ValueError('Please specify argument -p')
    
    # Load vectors
    positives = load_vector(args.p)
    negatives = load_vector(args.n)

    # T-SNE vectors
    positives = tsne(positives)
    negatives = tsne(negatives)

    # plot
    df = pd.DataFrame(positives, columns=['c1','c2'])
    df.loc[:, 'color'] = 'positive'
        
    if negatives is not None:
        df2 = pd.DataFrame(negatives, columns=['c1','c2'])
        df2.loc[:, 'color'] = 'negative'
        df = pd.concat([df, df2])

    sns.scatterplot(data=df, x='c1', y='c2', hue='color', legend=True)
    plt.show()
