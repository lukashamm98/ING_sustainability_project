from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.tokenize import word_tokenize

def safe_tokenise(para):
    try:
        return word_tokenize(para)
    except:
        try:
            return para.split(' ')
        except AttributeError: # Numerical token
            return [str(para)]

def fit(all_text, vecsize=50, alpha=0.0125, decay_rate=0.0005, epochs=20):
    nltk.download('punkt')
    print('Tokenising input paragraphs ...')
    tagged = [TaggedDocument(words=safe_tokenise(p), tags=['none']) for p in all_text if p is not None]
    model = Doc2Vec(
        vector_size=vecsize,
        alpha=alpha,
        min_count=1,
        epochs=epochs)
    print(f'Building vocab from {len(tagged)} paragraphs')
    model.build_vocab(tagged)
    return model

def load_model(path):
    return Doc2Vec.load(path)

def doc2vec(model, text):
    return model.infer_vector(safe_tokenise(text))