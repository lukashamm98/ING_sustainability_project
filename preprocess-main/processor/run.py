import os
import IPython
import glob
import h5py

import argparse

from processor.pdf.lib import read_pdf, save_to, read_csv
from processor.pdf import vec

def cmdparse():
    parser = argparse.ArgumentParser(description="Process sustainability report documents")
    parser.add_argument("-i", type=str, default=os.path.expanduser('~/data/sustain/all/'), help="path to read PDF files from")
    parser.add_argument("-o", type=str, default="out", help="path to save output files to")
    parser.add_argument("-m", choices=['pdf2csv', 'csv2vec', 'trainvec'], default='pdf2csv', help="mode to run")
    parser.add_argument("--model", type=str, default="model.bin", help="pretrained model to use")
    args = parser.parse_args()
    return args

# Extension of inputs based on the running mode
extension = {
    'pdf2csv': r'*.pdf',
    'csv2vec': r'*.csv',
    'trainvec': r'*.csv'
}


if __name__=="__main__":
    args = cmdparse()

    os.makedirs(args.o, exist_ok=True)

    # Read PDF and convert to tabular data
    ls = glob.glob(args.i + '/' + extension[args.m])

    if args.m == 'pdf2csv':
        for i,f in enumerate(ls):
            print(f'({i}) processing : {f}')
            # Read PDF as collection of text blocks
            ctree = read_pdf(f)
            if ctree is not None and len(ctree)>0:
                csvpath = args.o + '/' + ctree[0].filename.split('.')[0] + '.csv'
                print(f'... Saving output csv : {csvpath}')
                save_to(ctree, csvpath)
            else:
                print(f'WARNING: File {f} is corrupted')

    elif args.m == 'trainvec':
        # Train doc2vec vector
        if args.model is None or len(args.model)==0:
            raise FileNotFoundError("Need to specify output pretrained model with --model parameter")
        
        alltext = []
        for i,f in enumerate(ls):
            print(f'({i}) processing : {f}')
            doc = read_csv(f) # get collection of text blocks
            alltext += doc

        print('----------------------------------')
        print(f'{len(alltext)} text paragraphs collected')
        print('Preparing to train doc2vec model')
        print('----------------------------------')
        
        # Train all text
        model = vec.fit(alltext)

        # Save the model
        model.save(args.model)
        print(f'Model saved to {args.model}')

    elif args.m == 'csv2vec':
        # Load pretrained model (if exist)
        if args.model is None or len(args.model)==0:
            raise FileNotFoundError("Need to specify input pretrained model with --model parameter")
        model = vec.load_model(args.model)

        # Read csv and convert to vector
        outpath = os.path.join(args.o, 'output.h5')
        file = h5py.File(outpath, 'w')
        for i,f in enumerate(ls):
            print(f'({i}) processing : {f}')
            doc = read_csv(f) # get collection of text blocks
            dvec = [vec.doc2vec(model, para) for para in doc]

            # save vector (binary)    
            file.create_dataset(os.path.basename(f), data=dvec)
        print(f'Saving output vectors into : {outpath}')
        file.close()
