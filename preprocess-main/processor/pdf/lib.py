import os
import scipy
import scipy.stats
import numpy as np
import pandas as pd
from collections import namedtuple
from functools import partial
from io import StringIO
from bisect import insort_right

import fitz
from lxml import etree

BLOCK_IMAGE = 1
BLOCK_TEXT = 0

Block = namedtuple("Block", ['filename','tl','br','w','h','content', 'type', 'density'])
SBlock = namedtuple("SBlock", ['filename','page','content','priority'])

def escape(t):
    return t.replace("\"", "").strip()

def sblock_to_vec(sb):
    pass

def save_to(sbs, path):
    """
    Save collection of [[SBlock]] into CSV format
    """
    with open(path, 'w') as f:
        f.write(f'page,priority,content\n')
        for s in sbs:
            f.write(f'{s.page},{s.priority},"{escape(s.content)}"\n')

def read_pdf2(path: str):
    """
    Read PDF pages with PyPDF2 lib
    """
    import PyPDF2

    file = PyPDF2.PdfFileReader(open(path, 'rb'))
    num_pages = file.getNumPages()
    pages = [file.getPage(n) for n in range(num_pages)]
    return pages

def read_pdf(path: str, verbose: bool=False):
    """
    Read PDF pages with PyMuPDF
    """
    if verbose:
        print(f'Reading : {path}')
    try:
        pdf = fitz.open(path)
        num_pages = pdf.page_count
        metadata = pdf.metadata
        pages = [pdf.load_page(n) for n in range(num_pages)]

        # Read as textblocks
        filename = os.path.basename(path)

        # Generate content tree: List[Page] => List[Block]
        # NOTE: this code is not very scalable
        ctree = []
        for pageno, page in enumerate(pages):
            list_density = [] # pixel density of each block on this page [p]
            sorted_density = [] # pixel density of each block on this page [p], but sorted
            page_blocks = []
            for bl in page.get_text_blocks():
                x0, y0, x1, y1, obj, bl_no, bl_type = bl
                w = x1-x0
                h = y1-y0
                # Only process text blocks
                if bl_type==BLOCK_TEXT:
                    density = (w*h) / len(obj) # pixel density of this object
                    insort_right(sorted_density, density)
                    list_density.append(density)
                    page_blocks.append(Block(
                        filename=filename, 
                        tl=(x0,y0), br=(x1,y1), 
                        w=w, h=h, 
                        content=obj, type=bl_type, density=density))
            
            # Reconcile the density rank of each text block on the page
            # with this, we know exactly which text block is "likely" to be titles or paragraph
            # 
            # Text block with higher rank (priority) means larger text size, likely to be title
            density_rank = map(lambda d: scipy.stats.percentileofscore(sorted_density, d), list_density)
            for bl, rank in zip(page_blocks, density_rank):
                ctree.append(SBlock(
                    filename=filename,
                    page=pageno, content=bl.content.replace('\n','. ').strip(), 
                    priority=rank/100.))

        return ctree

    except RuntimeError:
        print(f'... failed to read {path}, probably unsupported PDF media')
        return None

def read_csv(filename):
    """
    Read CSV as text document
    """
    df = pd.read_csv(filename)
    return df['content'].tolist()

def parse_html(ht):
    tree = etree.parse(StringIO(ht))
    return tree
