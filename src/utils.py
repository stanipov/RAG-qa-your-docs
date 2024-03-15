"""
Respectfully stolen from https://github.com/langchain-ai/langchain/issues/3016
"""
from langchain.schema import Document
import json
from typing import Iterable
import os
from tqdm.auto import tqdm
import numpy as np

def save_docs_to_jsonl(array:Iterable[Document], file_path:str)->None:
    with open(file_path, 'w') as jsonl_file:
        for doc in array:
            jsonl_file.write(doc.json() + '\n')

def load_docs_from_jsonl(file_path)->Iterable[Document]:
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array
    

def uniquify(path):
    """
    Make a unique filename
    """
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path
    
def gen_report(question, ans, docs)-> str:
    """
    Generates a joint report made from the generated response and the provided documents.
    Note that this helper is not universal. My current documents have refernce/id in the beginning of each.
    In fact, each Document is a standalone regulation.
    """
    extracted_docs_s = ""
    _t = []
    
    for doc in docs:
        _t.append(doc.page_content)
        extracted_docs_s = ("\n\n"+40*"*"+"\n").join(_t)
    report = f"""QUESTION:
{question}
================================================================

Generated response:
{ans}
================================================================

Extracted documents {len(docs)}. 
These are the contens:

{extracted_docs_s}
        """
    return report
#======================================================================================
#
#  Parsing documents
#
#======================================================================================
def recursive_semant_splitter(docs: Iterable[Document], splitter, max_len:int, depth:int=2) -> Iterable[Document]:
    """ A simple iterative splitting using semantic splitter. Only documents larger than max_len will be splitted """
    first_split = []
    for dc in tqdm(docs, leave=True):
        first_split += splitter.transform_documents([dc])

    sem_split_docs = first_split
    if depth > 1:
        for _ in tqdm(range(depth)):
            len_dist = count_lengths(sem_split_docs)
            if np.max(len_dist) <= max_len:
                break
            sem_split_docs = _split_semant_rec(sem_split_docs, splitter, max_len)
    return sem_split_docs


def _split_semant_rec(docs, splitter, max_len):
    """ A helper for  recursive_semant_splitter()"""
    docs2split = []
    unsplitted = []
    for dc in docs:
        if len(dc.page_content) > max_len:
            docs2split.append(dc)
        else:
            unsplitted += [dc]
    t = splitter.transform_documents(docs2split)
    return unsplitted + t


def count_lengths(docs):
    """ Counts lenghts of all documents """
    lens = []
    for doc in docs:
        lens.append(len(doc.page_content))
    return np.array(lens, dtype=int)

