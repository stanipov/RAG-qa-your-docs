"""
Respectfully stolen from https://github.com/langchain-ai/langchain/issues/3016
"""
from langchain.schema import Document
import json
from typing import Iterable
import os

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
    
def gen_report(ans, docs)-> str:
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
