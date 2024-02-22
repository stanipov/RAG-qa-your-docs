"""
Wrapper to make some chains
"""
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.prompts import PromptTemplate
from typing import List

class WrapperChains:
    def __init__(self, llm):
        """
        llm: the generation HF pipeline
        """
        self.llm = llm

    def make_long_context_chain(self, prompt:PromptTemplate):
        """
        :param prompt: prompt to the chain for the documents
        :return: LLMChain
        """
        document_prompt = PromptTemplate(input_variables=["page_content"],
                                         template="{page_content}")
        document_variable_name = "context"
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        return StuffDocumentsChain(
            llm_chain=llm_chain,
            document_prompt=document_prompt,
            document_variable_name=document_variable_name,
        )