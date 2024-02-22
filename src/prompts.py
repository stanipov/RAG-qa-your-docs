from langchain.prompts import PromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain import hub

from typing import List

class RAGPromptTemplates:

    @staticmethod
    def long_context(instruct1:str, instruct2:str, query:str):
        """
        Prompt in a form:
        instruct1:
        -----
        {context}
        -----
        query
        {query}

        instruct2

        instruct1,2 : your actual instructions
        context, query - are runnables of the LangChain retrieval

        Example query:
        Given these texts: <-- instruct1
        -----
        {context}
        -----
        Answer this question: <-- query
        {query}
        If you don't know the answer, just say that you don't know. Keep the answer concise. <-- instruct2
        """
        prompt = f"{instruct1}"
        prompt +="-----\n{context}\n------\n"
        prompt += f"{query}"
        prompt += "\n{query}\n\n"
        prompt += f"{instruct2}"
        return PromptTemplate(template=prompt, input_variables=["context", "query"])

    @staticmethod
    def doc_prompt():
        return PromptTemplate(input_variables=["page_content"], template="{page_content}")

    @staticmethod
    def simple_rag_with_system(human_msg:str,
                               input_variables:List[str],
                               system_msg:str=""):
        """
        Chat template with system prompt.
        Example of human_msg:
        "You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the
        question. If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise.
        \nQuestion: {question} \nContext: {context} \nAnswer:"

        input_variables: list of variable, in the example above, these are "question", "context"
        """
        if system_msg != '':
            sys_msg_t = SystemMessage(
                content=(system_msg)
            )
        else:
            sys_msg_t = None
        human_msg_t = HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=input_variables,
                                                                     template=human_msg))
        if sys_msg_t:
            return ChatPromptTemplate.from_messages([sys_msg_t, human_msg])
        else:
            return ChatPromptTemplate.from_messages([human_msg_t])


