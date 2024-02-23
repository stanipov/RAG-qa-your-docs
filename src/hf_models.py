"""
Convenience wrapper to load a LLM and a tokenizer from HuggingFace
Contrastive search is explicitly used for the LLM generation.
Do your own modifications if you need another generation
strategy/parameters, just keep the API the same.

2024
"""


from langchain.llms import HuggingFacePipeline
from torch import bfloat16
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig


class Engine:
    def_bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    def __init__(self, model_id: str, cache_fld: str = None,
                 quant_config: BitsAndBytesConfig = None,
                 device_map: str = 'auto',
                 max_new_tokens: int = 2048,
                 top_k: int = 4,
                 penalty_alpha: float = 0.6
                 ):
        """
        Init the wrapper class.

        Assuming contrastive search for generation
        (see https://huggingface.co/blog/introducing-csearch)
        """
        if not quant_config:
            self.__quant_config = self.def_bnb_cfg
        else:
            self.__quant_config = quant_config

        self.__model_config = AutoConfig.from_pretrained(model_id)
        self.__model_id = model_id
        self.__device_map = device_map
        self.__cache_folder = cache_fld
        self.penalty_alpha = penalty_alpha
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens

    def load(self):
        """ Loads the model and the tokenizer """
        self.__load_llm()
        self.__load_tokenizer()

    def __load_llm(self):
        """ Loads HF model with quantization """
        self.llm_core_model = AutoModelForCausalLM.from_pretrained(
            self.__model_id,
            trust_remote_code=True,
            config=self.__model_config,
            quantization_config=self.__quant_config,
            device_map=self.__device_map,
            cache_dir=self.__cache_folder
        )

    def __load_tokenizer(self):
        """ Loads the tokenizer """
        self.tokenizer = AutoTokenizer.from_pretrained(self.__model_id, cache_dir=self.__cache_folder)

    def set_pipeline(self, batch_size: int = 4):
        """ Sets the generation pipeline. Change the contrastive search args if needed and call this method again"""
        pipe = pipeline(
            model=self.llm_core_model,
            tokenizer=self.tokenizer,
            task="text-generation",
            return_full_text=True,
            penalty_alpha=self.penalty_alpha,
            top_k=self.top_k,
            max_new_tokens=self.max_new_tokens
        )
        self.llm = HuggingFacePipeline(pipeline=pipe,
                                       batch_size=batch_size)

    def get_llm(self):
        return self.llm

    def get_tokenizer(self):
        return self.tokenizer