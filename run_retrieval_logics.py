import os
from typing import List, Union, Tuple
import argparse
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from rouge import Rouge
import openai

from langchain import hub
from langchain.retrievers import EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from sagemaker.session import Session
from sagemaker.analytics import ExperimentAnalytics
from sagemaker.experiments import Run

from utils import load_config
from run_fine_tune_model import Inferencer
from rag_logger import RAGLogger
from constants import SourceUrl

logger = RAGLogger(__name__).get_logger()

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")

class PDFLoader:
    """Load PDF source file
    """
    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

    def load(self, url: str) -> List[Document]:
        """
        Loads and returns the pages of a PDF document from a given URL.

        Args:
            url (str): The URL of the PDF document to load.

        Returns:
            List[Document]: A list of pages loaded from the PDF document.
        """
        loader = PyMuPDFLoader(url, headers=self.headers)
        pages = loader.load()
        return pages


class VectorStore:
    """Manages embedding models for vector representations of documents, supporting OpenAI and Cohere embeddings."""
    def __init__(self, emb_model_name: str):
        self.emb_model_name = emb_model_name

    def get_embeddings_model(self) -> Union[OpenAIEmbeddings, CohereEmbeddings]:
        """
        Selects and returns the embeddings model based on the initialization parameter.

        Raises:
            Exception: If the specified embedding model name is not supported.

        Returns:
            Union[OpenAIEmbeddings, CohereEmbeddings]: An instance of the embeddings model.
        """
        if self.emb_model_name.lower() == "openai":
            logger.info("Use OpenAI embedding")
            return OpenAIEmbeddings()
        elif self.emb_model_name.lower() == "cohere":
            logger.info("Use Cohere embedding")
            return CohereEmbeddings(model="embed-english-light-v3.0")
        else:
            raise Exception("Specified embedding model not implemented.")
    

class RetrieverConstructor:
    """_summary_
    """
    def __init__(self, retriever_type, top_k):
        self.retriever_type = retriever_type
        self.top_k = top_k

    def get_vector_retriever(
            self, documents: List[Document],
            embeddings_model: Embeddings
        ):
        """
        Constructs a vector-based retriever using FAISS from provided documents and an embeddings model.

        Args:
            documents (List[Document]): A list of documents to index.
            embeddings_model: The embeddings model to use for generating document vectors.

        Returns:
            FAISS: A FAISS vector store configured as a retriever.
        """
        vectorstore = FAISS.from_documents(documents, embeddings_model)
        return vectorstore.as_retriever(search_kwargs={"k": self.top_k})
    
    def get_keyword_retriever(self, documents: List[Document]):
        """
        Constructs a keyword-based retriever using BM25 from provided documents.

        Args:
            documents (List[str]): A list of documents to index.

        Returns:
            BM25Retriever: A BM25Retriever instance.
        """
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k =  self.top_k
        return bm25_retriever

    def get_retriever(self, documents: List[Document], embeddings_model: Embeddings):
        """
        Constructs and returns a retriever based on the specified type (vector, hybrid).

        Args:
            documents (List[Document]): A list of documents to index.
            embeddings_model: The embeddings model for vector representations.

        Raises:
            Exception: If the specified retriever type is not supported.

        Returns:
            Union[FAISS, EnsembleRetriever]: A retriever instance.
        """
        vector_retriever = self.get_vector_retriever(documents, embeddings_model)
        if self.retriever_type.lower() == "vector":
            return vector_retriever
        elif self.retriever_type.lower() == "hybrid":
            kw_retriever = self.get_keyword_retriever(documents)
            ensemble_retriever = EnsembleRetriever(retrievers=[vector_retriever, kw_retriever],
                                       weights=[0.4, 0.6])
            return ensemble_retriever
        else:
            raise Exception(f"Requested retriever type {self.retriever_type} not implemented.")

class Generator:
    """Facilitates the generation of text using language models, supporting both OpenAI models and fine-tuned models."""
    def __init__(self, llm_type, fine_tune_cfg_fp):
        self.llm_type = llm_type
        self.fine_tune_cfg_fp = fine_tune_cfg_fp

    def get_llm(self) -> Union[ChatOpenAI, Inferencer]:
        """
        Selects and returns the language model based on the initialization parameter.

        Raises:
            Exception: If the specified LLM type is not supported.

        Returns:
            Union[ChatOpenAI, Inferencer]: An instance of the language model.
        """
        if self.llm_type.lower() == "openai":
            return ChatOpenAI(model="gpt-4")
        elif self.llm_type.lower() == "finetuned":
            return Inferencer(self.fine_tune_cfg_fp)
        else:
            raise Exception("Requested LLM not implemented.")


class RAGChainExecutor:
    """Executes an RAG chain by retrieving relevant documents and generating answers based on a given prompt."""
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.prompt = hub.pull("rlm/rag-prompt")
        self.generator = generator

    def format_docs(self, docs) -> str:
        """
        Formats the content of retrieved documents into a single string.

        Args:
            docs (List[PyMuPDFLoader.Page]): A list of document pages.

        Returns:
            str: A string containing the formatted document contents.
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def invoke(self, question: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Invokes the RAG chain process to generate answers for a given question or list of questions.

        Args:
            question (Union[str, List[str]]): The question(s) to generate answers for.

        Returns:
            Union[str, List[str]]: The generated answer(s) for the input question(s).
        """

        rag_chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.generator
            | StrOutputParser()
        )

        if isinstance(question, list):
            return rag_chain.batch(question)
        return rag_chain.invoke(question)


class RAGRunner:
    """_summary_
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg

        self.loader = PDFLoader()
        self.vectorstore = VectorStore(self.cfg['emb_model_name'])        
        self.retriever_constructor = RetrieverConstructor(self.cfg['retriever']['retriever_type'], self.cfg['retriever']['top_k'])
        self.generator = Generator(self.cfg['generator']['llm_type'], self.cfg['generator'].get('fine_tune_cfg_path', ''))

    def run(self, source_url: str, question: str) -> Union[str, List[str]]:
        """
        Executes the RAG process, from loading documents from a URL to generating an answer.

        Args:
            source_url (str): The URL of the source document.
            question (str): The question to answer.

        Returns:
            str: The generated answer to the question.
        """
        pages = self.loader.load(source_url)
        emb_model = self.vectorstore.get_embeddings_model()
        retriever = self.retriever_constructor.get_retriever(pages, emb_model)
        llm = self.generator.get_llm()
        rag_executor = RAGChainExecutor(retriever, llm)
        
        answer = rag_executor.invoke(question)
        return answer


class Evaluator:
    """Evaluates the performance of the RAG pipeline using the Rouge metric on a given test dataset."""
    def __init__(self, configs: dict):
        self.rouge = Rouge()
        self.configs = configs

    def load_test_dataset(self, cfg: dict) -> Tuple[List[str], List[str]]:
        """
        Loads a test dataset from a CSV file specified in the configuration.

        Args:
            cfg (dict): Configuration details including the path to the dataset file.

        Returns:
            Tuple[List[str], List[str]]: A tuple containing lists of questions and their corresponding reference answers.
        """
        test_data = pd.read_csv(cfg['evaluator']['data_file'])
        questions = test_data['Question'].to_list()
        reference_answers = test_data['Answer'].to_list()
        return questions, reference_answers

    def evaluate_all(self) -> List:
        """
        Evaluates the RAG process across multiple configurations and returns the Rouge scores.

        Returns:
            List[dict]: A list of dictionaries containing evaluation scores for each configuration.
        """
        results = []

        for cfg in self.configs:
            scores = self.evaluate_single(cfg)
            results.append(scores)

        return results
    
    def log_evaluation(self, run: Run, cfg: dict, scores: dict):
        """Logs the evaluation scores."""
        run.log_parameters(cfg)
        for rouge_metric, values in scores.items():
            for metric_type, value in values.items():
                metric_name = f'{rouge_metric}_{metric_type}'
                run.log_metric(metric_name, value)

    def report_metrics(self) -> pd.DataFrame:
        """Reports the metrics of the evaluation using Sagemaker"""
        sagemaker_session = Session()
        analytics = ExperimentAnalytics(
            sagemaker_session=sagemaker_session,
            experiment_name='rag-exp'
        )
        df = analytics.dataframe()
        return df

    def evaluate_single(self, cfg: dict):
        """
        Evaluates the RAG process for a single configuration and returns the Rouge scores.

        Args:
            cfg (dict): The configuration to evaluate.

        Returns:
            dict: A dictionary containing the Rouge scores.
        """
        """Evaluates the RAG process for a single configuration and returns the Rouge scores."""
        logger.info('Start evaluating')
        logger.info(cfg)

        questions, reference_answers = self.load_test_dataset(cfg)

        cur_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        with Run(experiment_name=cfg['source'], run_name=f"run-{cur_time}") as run:
            runner = RAGRunner(cfg)
            answers = runner.run(source_url=SourceUrl[cfg['source'].upper()].value, question=questions)

            scores = self.rouge.get_scores(answers, reference_answers, avg=True)
            self.log_evaluation(run, cfg, scores)

        metrics_df = self.report_metrics()
        logger.info(scores)

        return scores

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Process command line arguments.')

    # Add arguments
    parser.add_argument('--config_files', required=True, nargs='+', help='List of config files defining RAG frameworks to be compared')

    # Parse arguments
    args = parser.parse_args()

    configurations = [load_config(fp) for fp in args.config_files]

    evaluator = Evaluator(configurations)
    all_results = evaluator.evaluate_all()
