import os
import re
import json
import openai
import tiktoken
import sys
import logging
from utils import RetrievalSystem, DocExtracter, K
from template import *
from template import general_cot_system, general_cot, general_medrag_system, general_medrag, question, options
from config import config

sys.path.append("src")

temperature_G = 0.0 #0 to 2
MAX = 10000 #corpus token
MODEL2 = "gpt-4o" #adjust expectant

# Set up OpenAI API key with error handling
try:
    openai.api_key = config["api_key"]
    if not openai.api_key:
        raise ValueError("OpenAI API key is missing.")
    logging.debug("OpenAI API key has been set.")
except KeyError:
    logging.error("API key not found in the configuration.")
    sys.exit(1)
except ValueError as ve:
    logging.error(ve)
    sys.exit(1)

# Define OpenAI client function with exception handling
def openai_client(**kwargs):
    try:
        response = openai.ChatCompletion.create(**kwargs)
        logging.debug("Received response from OpenAI API: %s", response)
        return response["choices"][0]["message"]["content"]
    except openai.error.OpenAIError as e:
        logging.error("An error occurred while accessing the OpenAI API: %s", e)
        return None
    except Exception as e:
        logging.error("An unexpected error occurred: %s", e)
        return None

class MedRAG:

    def __init__(self, llm_name=config["api_type"], rag=True, retriever_name="MedCPT", corpus_name="Textbooks", db_dir="./corpus", cache_dir=None):
        self.llm_name = llm_name
        self.rag = rag
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        self.cache_dir = cache_dir
        self.docExt = None

        logging.debug("Initializing MedRAG instance with LLM: %s", self.llm_name)

        if rag:
            self.retrieval_system = RetrievalSystem(self.retriever_name, self.corpus_name, self.db_dir)
        else:
            self.retrieval_system = None
        self.templates = {
            "cot_system": general_cot_system,
            "cot_prompt": general_cot,
            "medrag_system": general_medrag_system,
            "medrag_prompt": general_medrag
        }
        if self.llm_name.split('/')[0].lower() == "openai":
            self.model = self.llm_name.split('/')[-1]
            if self.model not in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", MODEL2]:
                logging.error("Model %s is not supported or you don't have access to it.", self.model)
                sys.exit(1)
            if "gpt-3.5" in self.model or "gpt-35" in self.model:
                self.max_length = 16384
                self.context_length = MAX
            elif MODEL2 in self.model:
                self.max_length = 32768
                self.context_length = MAX
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            logging.debug("Model set to: %s", self.model)
        else:
            logging.error("Only OpenAI models are supported in this context.")
            sys.exit(1)

    def answer(self, question, options=None, k=K, rrf_k=100, save_dir=None, snippets=None, snippets_ids=None):
        '''
        question (str): question to be answered
        options (Dict[str, str]): options to be chosen from
        k (int): number of snippets to retrieve
        rrf_k (int): parameter for Reciprocal Rank Fusion
        save_dir (str): directory to save the results
        snippets (List[Dict]): list of snippets to be used
        snippets_ids (List[Dict]): list of snippet ids to be used
        '''

        logging.debug("Entered the answer method.")
        logging.debug("Question: %s", question)

        if options is not None:
            options_text = '\n'.join([key + ". " + options[key] for key in sorted(options.keys())])
        else:
            options_text = ''
        logging.debug("Options: %s", options_text)

        # Retrieve relevant snippets
        if self.rag:
            if snippets is not None:
                retrieved_snippets = snippets[:k]
                scores = []
            elif snippets_ids is not None:
                if self.docExt is None:
                    self.docExt = DocExtracter(db_dir=self.db_dir, cache=True, corpus_name=self.corpus_name)
                retrieved_snippets = self.docExt.extract(snippets_ids[:k])
                scores = []
            else:
                assert self.retrieval_system is not None
                retrieved_snippets, scores = self.retrieval_system.retrieve(question, k=k, rrf_k=rrf_k)
                logging.debug("Retrieved snippets: %s", retrieved_snippets)

            contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(
                idx, retrieved_snippets[idx]["title"], retrieved_snippets[idx]["content"]
            ) for idx in range(len(retrieved_snippets))]
            if len(contexts) == 0:
                contexts = [""]
            contexts_text = "\n".join(contexts)
            contexts = [self.tokenizer.decode(self.tokenizer.encode(contexts_text)[:self.context_length])]
            logging.debug("Contexts: %s", contexts)
        else:
            retrieved_snippets = []
            scores = []
            contexts = []

        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Generate answers
        answers = []
        if not self.rag:
            prompt_cot = self.templates["cot_prompt"].render(question=question, options=options_text)
            print("Rendered prompt (CoT):", prompt_cot)
            messages = [
                {"role": "system", "content": self.templates["cot_system"]},
                {"role": "user", "content": prompt_cot}
            ]
            logging.debug("Messages for OpenAI API: %s", messages)
            ans = self.generate(messages)
            if ans is not None:
                answers.append(re.sub("\\s+", " ", ans))
            else:
                logging.error("Failed to generate an answer.")
        else:
            for context in contexts:
                prompt_medrag = self.templates["medrag_prompt"].render(
                    context=context, question=question, options=options_text
                )
                print("Rendered prompt (MedRAG):", prompt_medrag)
                messages = [
                    {"role": "system", "content": self.templates["medrag_system"]},
                    {"role": "user", "content": prompt_medrag}
                ]
                logging.debug("Messages for OpenAI API: %s", messages)
                ans = self.generate(messages)
                if ans is not None:
                    answers.append(re.sub("\\s+", " ", ans))
                else:
                    logging.error("Failed to generate an answer.")

        if save_dir is not None:
            with open(os.path.join(save_dir, "snippets.json"), 'w') as f:
                json.dump(retrieved_snippets, f, indent=4)
            with open(os.path.join(save_dir, "response.json"), 'w') as f:
                json.dump(answers, f, indent=4)

        if len(answers) == 0:
            logging.error("No answers were generated.")
            return None, retrieved_snippets, scores

        logging.debug("Exiting the answer method with answers: %s", answers)
        return answers[0] if len(answers) == 1 else answers, retrieved_snippets, scores

    def generate(self, messages):
        '''
        Generate response given messages
        '''
        logging.debug("Generating response with messages: %s", messages)
        print("Generating response with messages:", messages)
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature_G,
            )
            logging.debug("Received response: %s", response)
            print("Received response:", response)
            ans = response.choices[0].message.content
            return ans
        except openai.error.APIConnectionError as e:
            logging.error("Network error while accessing OpenAI API: %s", e)
            return None
        except openai.error.RateLimitError as e:
            logging.error("Rate limit exceeded: %s", e)
            return None
        except openai.error.InvalidRequestError as e:
            logging.error("Invalid request: %s", e)
            return None
        except openai.error.OpenAIError as e:
            logging.error("OpenAI API error: %s", e)
            return None
        except Exception as e:
            logging.error("An unexpected error occurred: %s", e)
            raise  # Re-raise the exception to see the traceback

if __name__ == "__main__":
    # Instantiate the MedRAG class
    medrag = MedRAG()

    # Call the answer method
    answer, retrieved_snippets, scores = medrag.answer(question, options)

    # Print the answer
    print("Answer:", answer)

    # Write the question and answer to a text file
    with open('output.txt', 'w') as f:
        f.write(f"Prompt: {question}\n\n")
        f.write(f"Response: {answer}\n")
