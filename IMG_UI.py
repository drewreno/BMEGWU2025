import os
import re
import json
import openai
import tiktoken
import sys
import logging
import base64
from typing import List
from utils import RetrievalSystem, DocExtracter
from templateUI import general_medrag_system, general_medrag, options_text, temperature_G, MAX, IMG_MAX, CHUNK_MAX
from config import config

# Configure the root logger to handle your application's logs
logging.basicConfig(
    level=logging.DEBUG,  # Capture all logs from your application
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Output logs to the terminal
    ]
)

# Disable propagation for specific loggers to reduce verbosity
for logger_name in ["httpcore", "httpx", "urllib3", "openai"]:
    logging.getLogger(logger_name).propagate = False

def image_paths_fn(max_images: int = IMG_MAX) -> list:
    images_folder = "corpus/images"
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    try:
        # Ensure the images_folder exists
        if not os.path.isdir(images_folder):
            raise FileNotFoundError(f"The directory '{images_folder}' does not exist.")

        # Get a list of image filenames with supported extensions
        image_files = [
            f for f in os.listdir(images_folder)
            if f.lower().endswith(supported_extensions) and os.path.isfile(os.path.join(images_folder, f))
        ]

        if not image_files:
            raise FileNotFoundError(f"No images found in '{images_folder}' with extensions {supported_extensions}.")

        # Select up to `max_images` filenames
        selected_files = image_files[:max_images]

        # Create full paths for each selected image file
        image_paths = [os.path.join(images_folder, f) for f in selected_files]

        return image_paths

    except FileNotFoundError as e:
        logging.error(f"Error: {e}")
        return []
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return []

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

# Function to encode images (modified to handle multiple images)
def encode_images(image_paths):
    base64_images = []
    for image_path in image_paths:
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                base64_images.append(encoded_string)
                logging.debug(f"Image {image_path} encoded successfully.")
        except FileNotFoundError:
            logging.error(f"Image file not found: {image_path}")
            continue  # Skip missing images
        except Exception as e:
            logging.error(f"Error encoding image {image_path}: {e}")
            continue  # Skip problematic images
    return base64_images

class MedRAG:

    def __init__(self, rag=True, retriever_name="MedCPT", corpus_name="Textbooks", db_dir="./corpus", cache_dir=None):
        self.rag = rag
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        self.cache_dir = cache_dir
        self.docExt = None

        logging.debug("Initializing MedRAG instance with LLM: %s", config["api_type"])

        if rag:
            self.retrieval_system = RetrievalSystem(self.retriever_name, self.corpus_name, self.db_dir)
        else:
            self.retrieval_system = None
        self.templates = {
            "medrag_system": general_medrag_system,
            "medrag_prompt": general_medrag
        }


    def answer(self, question, k=CHUNK_MAX, rrf_k=100, save_dir=None, snippets=None, snippets_ids=None, base64_images=None):
        '''
        question (str): question to be answered
        k (int): number of snippets to retrieve (now defaulting to MAX)
        rrf_k (int): parameter for Reciprocal Rank Fusion
        save_dir (str): directory to save the results
        snippets (List[Dict]): list of snippets to be used
        snippets_ids (List[Dict]): list of snippet ids to be used
        base64_images (List[str]): list of base64-encoded images
        '''

        logging.debug("Entered the answer method.")
        logging.debug("Question: %s", question)

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

            self.max_length = 32768
            self.context_length = MAX
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
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
        for context in contexts:
            prompt_medrag = self.templates["medrag_prompt"].render(
                context=context, question=question, options=options_text  # Options defined inside the template
            )
            logging.debug("Rendered prompt (MedRAG): %s", prompt_medrag)
            if base64_images:
                image_contents = [
                    {"type": "image_url",
                     "image_url": {
                         "url": f"data:image/jpeg;base64,{img}",
                         "detail": "high"
                     }
                    } for img in base64_images
                ]
            else:
                image_contents = []

            messages = [
                {
                    "role": "system",
                    "content": self.templates["medrag_system"]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_medrag}
                    ] + image_contents
                }
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

        response = openai.chat.completions.create(
                model=config["api_type"],
                messages=messages,
                # temperature=temperature_G, //not supported in o1
                max_completion_tokens=2000,
        )
        logging.debug("Received response: %s", response)
        ans = response.choices[0].message.content
        return ans


def process_selected_files(selected_images: List[str], selected_texts: List[str]) -> str:
    """
    Processes the selected image and text files.

    Args:
        selected_images (List[str]): List of image file paths.
        selected_texts (List[str]): List of text file paths.

    Returns:
        str: Output message or result from processing.
    """
    try:
        # Encode selected images
        base64_images = encode_images(selected_images)
        logging.info(f"Encoded {len(base64_images)} images.")

        # Read and concatenate text files
        combined_text = ""
        for text_file in selected_texts:
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    combined_text += content + "\n"
                    logging.debug(f"Read content from {text_file}.")
            except Exception as e:
                logging.error(f"Error reading text file {text_file}: {e}")
                continue  # Skip problematic text files

        if not combined_text:
            logging.warning("No text content was read from the selected files.")

        # Instantiate MedRAG
        medrag = MedRAG()

        # Assign the combined text to the question variable
        question = combined_text

        # Call the answer method
        answer, retrieved_snippets, scores = medrag.answer(
            question=question,
            base64_images=base64_images
        )

        if answer:
            logging.info("Answer generated successfully.")
            return answer
        else:
            logging.warning("No answer was generated.")
            return "No answer was generated."

    except Exception as e:
        logging.error(f"An error occurred in processing: {e}")
        return f"An error occurred: {e}"
