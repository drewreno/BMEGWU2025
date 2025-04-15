import os
import re
import json
import sys
import logging
from io import BytesIO
from typing import List
from PIL import Image
import torch
import tiktoken

from transformers import MllamaForConditionalGeneration, AutoProcessor

from utils import RetrievalSystem, DocExtracter
from templateUI import general_medrag_system, general_medrag, options_text, temperature_G, MAX, IMG_MAX, CHUNK_MAX
from config import config

# Configure the root logger to capture detailed startup logs
logging.basicConfig(
    level=logging.DEBUG,  # Capture all logs from your application
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logging.debug("Application startup initiated.")

# Disable propagation for specific loggers to reduce verbosity
for logger_name in ["httpcore", "httpx", "urllib3", "openai"]:
    logging.getLogger(logger_name).propagate = False

def image_paths_fn(max_images: int = IMG_MAX) -> list:
    images_folder = "corpus/images"
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    logging.debug(f"Looking for images in folder: {images_folder} with extensions {supported_extensions}")

    try:
        if not os.path.isdir(images_folder):
            raise FileNotFoundError(f"The directory '{images_folder}' does not exist.")

        image_files = [
            f for f in os.listdir(images_folder)
            if f.lower().endswith(supported_extensions) and os.path.isfile(os.path.join(images_folder, f))
        ]
        logging.debug(f"Found image files: {image_files}")

        if not image_files:
            raise FileNotFoundError(f"No images found in '{images_folder}' with extensions {supported_extensions}.")

        selected_files = image_files[:max_images]
        image_paths = [os.path.join(images_folder, f) for f in selected_files]
        logging.debug(f"Selected image paths: {image_paths}")
        return image_paths

    except FileNotFoundError as e:
        logging.error(f"Error: {e}")
        return []
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return []

def load_images(image_paths: List[str]) -> List[Image.Image]:
    """
    Loads images from the given file paths as PIL Image objects.
    """
    images = []
    logging.debug(f"Loading {len(image_paths)} images.")
    for image_path in image_paths:
        try:
            img = Image.open(image_path)
            images.append(img)
            logging.debug(f"Image '{image_path}' loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading image '{image_path}': {e}")
            continue  # Skip problematic images
    return images

class MedRAG:
    def __init__(self, rag=True, retriever_name="MedCPT", corpus_name="Textbooks", db_dir="./corpus", cache_dir=None,
                 model_id="meta-llama/Llama-3.2-11B-Vision-Instruct"):
        self.rag = rag
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        self.cache_dir = cache_dir
        self.docExt = None

        logging.debug(f"Initializing MedRAG instance with parameters: rag={rag}, retriever_name={retriever_name}, "
                      f"corpus_name={corpus_name}, db_dir={db_dir}, model_id={model_id}")

        if rag:
            try:
                self.retrieval_system = RetrievalSystem(self.retriever_name, self.corpus_name, self.db_dir)
                logging.debug("Retrieval system initialized successfully.")
            except Exception as e:
                logging.error(f"Failed to initialize RetrievalSystem: {e}")
                self.retrieval_system = None
        else:
            self.retrieval_system = None

        self.templates = {
            "medrag_system": general_medrag_system,
            "medrag_prompt": general_medrag
        }

        # Load the LLama model and processor with additional debugging messages
        try:
            logging.debug(f"Loading LLama model with model_id: {model_id}")
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            logging.debug("LLama model loaded successfully.")
            logging.debug("Loading LLama processor.")
            self.processor = AutoProcessor.from_pretrained(model_id)
            logging.debug("LLama processor loaded successfully.")
        except Exception as e:
            logging.error("Failed to load LLama model or processor: %s", e)
            sys.exit(1)

    def answer(self, question, k=CHUNK_MAX, rrf_k=100, save_dir=None, snippets=None, snippets_ids=None, pil_images=None):
        '''
        Generates an answer given the question and optionally additional text/image inputs.
        
        Args:
            question (str): The question to be answered.
            k (int): The number of snippets to retrieve.
            rrf_k (int): Parameter for Reciprocal Rank Fusion.
            save_dir (str): Directory to save the results.
            snippets (List[Dict]): List of provided snippets.
            snippets_ids (List[Dict]): List of snippet IDs to extract content.
            pil_images (List[PIL.Image.Image]): List of PIL image objects.
        '''
        logging.debug("Entered answer method.")
        logging.debug(f"Received question: {question[:100]}...")  # Log first 100 characters

        if self.rag:
            logging.debug("RAG (Retrieval Augmented Generation) is enabled.")
            if snippets is not None:
                retrieved_snippets = snippets[:k]
                scores = []
                logging.debug("Using provided snippets directly.")
            elif snippets_ids is not None:
                if self.docExt is None:
                    self.docExt = DocExtracter(db_dir=self.db_dir, cache=True, corpus_name=self.corpus_name)
                retrieved_snippets = self.docExt.extract(snippets_ids[:k])
                scores = []
                logging.debug("Extracted snippets using snippet IDs.")
            else:
                try:
                    assert self.retrieval_system is not None, "Retrieval system is not initialized."
                    retrieved_snippets, scores = self.retrieval_system.retrieve(question, k=k, rrf_k=rrf_k)
                    logging.debug(f"Retrieved snippets: {retrieved_snippets}")
                except Exception as e:
                    logging.error(f"Error during snippet retrieval: {e}")
                    retrieved_snippets, scores = [], []

            contexts = [
                "Document [{:d}] (Title: {:s}) {:s}".format(
                    idx, retrieved_snippets[idx]["title"], retrieved_snippets[idx]["content"]
                ) for idx in range(len(retrieved_snippets))
            ]
            if len(contexts) == 0:
                logging.debug("No contexts retrieved, setting default empty context.")
                contexts = [""]
            contexts_text = "\n".join(contexts)
            logging.debug(f"Combined context text: {contexts_text[:200]}...")  # First 200 characters

            self.max_length = 32768
            self.context_length = MAX
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            contexts = [self.tokenizer.decode(self.tokenizer.encode(contexts_text)[:self.context_length])]
            logging.debug(f"Tokenized context: {contexts}")
        else:
            retrieved_snippets = []
            scores = []
            contexts = []
            logging.debug("RAG is disabled; skipping snippet retrieval.")

        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                logging.debug(f"Created save directory: {save_dir}")

        # Generate answer(s)
        answers = []
        for context in contexts:
            # Updated: Use Python's format() instead of .render()
            prompt_medrag = self.templates["medrag_prompt"].format(
                context=context, question=question, options=options_text
            )
            logging.debug(f"Rendered prompt (MedRAG): {prompt_medrag[:200]}...")  # Beginning of prompt
            
            if pil_images:
                logging.debug("Processing PIL images for message content.")
                image_contents = [{"type": "image", "image": img} for img in pil_images]
            else:
                image_contents = []
                logging.debug("No PIL images provided.")

            messages = [
                {
                    "role": "system",
                    "content": self.templates["medrag_system"]
                },
                {
                    "role": "user",
                    "content": (
                        [{"type": "text", "text": prompt_medrag}] + image_contents
                    )
                }
            ]
            logging.debug(f"Constructed messages for LLama model: {messages}")
            ans = self.generate(messages)
            if ans is not None:
                cleaned_ans = re.sub(r"\s+", " ", ans)
                answers.append(cleaned_ans)
                logging.debug("Answer generated and cleaned successfully.")
            else:
                logging.error("Failed to generate an answer.")

        if save_dir is not None:
            try:
                with open(os.path.join(save_dir, "snippets.json"), 'w') as f:
                    json.dump(retrieved_snippets, f, indent=4)
                with open(os.path.join(save_dir, "response.json"), 'w') as f:
                    json.dump(answers, f, indent=4)
                logging.debug("Saved snippets and response JSON files successfully.")
            except Exception as e:
                logging.error(f"Error saving files to {save_dir}: {e}")

        if len(answers) == 0:
            logging.error("No answers were generated.")
            return None, retrieved_snippets, scores

        logging.debug(f"Exiting answer method with answers: {answers}")
        return answers[0] if len(answers) == 1 else answers, retrieved_snippets, scores


    def generate(self, messages):
        '''
        Generate response given messages using the LLama model.
        '''
        logging.debug("Starting generation with provided messages.")
        image = None
        user_contents = messages[1]["content"]
        logging.debug(f"User message contents: {user_contents}")

        for item in user_contents:
            if item.get("type") == "image":
                try:
                    image = item.get("image")
                    logging.debug("Image obtained directly for LLama.")
                except Exception as e:
                    logging.error(f"Error retrieving PIL image: {e}")
                break

        # Prepare the input text using the processor’s chat template
        try:
            input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            logging.debug(f"Processed input text: {input_text[:200]}...")  # Log beginning of input text
        except Exception as e:
            logging.error(f"Error applying chat template: {e}")
            input_text = ""

        # Prepare inputs for the model, with or without image data
        try:
            if image:
                inputs = self.processor(image, input_text, add_special_tokens=False, return_tensors="pt").to(self.model.device)
                logging.debug("Prepared inputs with image data for LLama model.")
            else:
                inputs = self.processor(input_text, add_special_tokens=False, return_tensors="pt").to(self.model.device)
                logging.debug("Prepared inputs without image data for LLama model.")
        except Exception as e:
            logging.error(f"Error preparing inputs for the model: {e}")
            return None

        # Generate the answer using the LLama model
        try:
            output = self.model.generate(**inputs, max_new_tokens=MAX)
            answer = self.processor.decode(output[0])
            logging.debug(f"LLama generation completed with answer: {answer[:200]}...")  # Log beginning of answer
        except Exception as e:
            logging.error(f"Error during model generation: {e}")
            answer = None

        return answer

def process_selected_files(selected_images: List[str], selected_texts: List[str]) -> str:
    """
    Processes the selected image and text files.
    
    Args:
        selected_images (List[str]): List of image file paths.
        selected_texts (List[str]): List of text file paths.
    
    Returns:
        str: Output message or result from processing.
    """
    logging.debug("Starting process_selected_files.")
    try:
        # Load selected images as PIL images
        pil_images = load_images(selected_images)
        logging.info(f"Loaded {len(pil_images)} images.")

        # Read and concatenate text files
        combined_text = ""
        for text_file in selected_texts:
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    combined_text += content + "\n"
                    logging.debug(f"Successfully read content from {text_file}.")
            except Exception as e:
                logging.error(f"Error reading text file '{text_file}': {e}")
                continue  # Skip problematic text files

        if not combined_text:
            logging.warning("No text content was read from the selected files.")

        # Instantiate MedRAG with detailed logging in its __init__
        medrag = MedRAG()

        # Use the combined text as the question
        question = combined_text
        logging.debug("Calling MedRAG.answer with combined text as the question.")

        # Call the answer method with the loaded PIL images
        answer, retrieved_snippets, scores = medrag.answer(
            question=question,
            pil_images=pil_images
        )

        if answer:
            logging.info("Answer generated successfully.")
            return answer
        else:
            logging.warning("No answer was generated.")
            return "No answer was generated."

    except Exception as e:
        logging.error(f"An error occurred in process_selected_files: {e}")
        return f"An error occurred: {e}"

# Log that the module has loaded successfully
logging.debug("Module loaded successfully and ready to process files.")
