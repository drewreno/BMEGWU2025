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

from transformers import MllamaForConditionalGeneration, AutoProcessor, AutoConfig, AutoTokenizer
from accelerate import init_empty_weights, infer_auto_device_map

from utils import RetrievalSystem, DocExtracter
from templateUI import general_medrag_system, general_medrag, options_text, temperature_G, MAX, IMG_MAX, CHUNK_MAX
from config import config

# Configure the root logger to capture detailed startup logs
logging.basicConfig(
    level=logging.DEBUG,
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
    images = []
    logging.debug(f"Loading {len(image_paths)} images.")
    for image_path in image_paths:
        try:
            img = Image.open(image_path)
            images.append(img)
            logging.debug(f"Image '{image_path}' loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading image '{image_path}': {e}")
            continue
    return images


class MedRAG:
    def __init__(
        self,
        rag=True,
        retriever_name="MedCPT",
        corpus_name="Textbooks",
        db_dir="./corpus",
        cache_dir=None,
        model_id="meta-llama/Llama-3.2-11B-Vision-Instruct"
    ):
        self.rag = rag
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        self.cache_dir = cache_dir
        self.docExt = None

        logging.debug(
            f"Initializing MedRAG instance with parameters: rag={rag}, retriever_name={retriever_name}, "
            f"corpus_name={corpus_name}, db_dir={db_dir}, model_id={model_id}"
        )

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

        try:
            logging.debug(f"Loading LLama model with model_id: {model_id}")
            model_config = AutoConfig.from_pretrained(model_id)
            with init_empty_weights():
                model_empty = MllamaForConditionalGeneration(model_config)
            model_empty.tie_weights()
            device_map = infer_auto_device_map(model_empty)
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                offload_folder="offload",
                offload_state_dict=True
            )
            self.model.tie_weights()
            logging.debug("LLama model loaded successfully.")
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.text_tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        except Exception as e:
            logging.error(f"Failed to load LLama model or processor: {e}")
            sys.exit(1)

    def answer(
        self,
        question,
        k=CHUNK_MAX,
        rrf_k=100,
        save_dir=None,
        snippets=None,
        snippets_ids=None,
        pil_images=None
    ):
        logging.debug("Entered answer method.")
        if self.rag:
            try:
                retrieved_snippets, scores = self.retrieval_system.retrieve(question, k=k, rrf_k=rrf_k)
            except:
                retrieved_snippets, scores = [], []
            contexts = [f"Document [{i}] (Title: {s['title']}) {s['content']}" for i, s in enumerate(retrieved_snippets)]
            context_text = "\n".join(contexts) or ""
            tokenizer = tiktoken.get_encoding("cl100k_base")
            context_text = tokenizer.decode(tokenizer.encode(context_text)[:MAX])
        else:
            retrieved_snippets, scores = [], []
            context_text = ""

        prompt = self.templates["medrag_prompt"].format(
            context=context_text, question=question, options=options_text
        )
        messages = [
            {"role": "system", "content": self.templates["medrag_system"]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        ans = self.generate(messages)
        if not ans:
            return None, retrieved_snippets, scores

        cleaned = re.sub(r"\s+", " ", ans).strip()

        # Print full input and uncut output to terminal
        print("\n=== Full Input ===\n")
        print(question)
        print("\n=== Uncut Output ===\n")
        print(cleaned)

        # Primary extraction: JSON block
        json_match = re.search(r"(\{[\s\S]*?\})", cleaned)
        if json_match:
            return json_match.group(1).strip(), retrieved_snippets, scores

        # Fallback extraction: from 'assistant {' onward
        fallback_match = re.search(r"assistant\s*(\{[\s\S]*)", cleaned)
        if fallback_match:
            return fallback_match.group(1).strip(), retrieved_snippets, scores

        # If all else fails, return entire cleaned output
        return cleaned, retrieved_snippets, scores

    def generate(self, messages):
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        tokens = self.text_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        output = self.model.generate(**tokens, max_new_tokens=MAX)
        return self.text_tokenizer.decode(output[0], skip_special_tokens=True)


def process_selected_files(selected_images: List[str], selected_texts: List[str]) -> str:
    pil_images = load_images(selected_images)
    combined = ""
    for tf in selected_texts:
        try:
            combined += open(tf, 'r', encoding='utf-8').read() + "\n"
        except:
            continue
    medrag = MedRAG()
    answer, _, _ = medrag.answer(question=combined, pil_images=pil_images)
    return answer or "No answer generated."
