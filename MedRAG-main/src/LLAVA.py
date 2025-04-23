import os
import re
import json
import sys
import logging
from io import BytesIO
from typing import List
from PIL import Image
import torch

from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from utils import RetrievalSystem, DocExtracter
from templateUI import (
    general_medrag_system,
    general_medrag,
    options_text,
    temperature_G,
    MAX,
    IMG_MAX,
    CHUNK_MAX,
)
from config import config

# 1) Logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
for logger_name in ["httpcore", "httpx", "urllib3", "openai"]:
    logging.getLogger(logger_name).propagate = False

# 2) Image discovery helper
def image_paths_fn(max_images: int = IMG_MAX) -> List[str]:
    images_folder = "corpus/images"
    supported_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    logging.debug(f"Looking in {images_folder} for {supported_extensions}")

    try:
        if not os.path.isdir(images_folder):
            raise FileNotFoundError(f"{images_folder} not found")
        files = [
            f
            for f in os.listdir(images_folder)
            if f.lower().endswith(supported_extensions)
            and os.path.isfile(os.path.join(images_folder, f))
        ]
        if not files:
            raise FileNotFoundError("No supported images")
        return [os.path.join(images_folder, f) for f in files[:max_images]]
    except Exception as e:
        logging.error(e)
        return []

# 3) Image loader
def load_images(image_paths: List[str]) -> List[Image.Image]:
    images = []
    for path in image_paths:
        try:
            images.append(Image.open(path))
        except Exception as e:
            logging.error(f"Failed to load {path}: {e}")
    return images

# 4) MedRAG core
class MedRAG:
    def __init__(
        self,
        rag: bool = True,
        retriever_name: str = "MedCPT",
        corpus_name: str = "Textbooks",
        db_dir: str = "./corpus",
        cache_dir: str = None,
        model_id: str = r"C:/Users/Andrew Cassarino/LLaVA-Med", # microsoft-llava-med-v1.5-mistral-7b
    ):
        self.rag = rag
        logging.debug(f"Init MedRAG(rag={rag}, model={model_id})")

        # 4.1) Optional RAG retrieval
        if rag:
            try:
                self.retrieval_system = RetrievalSystem(
                    retriever_name, corpus_name, db_dir
                )
                logging.debug("RetrievalSystem ready.")
            except Exception as e:
                logging.error(f"RetrievalSystem error: {e}")
                self.retrieval_system = None
        else:
            self.retrieval_system = None

        # 4.2) Load processor, tokenizer & model from local files only
        logging.debug("Loading AutoProcessor, AutoTokenizer & AutoModelForCausalLM")
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=True,
        )

        # ensure pad token exists for tokenizer (we won’t use processor padding)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )

        # resize embeddings if we added pad_token
        self.model.resize_token_embeddings(len(self.tokenizer))

        # 4.3) Set up new-generation-config (instead of mutating model.config)
        self.context_len = self.model.config.max_position_embeddings
        self.tokenizer.model_max_length = self.context_len

        gen_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=self.context_len
        )
        self.model.generation_config = gen_config

        logging.debug("Model loaded successfully.")

        self.templates = {
            "medrag_system": general_medrag_system,
            "medrag_prompt": general_medrag,
        }

    # 5) Generate from a single prompt string (not chat-dicts)
    def generate(self, text: str, pil_images: List[Image.Image] = None) -> str:
        # tokenize text
        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.context_len,
        )

        # preprocess images separately if provided
        if pil_images:
            image_inputs = self.processor.feature_extractor(
                images=pil_images,
                return_tensors="pt",
            )
        else:
            image_inputs = {}

        # merge inputs and move to device
        inputs = {
            **{k: v.to(self.model.device) for k, v in text_inputs.items()},
            **{k: v.to(self.model.device) for k, v in image_inputs.items()},
        }

        # generate
        output_ids = self.model.generate(**inputs, max_new_tokens=MAX)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # 6) Full Q&A pipeline
    def answer(
        self,
        question: str,
        k: int = CHUNK_MAX,
        rrf_k: int = 100,
        save_dir: str = None,
        snippets: List = None,
        snippets_ids: List = None,
        pil_images: List[Image.Image] = None,
    ):
        logging.debug("Answer() called.")
        # RAG retrieval
        if self.rag and self.retrieval_system:
            try:
                snippets, scores = self.retrieval_system.retrieve(
                    question, k=k, rrf_k=rrf_k
                )
            except:
                snippets, scores = [], []
            contexts = [
                f"Document [{i}] (Title: {s['title']}) {s['content']}"
                for i, s in enumerate(snippets)
            ]
            context_text = "\n".join(contexts)[:MAX]
        else:
            snippets, scores, context_text = [], [], ""

        # Build a single prompt string
        prompt = self.templates["medrag_prompt"].format(
            context=context_text, question=question, options=options_text
        )

        # Generate answer
        raw = self.generate(prompt, pil_images=pil_images)
        cleaned = re.sub(r"\s+", " ", raw).strip()

        print("\n=== Full Input ===\n", question)
        print("\n=== Full Output ===\n", cleaned)

        # Extract JSON if present
        m = re.search(r"(\{[\s\S]*?\})", cleaned)
        if m:
            return m.group(1), snippets, scores

        m2 = re.search(r"assistant\s*(\{[\s\S]*)", cleaned)
        if m2:
            return m2.group(1), snippets, scores

        return cleaned, snippets, scores

# 7) File-based wrapper
def process_selected_files(selected_images: List[str], selected_texts: List[str]) -> str:
    pil_images = load_images(selected_images)
    text = ""
    for tf in selected_texts:
        try:
            text += open(tf, encoding="utf-8").read() + "\n"
        except:
            continue
    answer, *_ = MedRAG().answer(question=text, pil_images=pil_images)
    return answer or "No answer generated."
