import os
import shutil
import tqdm
import json
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

def remove_non_ascii(s):
    return ''.join(c for c in s if ord(c) < 128)

def ends_with_ending_punctuation(s):
    ending_punctuation = ('.', '?', '!')
    return any(s.strip().endswith(char) for char in ending_punctuation)

def concat(title, content):
    title = title.strip()
    content = content.strip()
    if ends_with_ending_punctuation(title):
        return f"{title} {content}"
    else:
        return f"{title}. {content}"

def empty_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

if __name__ == "__main__":
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    input_dir = "corpus/textbooks/en"
    chunk_dir = "corpus/textbooks/chunk"
    index_dir = "corpus/textbooks/index"
    fnames = sorted(os.listdir(input_dir))

    # Empty the chunk and index directories
    empty_directory(chunk_dir)
    empty_directory(index_dir)
    
    for fname in tqdm.tqdm(fnames):
        fpath = os.path.join(input_dir, fname)
        try:
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read().strip()
                if not text:
                    continue  # Skip if the file is empty
                # Remove non-ASCII characters from text
                text = remove_non_ascii(text)
                texts = text_splitter.split_text(text)
        except Exception as e:
            print(f"Error reading file {fpath}: {e}")
            continue

        saved_text = []
        for i, chunk in enumerate(texts):
            content = re.sub(r"\s+", " ", chunk).strip()
            if not content:
                continue  # Skip if content is empty
            # Remove non-ASCII characters from content
            content = remove_non_ascii(content)
            data = {
                "id": f"{fname.replace('.txt', '')}_{i}",
                "title": fname.replace(".txt", ""),
                "content": content,
                "contents": concat(fname.replace(".txt", ""), content)
            }
            saved_text.append(json.dumps(data))

        output_path = os.path.join(chunk_dir, fname.replace('.txt', '.jsonl'))
        with open(output_path, 'w') as f:
            f.write('\n'.join(saved_text))
