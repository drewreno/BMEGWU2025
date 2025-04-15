import os
import gzip
import tqdm
import json

def ends_with_ending_punctuation(s):
    ending_punctuation = ('.', '?', '!')
    return any(s.endswith(char) for char in ending_punctuation)

def concat(title, content):
    if ends_with_ending_punctuation(title.strip()):
        return title.strip() + " " + content.strip()
    else:
        return title.strip() + ". " + content.strip()

def extract(gz_fpath):
    titles = []
    abstracts = []
    title = ""
    abs = ""
    ids = []

    # Specify encoding='utf-8' to avoid UnicodeDecodeError
    with gzip.open(gz_fpath, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == "<Article>" or line.startswith("<Article "):
                title = ""
                abs = ""
            elif line == "</Article>":
                if abs.strip() == "":
                    continue
                titles.append(title)
                abstracts.append(abs)
                ids.append(id)
            elif line.startswith("<PMID"):
                id = line.strip("</PMID>").split(">")[-1]
            elif line.startswith("<ArticleTitle>"):
                title = line[14:-15]
            elif line.startswith("<AbstractText"):
                if len(abs) == 0: 
                    abs += "".join(line[13:-15].split('>')[1:])
                else:
                    abs += " " + "".join(line[13:-15].split('>')[1:])

    return titles, abstracts, ids

if __name__ == "__main__":
    fnames = sorted([fname for fname in os.listdir("corpus/pubmed/baseline") if fname.endswith("xml.gz")])
    
    if not os.path.exists("corpus/pubmed/chunk"):
        os.makedirs("corpus/pubmed/chunk")

    for fname in tqdm.tqdm(fnames):
        if os.path.exists(f"corpus/pubmed/chunk/{fname.replace('.xml.gz', '.jsonl')}"):
            continue
        gz_fpath = os.path.join("corpus/pubmed/baseline", fname)
        titles, abstracts, ids = extract(gz_fpath)
        saved_text = [
            json.dumps({
                "id": f"PMID:{ids[i]}",
                "title": titles[i],
                "content": abstracts[i],
                "contents": concat(titles[i], abstracts[i])
            }) for i in range(len(titles))
        ]
        with open(f"corpus/pubmed/chunk/{fname.replace('.xml.gz', '.jsonl')}", 'w') as f:
            f.write('\n'.join(saved_text))
