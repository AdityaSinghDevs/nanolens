from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent/ "training"

text_path = BASE_DIR / "input.txt"

texts = []

for txt_file in sorted(BASE_DIR.glob("*.txt")):
    with open(txt_file, "r", encoding="utf-8") as f:
        texts.append(f.read())

text = "\n\n".join(texts)

chars = sorted(list(set(text)))
vocab_size = len(chars)

#Char level tokenization
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

if __name__ == "__main__":
    print(f"Total characters: {len(text):,}")
    print(f"Vocab size: {vocab_size}")
    print(f"Sample (first 200 chars):\n{text[:200]}")