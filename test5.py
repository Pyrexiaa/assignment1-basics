from cs336_basics.jy import train_bpe
import json
from tests.common import gpt2_bytes_to_unicode

FILE = "tests/fixtures/corpus.en"
VOCAB_SIZE = 500
SPECIAL_TOKENS = [
    "<|endoftext|>"
]
vocab, merges = train_bpe(FILE, VOCAB_SIZE, SPECIAL_TOKENS)

# Path to the reference tokenizer vocab and merges
reference_vocab_path = "tests/fixtures/train-bpe-reference-vocab.json"
reference_merges_path = "tests/fixtures/train-bpe-reference-merges.txt"

# Compare the learned merges to the expected output merges
gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
with open(reference_merges_path, encoding="utf-8") as f:
    gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
    reference_merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_reference_merges
    ]

# Debug: check merges difference with index-by-index comparison
if merges != reference_merges:
    print("⚠️ Merges differ!")

    min_len = min(len(merges), len(reference_merges))
    for i in range(min_len):
        if merges[i] != reference_merges[i]:
            print(f"  At index {i}: learned={merges[i]} | reference={reference_merges[i]}")

    if len(merges) > len(reference_merges):
        print("  Learned has extra merges at the end:", merges[len(reference_merges):])
    elif len(reference_merges) > len(merges):
        print("  Reference has extra merges at the end:", reference_merges[len(merges):])
else:
    print("✅ Merges match.")

# Compare the vocab to the expected output vocab
with open(reference_vocab_path, encoding="utf-8") as f:
    gpt2_reference_vocab = json.load(f)
    reference_vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
    }

# Debug: check vocab difference
if set(vocab.keys()) != set(reference_vocab.keys()):
    print("⚠️ Vocab keys differ!")
    print("  Extra in learned vocab keys:", set(vocab.keys()) - set(reference_vocab.keys()))
    print("  Missing in learned vocab keys:", set(reference_vocab.keys()) - set(vocab.keys()))
if set(vocab.values()) != set(reference_vocab.values()):
    print("⚠️ Vocab values differ!")
    print("  Extra in learned vocab values:", set(vocab.values()) - set(reference_vocab.values()))
    print("  Missing in learned vocab values:", set(reference_vocab.values()) - set(vocab.values()))
else:
    print("✅ Vocab match.")