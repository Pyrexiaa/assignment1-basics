import os
import regex as re
from collections import Counter

# regex-based pre-tokenizer used by GPT-2
REGEX_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def init_vocab(special_tokens: list[bytes | str] | None = None) -> dict[int, bytes]:
    """Initialize vocabulary with consistent int->bytes mapping."""
    vocab = {}
    next_id = 0
    
    # Add special tokens first
    if special_tokens:
        for tok in special_tokens:
            if isinstance(tok, str):
                tok = tok.encode("utf-8")
            vocab[next_id] = tok
            next_id += 1
    
    # Add all byte values
    for i in range(256):
        vocab[next_id] = bytes([i])
        next_id += 1
    
    return vocab

def build_id2tok(vocab: dict) -> dict[int, bytes]:
    """Build reverse mapping (same as vocab since we use int->bytes consistently)."""
    return vocab.copy()

def decode_key(key, vocab, id2tok=None):
    """Decode a key (int ID or bytes) to string representation."""
    if id2tok is None:
        id2tok = vocab
    
    if isinstance(key, int):
        if key in id2tok:
            try:
                return id2tok[key].decode("utf-8")
            except UnicodeDecodeError:
                return f"<byte {id2tok[key].hex()}>"
        else:
            return f"<unk {key}>"
    
    elif isinstance(key, bytes):
        try:
            return key.decode("utf-8")
        except UnicodeDecodeError:
            return f"<byte {key.hex()}>"
    
    else:
        return str(key)

def display_vocab(vocab: dict[int, bytes]) -> dict[str, int]:
    """Display vocabulary in human-readable format."""
    printable = {}
    for token_id, token_bytes in vocab.items():
        printable[decode_key(token_bytes, vocab)] = token_id
    return printable

def decode_token(token_id: int, vocab: dict[int, bytes]) -> str:
    """Decode a single token ID to string (O(1) lookup)."""
    if token_id in vocab:
        try:
            return vocab[token_id].decode("utf-8")
        except UnicodeDecodeError:
            return vocab[token_id].decode("utf-8", errors="replace")
    return f"<unk {token_id}>"

def decode_tokens(token_ids: list[int], vocab: dict[int, bytes]) -> str:
    """Decode a list of token IDs to string."""
    return "".join(decode_token(token_id, vocab) for token_id in token_ids)

def pretokenize(texts: list[str]):
    pattern = re.compile(REGEX_PATTERN)
    for t in texts:
        for match in re.finditer(pattern, t):
            yield match.group(0)

def get_stats(token_counts: Counter[tuple[int, ...]]) -> Counter[tuple[int, int]]:
    """Count adjacent pairs efficiently using frequency data."""
    pairs = Counter()
    for token_seq, freq in token_counts.items():
        for i in range(len(token_seq) - 1):
            pairs[(token_seq[i], token_seq[i + 1])] += freq
    return pairs

def merge_pair(token_counts: Counter[tuple[int, ...]], pair: tuple[int, int], new_token_id: int) -> Counter[tuple[int, ...]]:
    """Merge all occurrences of the given pair in the token sequences."""
    first, second = pair
    new_token_counts = Counter()

    for token_seq, freq in token_counts.items():
        merged = []
        i = 0
        while i < len(token_seq):
            # i is used to track the indices and find the pairs to merge
            if i < len(token_seq) - 1 and token_seq[i] == first and token_seq[i + 1] == second:
                merged.append(new_token_id)
                i += 2
            else:
                merged.append(token_seq[i])
                i += 1
        new_token_counts[tuple(merged)] += freq
    
    return new_token_counts

def train_bpe(input_path: str, vocab_size: int = 500, special_tokens: list[str] | None = None):
    """Train BPE with optimized implementation matching the expected behavior."""
    assert os.path.exists(input_path), f"File not found: {input_path}"

    # Initialize vocabulary with consistent int->bytes mapping
    vocab = init_vocab(special_tokens)
    next_id = max(vocab.keys()) + 1
    
    # Create byte-to-id mapping for initial conversion (matching second implementation)
    byte_to_id = {v: k for k, v in vocab.items() if len(v) == 1}
    
    # Process file similar to the working implementation
    token_counts = Counter()
    
    # Read file as binary and decode with error handling (like chunking approach)
    with open(input_path, "rb") as f:
        content = f.read()
    
    # Decode with error handling (same as chunking approach)
    text = content.decode("utf-8", errors="ignore")
    
    # Remove special tokens from text as it is not used in BPE training as they are just separators
    if special_tokens:
        pattern = "|".join(map(re.escape, special_tokens))
        text = re.split(pattern, text)

    # Process text using pretokenization
    for tok in pretokenize(text):
        bs = tok.encode("utf-8")
        ids = tuple(byte_to_id[bytes([b])] for b in bs)  # Use bytes objects as keys
        token_counts[ids] += 1

    merges = []
    
    while len(vocab) < vocab_size:
        # Get pair statistics efficiently
        stats = get_stats(token_counts)
        if not stats:
            break

        # # Pick most frequent pair with deterministic tie-breaking
        # best_pair = max(stats, key=lambda p: (stats[p], vocab[p[0]] + vocab[p[1]]))

        # The above pair selection is not working because of the concatenation order
        # If concatenation: b'a' + b'bc' = b'abc' and b'ab' + b'c' = b'abc'
        # When compared lexicographically: b'abc' = b'abc', so first b'abc' would be chosen
        # However, if we compare the tuples directly, b'a' < b'ab', so pair with b'ab' would be chosen first
        # Hence, the second pair of b'ab' + b'c' would be chosen first
        # Insight: Concatenation loses the information of word boundaries, resulting in different original pairs
        best_pair = max(stats, key=lambda p: (stats[p], (vocab[p[0]], vocab[p[1]])))

        # Add new merged token to vocab
        merged_bytes = vocab[best_pair[0]] + vocab[best_pair[1]]
        vocab[next_id] = merged_bytes
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        # Merge pairs in token sequences
        token_counts = merge_pair(token_counts, best_pair, next_id)
        
        next_id += 1

    # Return the raw vocab (int -> bytes) and merges, not the display version
    print(f"Final vocab size: {len(vocab)}")
    
    return vocab, merges

def train_bpe_from_string(input_string: str, vocab_size: int = 500, special_tokens: list[str] | None = None):
    """Train BPE directly from string (for testing)."""
    # Initialize vocabulary with consistent int->bytes mapping
    # Initialize vocabulary with consistent int->bytes mapping
    vocab = init_vocab(special_tokens)
    next_id = max(vocab.keys()) + 1
    
    # Create byte-to-id mapping for initial conversion (matching second implementation)
    byte_to_id = {v: k for k, v in vocab.items() if len(v) == 1}
    
    # Process file similar to the working implementation
    token_counts = Counter()

    # Remove special tokens from text as it is not used in BPE training as they are just separators
    if special_tokens:
        pattern = "|".join(map(re.escape, special_tokens))
        text = re.split(pattern, input_string)

    # Process text using pretokenization
    for tok in pretokenize(text):
        bs = tok.encode("utf-8")
        ids = tuple(byte_to_id[bytes([b])] for b in bs)  # Use bytes objects as keys
        token_counts[ids] += 1

    print(f"Token Counts: {token_counts.most_common(5)}")

    merges = []

    while len(vocab) < vocab_size:
        # Get pair statistics efficiently
        stats = get_stats(token_counts)
        if not stats:
            break

        # # Pick most frequent pair with deterministic tie-breaking
        # best_pair = max(stats, key=lambda p: (stats[p], vocab[p[0]] + vocab[p[1]]))

        # The above pair selection is not working because of the concatenation order
        # If concatenation: b'a' + b'bc' = b'abc' and b'ab' + b'c' = b'abc'
        # When compared lexicographically: b'abc' = b'abc', so first b'abc' would be chosen
        # However, if we compare the tuples directly, b'a' < b'ab', so pair with b'ab' would be chosen first
        # Hence, the second pair of b'ab' + b'c' would be chosen first
        # Insight: Concatenation loses the information of word boundaries, resulting in different original pairs
        best_pair = max(stats, key=lambda p: (stats[p], (vocab[p[0]], vocab[p[1]])))

        # Add new merged token to vocab
        merged_bytes = vocab[best_pair[0]] + vocab[best_pair[1]]
        vocab[next_id] = merged_bytes
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        # Merge pairs in token sequences
        token_counts = merge_pair(token_counts, best_pair, next_id)
        
        next_id += 1

    # Return the raw vocab (int -> bytes) and merges, not the display version
    print(f"Final vocab size: {len(vocab)}")
    
    return vocab, merges

if __name__ == "__main__":
    test_string = (
        """low low low! low low lower? lower widest. widest widest newest newest newest newest newest newest"""
    )

    train_bpe_from_string(
        input_string=test_string,
        vocab_size=500,
        special_tokens=["<|endoftext|>"]
    )