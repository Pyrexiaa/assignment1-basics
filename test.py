import pickle
from cs336_basics.tokenization.bpe import train_bpe

FILE = "tests/fixtures/tinystories_sample_5M.txt"
VOCAB_SIZE = 1000
SPECIAL_TOKENS = [
    "<|endoftext|>"
]
vocab, merges = train_bpe(FILE, VOCAB_SIZE, SPECIAL_TOKENS)

for run in range(1):
    print(f"\n=== RUN {run+1} ===")
    vocab, merges = train_bpe(FILE, VOCAB_SIZE, SPECIAL_TOKENS)
    print(f"First 5 merges: {merges[:5]}")
    print(f"Merge at index 401: {merges[401] if len(merges) > 401 else 'N/A'}")

# [print(k, ":", repr(v)) for k,v in vocab.items()]
# [print(f"{i:>5}  {m}") for i,m in enumerate(merges)]


snapshot_path = "tests/_snapshots/test_train_bpe_special_tokens.pkl"
with open(snapshot_path, "rb") as f:
    snapshot = pickle.load(f)

# Compare the lengths and first few differences
print(f"Current merges length: {len(merges)}")
print(f"Snapshot merges length: {len(snapshot['merges'])}")

# Find the first difference
for i, (current, expected) in enumerate(zip(merges, snapshot['merges'])):
    if current != expected:
        print(f"First difference at index {i}:")
        print(f"  Current:  {current}")
        print(f"  Expected: {expected}")
        break

# Show some context around the difference
start = max(0, i-5)
end = min(len(merges), i+6)
print(f"\nContext around difference (index {start}-{end}):")
for j in range(start, end):
    marker = " -> " if j == i else "    "
    current_merge = merges[j] if j < len(merges) else "N/A"
    expected_merge = snapshot['merges'][j] if j < len(snapshot['merges']) else "N/A"
    print(f"{marker}{j:3d}: {current_merge} | {expected_merge}")

print("\n\n\n================ (wrong_additional)")
wrong_additional = [i for i in merges if i not in snapshot["merges"]]
[print(i) for i in wrong_additional]

print("\n\n\n================ (not_found)")
not_found = [i for i in snapshot["merges"] if i not in merges]
[print(i) for i in not_found]

print(train_bpe.__module__)