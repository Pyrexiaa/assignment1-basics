import pickle
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode

snapshot_path = "tests/_snapshots/test_train_bpe_special_tokens.pkl"

with open(snapshot_path, "rb") as f:
    snapshot = pickle.load(f)

# print(snapshot.keys())

# for id,values in zip(snapshot["vocab_keys"], snapshot["vocab_values"]):
#     print(f"{id:>5}  {values}")

[print(f"{i:>5}  {m}") for i,m in enumerate(snapshot["merges"])]
