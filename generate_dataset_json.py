# # Saving the oscar dataset(28GB) to json format. (only execute once)
from datasets import load_from_disk
dataset = load_from_disk("lang_detected")["train"]
# Set num_proc according to your cpu count, num_proc=20 means 20 thread will be executed paralelly.
dataset.to_json("dataset_json", num_proc=20)