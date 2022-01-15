from datasets import load_dataset
import langdetect

dataset = load_dataset("oscar", "unshuffled_deduplicated_tr")
#dataset = load_dataset("clinc_oos", "small")

def detect(x):
	try:
		return {"lang": langdetect.detect(x["text"])}
	except:
		return {"lang": "unk"}

dataset = dataset.map(detect)

dataset.save_to_disk("lang_detected")
