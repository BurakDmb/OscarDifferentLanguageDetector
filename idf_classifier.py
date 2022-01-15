import numpy as np
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix

dataset = load_from_disk("lang_detected")["train"].train_test_split(0.2)

vec = TfidfVectorizer(use_idf=True)

train = vec.fit_transform(dataset["train"]["text"])
test = vec.transform(dataset["test"]["text"])

prediction = np.array(test.mean(1) > train.mean(), dtype=np.int32)
#prediction = model.predict(test)
g_t = np.array([0 if i == "tr" else 1 for i in dataset["test"]["lang"]]).reshape(-1,1)

print(confusion_matrix(g_t, prediction))
print(classification_report(g_t, prediction))

