import pandas as pd

import spacy

import crosslingual_coreference

df = pd.read_csv('data/new_wiener.csv',',')
content = df['content']
print(content)

# text = (
#     "Do not forget about Momofuku Ando! He created instant noodles in Osaka. At"
#     " that location, Nissin was founded. Many students survived by eating these"
#     " noodles, but they don't even know him.Noodels were very tasty and it is very popular ammong the Japan"
# )
#
#
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe(
    "xx_coref", config={"chunk_size": 2500, "chunk_overlap": 2, "device": -1}
)

doc = nlp(content[0])
# print(doc._.coref_clusters)
# print(doc)

resolved_content = doc._.resolved_text
with open("sample.txt", "w") as f:
    f.write(resolved_content)
