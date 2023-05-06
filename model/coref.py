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
    "xx_coref", config={"chunk_size": 50, "chunk_overlap": 2, "device": -1}
)


# result = []
# for i in content:
#     doc = nlp(i)
#     resolved_content = doc._.resolved_text
#     result.append(resolved_content)
#
# df['resolved_coref'] = result

# doc = nlp(i)
# resolved_content = doc._.resolved_text
# resolved_content = doc._.resolved_text
# with open("sample.txt", "w") as f:
#     f.write(resolved_content)


result = []

for i, row in df.iterrows():
    try:
        doc = nlp(row['content'])
        resolved_content = doc._.resolved_text
        result.append(resolved_content)

        print(f"Finished document {row['id']}: {e}")
    except Exception as e:
        print(f"Error processing document {row['id']}: {e}")
        continue

df['resolved_coref'] = result
df.to_csv('coref.csv', sep='\t')