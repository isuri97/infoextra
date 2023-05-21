import re
import nltk
import pandas as pd
from allennlp.predictors.predictor import Predictor

df = pd.read_csv('data/wiener-100.csv')
df['coref'] = df['coref'].str.replace(r'\[|\]', '')
df['coref']

for index, row in df.iterrows():
  paragraph = row['coref']
  try:
    sentences = nltk.sent_tokenize(paragraph)
    print(sentences)
    filename = f"sentence_files/{index}.txt"
    with open(filename, 'w') as f:
      for sentence in sentences:
          parts = re.split(r"\s+and\s+", sentence)
          if parts is not None:
            for i in parts:
              f.write(i + '\n')
          else:
            f.write(sentence + '\n')
  except TypeError:
    pass


from pathlib import Path
import os,glob

folder = Path('sRL/sentence_files')

# get all the files in the folder
files = folder.glob('**/*.txt') # assuming the files are csv

count = 0

for filename in glob.glob(os.path.join(folder, '*.txt')):
    print(filename)
    with open(filename, 'r') as f:
        text = f.read()

        sentences = nltk.sent_tokenize(text)

        sentences_list = []
        more_verb = []

        try:
            for i in sentences:
                tree = predictor.predict(sentence=i)
                for verb in tree['verbs']:
                    srl_text = verb['description']
                    more_verb.append(srl_text + '\n')
            sentences_list.append(more_verb)
            print('kkkkkk')
        except IndexError:
            pass

        fn = f"arg-file/{count}.txt"
        with open(fn, 'w') as f:
            for i in sentences_list:
                for j in i:
                    f.write(j)

        count = count + 1