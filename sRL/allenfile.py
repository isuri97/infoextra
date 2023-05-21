import re
import nltk
import pandas as pd
from allennlp.predictors.predictor import Predictor

download = "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
# predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz")
predictor = Predictor.from_path(download)

df = pd.read_csv('data/wiener-100.csv')
df['coref'] = df['coref'].str.replace(r'\[|\]', '')
df['coref']
count = 0

for index, paragraph in zip(df['id'].to_list(),df['coref'].to_list()):
    # paragraph = row['coref']
    print(f'index {index}| paragraph : {paragraph}')
    try:
        sentences = nltk.sent_tokenize(paragraph)
        # print(sentences)
        filename = f"sRL/sentence_files/{index}.txt"
        with open(filename, 'w') as f:
            for sentence in sentences:
                parts = re.split(r"\s+and\s+", sentence)
                if parts is not None:
                    for i in parts:
                        f.write(i + '\n')
                else:
                    f.write(sentence + '\n')
        count = count + 1
    except TypeError:
        pass

from pathlib import Path
import os, glob

folder = Path('sRL/sentence_files')

# get all the files in the folder
files = folder.glob('**/*.txt')  # assuming the files are csv

# dir = 'sRL/sentence_files/'
# file_lst = os.listdir(dir)


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

        fn = "sRL/files_dir/" + count + ".txt"

        with open(fn, 'w') as ff:
            # print(f'sentence list = {sentences_list}')
            ff.writelines(sentences_list)
            print('sentences written')

        count = count + 1
