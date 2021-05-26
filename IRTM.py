#Library
import en_core_web_sm
import pandas as pd
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import itertools
from scipy.sparse import csr_matrix
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re
import io
!pip install fasttext
import fasttext
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
pd.set_option('display.max_rows', None)

#load data
file = open('Tom-Clancy-Jack-Ryan-09-Rainbow-Six.txt', 'r', encoding='utf-8', errors='ignore')
with file as f:
    filedata = f.read()

#Preprocessing
sentences = [re.sub('[^a-zA-Z]', ' ', sentence) for sentence in nltk.sent_tokenize(filedata) if len(sentence)>1]

#Tokenlization and remove stop words
word_in_sentences = [sentence.split(' ') for sentence in sentences]
stop_words = stopwords.words('english')
def remove_stop_words(words):
  return [word for word in words if word and word not in stop_words]
word_in_sentences = list(map(remove_stop_words, word_in_sentences))


#Build the dataset
dataset = pd.DataFrame({'Sentence':sentences, 'Word':word_in_sentences})

# NER
nlp = en_core_web_sm.load()
nlp.max_length = len(sentences)

full_name = {}
part_name = {}
character_in_sentence = []

for i in range(len(sentences)):
  doc = nlp(sentences[i])
  name_entity = [str(x) for x in doc.ents if x.label_ == 'PERSON']
  character_in_sentence.append(name_entity)
  # Entity Normalization
  if name_entity != []:
    for name in name_entity:
      if name in full_name:
        full_name[name] += 1
      elif len(name.split(' ')) > 1:
        full_name[name] = 1
        for part in name.split(' '):
          if part not in part_name:
            part_name[part] = name
      elif name not in part_name:
        full_name[name] = 1
      else:
        full_name[part_name[name]] += 1
        dataset.iloc[i].Sentence = re.sub(name, part_name[name], sentences[i])
      
dataset['Character'] = character_in_sentence
character_df = pd.DataFrame(full_name.items(), columns=['name', 'frequency'])
character_df = character_df.loc[character_df['frequency']>30]
character_df = character_df.loc[~character_df['name'].isin(['Rainbow', 'Dmitriy Arkadeyevich','Rainbow Six', 'Domingo Chavez'])].reset_index(0)
label = [0, 2, 2, 2, 4, 4, 4, 4, 1, 0, 2, 2, 2, 4, 4, 3, 4, 4, 0, 3, 3, 3, 4, 4, 4, 0, 4, 4, 4, 4, 4]
character_df['label'] = label

# store the df
character_df.to_csv('character.csv', index=False)
dataset.to_csv('dataset.csv', index=False)

characters = character_df['name'].values
word_to_id = dict(zip(characters, range(len(characters))))
id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))
sentences_as_ids = [np.sort([word_to_id[w] for w in word_to_id if w in sentence]).astype('uint32') for sentence in dataset.Sentence]
#creating a pair of indices for each character found in the sentence
#row_ind = sentence index. col_ind = character_index
row_ind, col_ind = zip(*itertools.chain(*[[(i, w) for w in sentence] for i, sentence in enumerate(sentences_as_ids)]))


train_set, test_set = train_test_split(list(id_to_word.keys()), test_size=0.2, random_state=50)
text = []
for i in range(len(row_ind)):
  if col_ind[i] in train_set:
    temp = dataset.Sentence[row_ind[i]]
    sentence = '__label__' + str(character_df.loc[character_df['name']==id_to_word[col_ind[i]], 'label'].values)+ ' ' + temp
    text.append(sentence)
with io.open('train.txt', 'w', encoding='utf-8') as f:
  for senten in text:
    f.write(senten+'\n')
    
    
def create_co_occurence_matrix(characters, sentences):
    #indexing characters
    word_to_id = dict(zip(characters, range(len(characters))))
    #returns list of lists: ids of characters found in each sentence
    sentences_as_ids = [np.sort([word_to_id[w] for w in word_to_id if w in sentence]).astype('uint32') for sentence in sentences]
    #creating a pair of indices for each character found in the sentence
    #row_ind = sentence index. col_ind = character_index
    row_ind, col_ind = zip(*itertools.chain(*[[(i, w) for w in sentence] for i, sentence in enumerate(sentences_as_ids)]))
    #data is binary: did the character appear in the sentence or not?
    #since this must be true for every (row_ind, col_ind), set value to 1
    data = np.ones(len(row_ind), dtype='uint32')
    #determining dimension of square matrix (equal to len(characters) if all are found in sentences)
    max_word_id = max(itertools.chain(*sentences_as_ids)) + 1
    #using CSR matrix for efficient matrix storage and multiplication.
    #Each cell in a row = 1 if the tuple coordinates are in zip(row_ind, col_ind)
    #else, cell value = 0
    sentence_words_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(len(sentences_as_ids), max_word_id))
    #matrix algebra: multiply transpose of sentence words matrix by itself: gives dimension words by words
    words_cooc_matrix = sentence_words_matrix.T * sentence_words_matrix
    #we don't care about a character co-occurring with itself (since it does in every relevant sentence)
    words_cooc_matrix.setdiag(0)
    return words_cooc_matrix, word_to_id

#Outputs a plot of the top n characters ranked in descending order of importance based on each centrality metric
def character_importance(key, centrality_dict, word_mapping, n):
  char_importance = pd.DataFrame.from_dict(centrality_dict[key], orient='index')
  char_importance['name'] = word_mapping.keys()
  char_importance = char_importance.sort_values(by=0, ascending=False).iloc[:n,:].sort_values(by=0, ascending=True)
  fig = plt.figure(figsize=(20,10))
  plt.barh(char_importance['name'], char_importance[0], align='center')
  plt.title('Character {}'.format(key))
  plt.show()

def visualize(co_mat, word_mapping, n):
  #Abstract art: networks visualization of the relationships between characters.
  plt.figure(figsize=(20,20))
  G = nx.from_numpy_matrix(co_mat.toarray(), create_using=nx.DiGraph)
  layout = nx.spring_layout(G)
  normalized_frequency = np.array(character_df['frequency']) / np.max(character_df['frequency'])
  colors = character_df['label'].values
  id_to_word = dict(zip(character_df.index, character_df['name']))
  nx.draw(G, layout, node_color=colors, node_size=np.sqrt(normalized_frequency) * 6000, labels = id_to_word, font_size=15)
  plt.show()

  #Calculating the 4 types of centrality
  bc = nx.betweenness_centrality(G, normalized=False)
  dc = nx.degree_centrality(G)
  cc = nx.closeness_centrality(G)
  ec = nx.eigenvector_centrality(G)
  #Running through each of the types
  centrality_dict = {'Betweenness Centrality':bc, 'Degree Centrality':dc, 'Closeness Centrality':cc, 'Eigenvector Centrality':ec}
  for key in centrality_dict.keys():
    character_importance(key, centrality_dict, word_mapping, n)

    
#Visuliztion    
co_mat, word_mapping = create_co_occurence_matrix(character_df['name'].values, dataset['Sentence'])
visualize(co_mat, word_mapping, 20)
