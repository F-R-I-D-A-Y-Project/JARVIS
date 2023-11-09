import pathlib
import re
import csv
import numpy as np
from pprint import pprint

class ResponseBuilder:
    def __init__(self, path_to_dataset: str) -> None:
        self.dataset = path_to_dataset
        self.data_base_texts = []
        self.dictionary = {}

    def get(self, sentence: str) -> str: 
        if(self.data_base_texts==[]):
            self.populate_dictionary()
        featureVectors = np.zeros((len(self.data_base_texts),len(self.dictionary)))
        for i in range(len(self.data_base_texts)):
            for word in self.data_base_texts[i].split(" ") :
                featureVectors[i][self.dictionary[self.preprocess_text(word)]]+=1 
        inputFeatureVector = np.zeros(len(self.dictionary))

        for word in sentence.split(" "):
            preprocess_word = self.preprocess_text(word)
            if preprocess_word in self.dictionary:
                inputFeatureVector[self.dictionary[preprocess_word]]+=1
        
        max_similarity = -1.0
        most_similar_index = -1

        for i in range(len(featureVectors)):
            similarity = self.calculate_cosine_similarity(inputFeatureVector, featureVectors[i])
            if (similarity > max_similarity):
                max_similarity = similarity
                most_similar_index = i
        with open(self.dataset) as file:
            csv_reader = csv.reader(file,delimiter=';')
            for row in csv_reader:
                if row[0]==self.data_base_texts[most_similar_index]:
                    return row[1]
        return ""
        

    def populate_dictionary(self):
        num=0
        with open(self.dataset) as file:
            csv_reader = csv.reader(file, delimiter=';')
            for row in csv_reader:
                self.data_base_texts.append(row[0])
                for s in row[0].split(" "):
                    if self.preprocess_text(s) not in self.dictionary.keys():
                        self.dictionary[self.preprocess_text(s)]=num
                        num+=1


    def preprocess_text(self, text: str) -> str:
        return re.sub(r'[^\w\s;]', '', text).lower()


    def calculate_cosine_similarity(self, vectorA, vectorB) -> float:
        if np.dot(vectorA,vectorA)*np.dot(vectorB,vectorB) != 0:
            return np.dot(vectorA,vectorB)/(np.dot(vectorA,vectorA)*np.dot(vectorB,vectorB))**0.5
        else:
            return np.inf


class Model:
    def __init__(self, path_to_dataset: str=(pathlib.Path(__file__).parent/'dataBase.csv').absolute()) -> None:
        self.manager = ResponseBuilder(path_to_dataset)
        pass

    def answer(self, prompt: str) -> str:
        sentences = re.split(r'[.?!] ', prompt)
        print(sentences)
        final_answer: str = ""
        for sentence in sentences:
            response = self.manager.get(sentence)
            if response != '':
                if response.endswith('.') or response.endswith('?') or response.endswith('!') or response.endswith(' '):
                    final_answer += response
                else:
                    final_answer += response + '. '
                final_answer += '\n'
        return final_answer

    __call__ = answer