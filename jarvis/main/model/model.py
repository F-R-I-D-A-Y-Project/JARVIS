import pathlib
import re
import csv
import numpy as np

class ResponseBuilder:
    def __init__(self, path_to_dataset: str) -> None:
        self.dataset = path_to_dataset
        self.data_base_texts = []
        self.dictionary = {}

    def get(self, sentence: str) -> str: 
        if(data_base_texts==[]):
            populate_dictionary()
        featureVectors = np.zeros((len(data_base_texts),len(dictionary)))

        for i in range(len(data_base_texts)):
            for word in data_base_texts[i].split(" ") :
                featureVectors[i][dictionary[pre_process_text(word)]]+=1
        inputFeatureVector = np.zeros(len(dictionary))

        for word in sentence.split(" "):
            preprocess_word = preprocess_text(word)
            if preprocess_word in dictionary:
                inputFeatureVector[dictionary[preprocess_word]]+=1
        
        max_similarity = -1.0;
        most_similar_index = -1;

        for i in range(len(featureVectors)):
            similarity = calculate_cosine_similarity(inputFeatureVector, featureVectors[i]);
            if (similarity > max_similarity):
                max_similarity = similarity
                most_similar_index = i
        with open('exemplo.csv', mode='r') as file:
            csv_reader = csv.reader(file,delimiter=';')
            for row in csv_reader:
                if row[0]==data_base_texts[most_similar_index]:
                    return row[1]
        return ""
        

    def populate_dictionary(self):
        data_base_texts = []
        num=0
        dictionary = {}
        with open('exemplo.csv', mode='r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                for str in row:
                    final_txt+=str
                    dictionary[preprocess_text(str)]=num;
                    num+=1;
                data_base_texts.append(final_txt.split(";")[0])
        self.data_base_texts = data_base_texts
        self.dictionary = dictionary


    def preprocess_text(self, text: str) -> str:
        return re.sub(r'[^\w\s;]', '', text).lower()

    #morreu com o dictionary
    def __get_word_index(self, word: str, string_set: set[str]) -> int: ...

    def __calculate_cosine_similarity(self, vectorA, vectorB) -> float:
        return np.dot(vectorA,vectorB)/(np.dot(vectorA,vectorA)*np.dot(vectorB,vectorB))**0.5



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

    __call__ = answer