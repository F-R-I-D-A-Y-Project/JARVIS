import pathlib
import re

class ResponseBuilder:
    def __init__(self, path_to_dataset: str) -> None:
        self.dataset = path_to_dataset

    def get(self, sentence: str) -> str: 
        pass

    def __populate_dictionary(self): ...

    def ___preprocess_text(self, text: str) -> list[str]: ...

    def __get_word_index(self, word: str, string_set: set[str]) -> int: ...

    def __calculate_cosine_similarity(self, vectorA: list[float], vectorB: list[float]) -> float: ...


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