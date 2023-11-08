import pathlib
from openai import OpenAI

class GPT:
    def __init__(self, key_path: pathlib.Path, model: str="gpt-3.5-turbo") -> None:
        with open(key_path, 'r') as f:
            openai.api_key =  f.read().strip()
        self.__model = model
        self.client = OpenAI()

    def __call__(self, prompt: str) -> str:
        message = [{"role": "user", "content": prompt}]
        return self.client.chat.completions.create(
            model=self.__model,
            messages=message
        ).choices[0].message


print(GPT(pathlib.Path(__file__).parent / 'key.txt')("what's your name"))
