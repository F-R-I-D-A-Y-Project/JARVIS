"""
    This code is based on Andrej Karpathy's video and orca from snap-stanford. Links:

    - https://www.youtube.com/watch?v=kCc8FmEb1nY
    - https://github.com/snap-stanford/orca 
"""

import torch.nn as nn
import torch
from torch.optim import AdamW, Optimizer
import torch.nn.functional as F
import warnings
import pickle
import pathlib
import subprocess
import sys
import pathlib
from typing import Self

sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

from model.dbreader import DBManager

warnings.simplefilter("ignore")

class UntrainedModelError(Exception):
    pass


class Head(nn.Module):
    def __init__(self, head_size, n_embd, dropout, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei)
        v = self.value(x) 
        out = wei @ v
        return out
    
class MultiHeadAttention(nn.Module): 
    def __init__(self, n_heads, head_size, n_embd=384, drop=0.2) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.projection = nn.Linear(n_heads * head_size, n_embd)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.dropout(self.projection(x))
        return x

class FeedForward(nn.Module):
    def __init__(self, n_embd, intermediate_n_embd=4, dropout=0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, intermediate_n_embd * n_embd),
            nn.ReLU(),
            nn.Linear(intermediate_n_embd * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self: Self, n_embd, n_head) -> None:
        super().__init__()
        head_size = n_embd//n_head
        self.ln1 = nn.LayerNorm(n_embd)
        self.att = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)


    def forward(self, x):
        x+=self.att(self.ln1(x))
        x+=self.ffwd(self.ln2(x))
        return x
    
class Transformer(nn.Module): 
    def __init__(self,vocab_size: int,
                 n_embd: int,
                 block_size: int,
                 n_head: int,
                 n_layer: int,
                 device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx): 
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        return logits

    def generate(self, idx, max_new_tokens=300) -> torch.Tensor:
        ...


class Model:
    '''
        This class is responsible for the NLP model of the chatbot.
    '''
    def __init__(self: Self, path_to_dataset: str,
                 batch_size: int=64,
                 block_size: int=256,
                 per_epoch_iter: int=5000,
                 learning_rate: float=3e-4,
                 eval_iters: int=200,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 dropout: float=0.2,
                 n_layer: int=6,
                 n_head: int=6,
                 n_embd: int=384,
                 eval_interval: int=500,) -> None:
        '''
            Constructor of the class. It receives the path to the dataset, but does not train the model.
            
            Args:
                path_to_dataset (str): path to dataset used for training
        '''
        self.__device = device
        self.__batch_size = batch_size
        self.__block_size = block_size
        self.__per_epoch_iter = per_epoch_iter
        self.__learning_rate = learning_rate
        self.__eval_iters = eval_iters
        self.__dropout = dropout
        self.__n_layer = n_layer
        self.__n_head = n_head
        self.__n_embd = n_embd
        self.__eval_interval = eval_interval
        self.__dataset = DBManager(pathlib.Path(path_to_dataset))

    @property
    def dropout(self: Self) -> float:
        '''
            This property returns the dropout used for training the model.
        '''
        return self.__dropout
    
    @property
    def n_layer(self: Self) -> int:
        '''
            This property returns the number of layers used in the model.
        '''
        return self.__n_layer
    
    @property
    def n_head(self: Self) -> int:
        '''
            This property returns the number of heads in the multihead attention mechanism
        '''
        return self.__n_head
    
    @property
    def n_embd(self: Self) -> int:
        '''
            This property returns the embedding dimension.
        '''
        return self.__n_embd
    
    @property
    def eval_interval(self: Self) -> int:
        '''
            This property returns the evaluation interval used for training the model.
        '''
        return self.__eval_interval

    @property
    def dataset(self: Self) -> DBManager:
        '''
            This property returns the path to the dataset.
        '''
        return self.__dataset
    
    @property
    def device(self: Self) -> torch.device:
        '''
            This property returns the device used for the model.
        '''
        return self.__device
    
    @property
    def batch_size(self: Self) -> int:
        '''
            This property returns the batch size used for training the model.
        '''
        return self.__batch_size
    
    @batch_size.setter
    def batch_size(self: Self, value: int) -> None:
        '''
            This property sets the batch size used for training the model.
        '''
        self.__batch_size = value
    
    @property
    def block_size(self: Self) -> int:
        '''
            This property returns the block size used for training the model.
        '''
        return self.__block_size
    
    @block_size.setter
    def block_size(self: Self, value: int) -> None:
        '''
            This property sets the block size used for training the model.
        '''
        self.__block_size = value
    
    @property
    def per_epoch_iter(self: Self) -> int:
        '''
            This property returns the maximum number of iterations used for training the model.
        '''
        return self.__per_epoch_iter
    
    @per_epoch_iter.setter
    def per_epoch_iter(self: Self, value: int) -> None:
        '''
            This property sets the maximum number of iterations used for training the model.
        '''
        self.__per_epoch_iter = value
    
    @property
    def learning_rate(self: Self) -> float:
        '''
            This property returns the learning rate used for training the model.
        '''
        return self.__learning_rate
    
    @learning_rate.setter
    def learning_rate(self: Self, value: float) -> None:
        '''
            This property sets the learning rate used for training the model.
        '''
        self.__learning_rate = value
    
    @property
    def eval_iters(self: Self) -> int:
        '''
            This property returns the evaluation iterations used for training the model.
        '''
        return self.__eval_iters
    
    @property
    def model(self: Self):
        '''
            This property returns the model.
        '''
        return self.__model
    

    def fit(self: Self, train_test_split: float=0.8, *, 
            epochs: int=5000, 
            verbose: bool=True,
            optimizer: Optimizer|None=None) -> None:
        '''
            This method is responsible for training the model.
            It reads the dataset, captures the amount of unique words existent in the dataset, 
            creates the Transformers Model and trains it

            If the dataset was unchanged, it loads the model from the pickle file. If not, the training algorithm will be executed.

            Args:
                path_to_dataset (str): path to dataset used for training
        '''
        if (pathlib.Path(__file__).parent.parent.parent.parent / 'model.pkl').exists():
            with open('model.pkl', 'rb') as f:
                self.__model = pickle.load(f)
        else:
            if optimizer == None:
                optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
            self.__train(train_test_split, epochs, verbose, optimizer)

    def __serialize_model(self: Self) -> None:
        '''
            Serializes the model into a pickle file to avoid retraining it every time the chatbot is executed.
        '''

        with (pathlib.Path(__file__).parent.parent.parent.parent / 'model.pkl').open('wb') as f:
            pickle.dump(self.__model, f)

    def __train(self: Self, train_test_split: float, epochs: int, verbose: bool, optimizer: Optimizer) -> None:
        '''
            Training algorithm of the Transformers model
        '''

        self.__model = Transformer(block_size=self.block_size,
                                   n_embd=self.n_embd,
                                   n_head=self.n_head,
                                   n_layer=self.n_layer,
                                   vocab_size=400, #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                   ).to(self.device)

        while True:
            X_train, Y_train, X_test, Y_test = self.dataset.split(train_test_split)

            for epoch in range(epochs):
                if verbose:
                    print(f'Epoch {epoch+1}/{epochs}')
                self.model.train()
                self.__train_epoch(optimizer, X_train, Y_train, verbose)
                self.model.eval()
            
            if True: #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                break             

        if verbose: 
            print('\n','==================================',*self.model.parameters(),'==================================','\n'*2, sep='\n')
            print('Model results: \n')
            # print()
        
        self.__serialize_model()

    def __train_epoch(self: Self, optimizer: Optimizer, X_train, Y_train, verbose: bool) -> None:
        '''
            Training epoch of the Transformers model
        '''
        logits = self.model(X_train)
        loss = self.eval_loss(logits, Y_train)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if verbose:
            print("loss for this epoch: %.4f" % loss.item())

    def eval_loss(self: Self, logits, Y_train) -> torch.Tensor: 
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        Y_train = Y_train.view(B*T)
        loss = F.cross_entropy(logits, Y_train)
        return loss

    def predict(self: Self, message: str) -> str:
        '''
            This method is responsible for returning the answer of the model for the chatbot.
            It receives a message, tokenizes it, and passes it to the Transformers Model

            Args:
                message (str): message to be answered by the chatbot

        '''
        if not hasattr(self, '_Model__model'):
            raise UntrainedModelError("Model not trained yet. Use 'fit()' method to train it.")
        return self.decode(self.model.generate(message))

    __call__ = predict
    
    def check_db_change(self: Self, commit_on_change: bool=True) -> None:
        '''
            Verifies changes in the dataset. If there are changes, it will delete the serialized model.

            OBS: The changes are verified via git, so in order to properly verify the difference, commits will be made
            every time 
        '''
        out = subprocess.run(f'git diff {self.__dataset}/../approved.csv', shell=True, cwd=pathlib.Path(__file__).parent.parent.parent.parent.absolute(),
                       capture_output=True).stdout.strip()
        if out:
            subprocess.run(f'rm -f model.pkl', shell=True, cwd=pathlib.Path(__file__).parent.parent.parent.parent.absolute())
            self.__update_dataset()
            if commit_on_change:
                subprocess.run([f'git add .', 'git commit -am "Update dataset" --amend'], shell=True, cwd=pathlib.Path(__file__).parent.parent.parent.parent.absolute())

    def __update_dataset(self: Self) -> None: pass