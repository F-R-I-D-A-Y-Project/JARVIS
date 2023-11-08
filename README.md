# F.R.I.D.A.Y


Python project developed for the discipline "Laboratório de Programação 2", at "Instituto Militar de Engenharia"

## About 

F.R.I.D.A.Y is a chatbot and personal assistant AI developed in Python. Capable of answering question made by user and executing shell commands for the user. It was developed using the Transformers architecture, and 
it is serialized so the model doesn't need to be trained every time

## Team members

- Fabricio Asfora Romero Assunção
- Roberto Suriel de Melo Rebouças
- Johannes Elias Joseph Salomão

## Compatibility

F.R.I.D.A.Y is compatible with Python 3.10+

### Packages used

### Shell interaction
- [pexpect](https://pexpect.readthedocs.io/en/stable/)
- [platform](https://docs.python.org/3/library/platform.html)
### Model creation
- [pytorch](https://pytorch.org/docs/stable/index.html)
- [torchtext](https://pytorch.org/text/stable/index.html)
- [tiktoken](https://github.com/openai/tiktoken)
### Documentation
- [sphinx](https://www.sphinx-doc.org/en/master/)
### GUI
- [tkinter](https://docs.python.org/3/library/tkinter.html)
### STL
- [subprocess](https://docs.python.org/3/library/subprocess.html)
- [sys](https://docs.python.org/3/library/sys.html)
- [time](https://docs.python.org/3/library/time.html)
- [pathlib](https://docs.python.org/3/library/pathlib.html)
- [csv](https://docs.python.org/3/library/csv.html)
- [pickle](https://docs.python.org/3/library/pickle.html)


## How to use
To create and activate the Python virtual environment, use:

- On Linux:

``` sh
source init_venv.sh
activate_venv
```

- On Windows:
``` ps1
. .\init_venv.ps1
InitVenv
```

Then, all you have to do is run :

``` sh
friday run
```

on your terminal after executing ``` source init_venv.sh``` (or ``` . .\init_venv.ps1```)

## Documentation

The documentation can be found [here](https://github.com/F-R-I-D-A-Y-Project/F.R.I.D.A.Y-Python/docs)
