import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

from shell.process import Process
from hmi.interface import HMI
from model.model import Model


def main():
    model = Model('datasets/example.txt')
    with Process() as proc:
        hmi = HMI(model, proc)
        try:
            hmi.run()
        except KeyboardInterrupt:
            print("Exiting F.R.I.D.A.Y. process")


if __name__ == "__main__":
    main()
