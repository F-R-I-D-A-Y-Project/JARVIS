import argparse, subprocess, pathlib

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('args', nargs='+')
    return parser.parse_args()

def main():
    args = parse_args()
    if args.args[0] == 'run':
        path = pathlib.Path('.') / 'source' / 'main' / '__main__.py'
        subprocess.run(f'python {path.absolute()}', shell=True)

if __name__ == '__main__':
    main()
