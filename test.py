import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)

    return parser.parse_args()

def main(args):
    pass


if __name__ == '__main__':
    main(parse_args())