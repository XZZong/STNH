from stnh import STNH
from utils import parameter_parser, args_printer
import yaml


def main():        
    args = parameter_parser()
    args_printer(args)

    stnh = STNH(args)
    stnh.fit()

if __name__ == '__main__':
    main()
