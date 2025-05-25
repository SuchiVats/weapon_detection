from .default_args import get_args_parser, populate_args

# rfdetr/util/args.py

def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser()
    # Add all your args like:
    parser.add_argument('--num_classes', type=int, default=1)
    # ...
    return parser

def populate_args(**kwargs):
    import argparse
    return argparse.Namespace(**kwargs)
