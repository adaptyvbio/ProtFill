import yaml

from protfill.scripts.run import make_parser, parse, run


def main():
    argparser = make_parser()
    argparser.add_argument(
        "--config", type=str, help="Path to config file", required=True
    )
    args = parse(argparser=argparser)
    config = args.config
    with open(config, "r") as stream:
        config = yaml.safe_load(stream)
    for k, v in config.items():
        setattr(args, k, v)
    run(args)
