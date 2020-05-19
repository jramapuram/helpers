import argparse
from visdom import Visdom


parser = argparse.ArgumentParser(description='Visdom Log Writer.')

parser.add_argument('--visdom-url', type=str, required=True,
                    help='visdom URL for graphs, needs http://url')
parser.add_argument('--visdom-port', type=int, required=True,
                    help='visdom port for graphs')
parser.add_argument('--log-file', type=str, required=True,
                    help='the file to  (default: None)')
args = parser.parse_args()


if __name__ == "__main__":
    visdom = Visdom(server=args.visdom_url, port=args.visdom_port,
                    use_incoming_socket=False,
                    raise_exceptions=False)
    visdom.replay_log(args.log_file)
