import argparse, json

from util import train, test
from data_process import setup_seed

setup_seed(20)
LSTM_PATH = '../model.pkl'



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('--conf', default='conf.json',dest='conf')
    args = parser.parse_args()
	
    with open(args.conf, 'r') as f:
        conf = json.load(f)	

    flag = 'ms'

    train(conf, LSTM_PATH, flag)
    test(conf, LSTM_PATH, flag)


