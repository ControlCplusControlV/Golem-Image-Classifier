#!/usr/local/bin/python
import argparse
import rpyc
from rpyc.utils.factory import unix_connect
if __name__ == "__main__":
    # Initialize a Connection to the Model
    conn = rpyc.utils.factory.unix_connect('/golem/run/uds_socket')
    conn._config['sync_request_timeout'] = None
    c = conn.root
    parser = argparse.ArgumentParser()
    # Required Args to train a model
    parser.add_argument("-p", "--predict", type=str)
    parser.add_argument("-c", "--classes", nargs='+')
    parser.add_argument("-t", "--train", type=str)
    parser.add_argument("-v", "--validloc", type=str)
    parser.add_argument('--start', dest='start', action='store_true')
    args = parser.parse_args()
    # do stuff over rpyc
    if args.start:
        if c.buildModel(args.train, args.validloc, args.classes):
            print("Model Initialized Successfully")
        else:
            print("Error Initializing Model")
    else:
        if args.predict is not None:
            prediction = c.predict(args.predict)
            print(args.classes[prediction[0]])
        if args.train is not None:
            status = c.train(args.classes, args.train, args.validloc)
            if status:
                print("Model Successfully Trained")
            else:
                print("Error Occured in Training Model, Process did not finish")
