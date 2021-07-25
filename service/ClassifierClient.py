#!/usr/local/bin/python
import rpyc
import argparse
if __name__ == "__main__":
    # Initialize a Connection to the Model
    conn = rpyc.connect("localhost", 12345)
    conn._config['sync_request_timeout'] = None
    c = conn.root
    parser = argparse.ArgumentParser()
    #Required Args to train a model
    parser.add_argument("-p", "--predict", type=str) 
    parser.add_argument("-c", "--classes", nargs='+')
    parser.add_argument("-t", "--train", type=str)
    parser.add_argument("-v", "--validloc", type=str)
    parser.add_argument('--start', dest='start', action='store_true')
    args = parser.parse_args()
    # do stuff over rpyc
    if args.start:
        if c.buildModel(args.train, args.validloc,args.classes):
            print("Model Initialized Successfully")
        else:
            print("Error Initializing Model")
    else:
        if args.predict != None:
            prediction = c.predict(args.classes, args.predict)
            print(args.classes[prediction[0]])
        if args.train != None:
            status = c.train(args.classes, args.train, args.validloc)
            if status:
                print("Model Successfully Trained")
            else:
                print("Error Occured in Training Model, Process did not finish")
