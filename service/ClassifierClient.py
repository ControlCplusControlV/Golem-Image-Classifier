#!/usr/local/bin/python
import rpyc
import argparse
if __name__ == "__main__":
    # Initialize a Connection to the Model
    conn = rpyc.connect("localhost", 12345)
    c = conn.root
    parser = argparse.ArgumentParser()
    #Required Args to train a model
    parser.add_argument("-p", "--predict", type=str) 
    parser.add_argument("-c", "--classes", nargs='+')


    args = parser.parse_args()
    # do stuff over rpyc
    prediction = c.predict(args.classes, args.predict)
    print(args.classes[prediction[0]])