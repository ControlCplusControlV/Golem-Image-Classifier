import os
import requests
from subprocess import Popen, PIPE, STDOUT
# Image path
# Validation path
# Train path

tasks = [
    ["predict"],
    ["train", ""]
    ]

if __name__ == '__main__':
    p = Popen(["python3", "requestor.py", "-d", "dataset", "-c", "dog", "monkey", "cat", "cow"]) # Start Requestor Server
    stdout_data = p.communicate(input='["predict", "test1.jpg"]')[0]
    stdout_data1 = p.communicate(input='["train", "test1.jpg"]')[0]

    # Wait for service to startup
    """
    parser.add_argument('task', required=True)  # "predict" or "train"
    parser.add_argument('validpath', required=False) # For Train
    parser.add_argument('trainpath', required=False) # For Train
    parser.add_argument('imagename', required=False)
    """
