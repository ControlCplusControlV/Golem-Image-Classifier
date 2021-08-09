from subprocess import Popen, PIPE, STDOUT
import subprocess
import time

if __name__ == '__main__':
    p = Popen(["python3", "requestor.py", "-d", "dataset", "-c", "dog", "monkey", "cat", "cow"], stdin=subprocess.PIPE) # Start Requestor Server
    tasks = ['predict test1.jpg','train train.tar.gz valid.tar.gz']
    task1 = p.stdin.write((tasks[0] + "\n").encode("utf-8"))
    task2 = p.stdin.write((tasks[1] + "\n").encode("utf-8"))
    # Having \n on the final task, or most recent one without anothet ask coming up causes an EOF error, rather than the provider just
    # continuing on normally
    task3 = p.stdin.write((tasks[0]).encode("utf-8"))
