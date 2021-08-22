from subprocess import Popen, PIPE, STDOUT
import subprocess
import time

if __name__ == '__main__':
    process = Popen(["python3", "requestor.py", "-d", "dataset", "-c", "dog", "monkey", "cat", "cow"], stdin=subprocess.PIPE) # Start Requestor Service
    """
    Tasks are shown already made like this, but if you are running a service it may make more sense to make each 
    task a list like ["predict", "test1.jpg"] and concatenate them. This service also uses stdin in write, and
    while at first communicate may seem more proper, it awaits the task to end which it never does
    """
    tasks = ['predict test1.jpg','train train.tar.gz valid.tar.gz']
    task1 = process.stdin.write((tasks[0] + "\n").encode("utf-8"))
    task2 = process.stdin.write((tasks[1] + "\n").encode("utf-8"))
    # Having \n on the final task, or most recent one without anothet ask coming up causes an EOF error, rather than the provider just
    # continuing on normally
    task3 = process.stdin.write((tasks[0]).encode("utf-8"))
    # Then run you're handling of output here, tasks are queued up in stdin, in this case just wait 15 minutes
    # So we can see the output
    time.sleep(900)
    # Then kill subprocess when done
    process.kill()
