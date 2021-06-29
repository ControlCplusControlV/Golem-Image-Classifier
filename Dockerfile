FROM ubuntu:21.04
RUN apt-get -y update && apt-get -y upgrade
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN apt-get install ffmpeg libsm6 libxext6  -y
ADD requirements.txt .
ADD imageclassifier.py .
RUN mkdir /home/dataset/
ADD dataset /home/dataset/
RUN pip install -r requirements.txt
RUN mkdir /home/output
#RUN pip3 list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 pip3 install -U 
RUN chmod +x imageclassifier.py
RUN python3 imageclassifier.py --trainmodel True
RUN python3 imageclassifier.py --predict True
ENTRYPOINT [ "sh" ]