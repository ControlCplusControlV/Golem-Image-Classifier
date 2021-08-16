FROM python:3.8-slim
VOLUME /golem/work
COPY ImageClassification.py /golem/run/
COPY ClassifierClient.py /golem/run/
COPY requirements.txt /
RUN pip install -r requirements.txt
RUN chmod +rwx /golem/run/*
RUN chmod +rwx /bin/tar
