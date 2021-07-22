FROM python:3.8-slim
VOLUME /golem/work
COPY /dataset/ /golem/work/dataset/
COPY ImageClassification.py /golem/run/
COPY ClassifierClient.py /golem/run/
COPY cats-dogs-monkeys-cows/ /golem/work/cats-dogs-monkeys-cows/
RUN pip install numpy
RUN pip install keras
RUN pip install tensorflow
RUN pip install argparse
RUN pip install rpyc
RUN pip install sklearn
RUN pip install Pillow
RUN chmod +rwx /golem/work/*
RUN chmod +rwx /golem/run/*
RUN chmod +rwx /bin/tar
ENTRYPOINT [ "sh" ]
