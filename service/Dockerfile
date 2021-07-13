FROM python:3.8-slim
VOLUME /golem/work 
COPY /model/ /golem/work/model/
COPY /dataset/ /golem/work/dataset/
COPY imageclassifier.py /golem/run/
COPY test2.jpg /golem/work
RUN pip install numpy
RUN pip install keras
RUN pip install tensorflow
RUN pip install argparse
RUN pip install sklearn
RUN pip install Pillow
RUN chmod +rwx /golem/work/*
RUN chmod +rwx /golem/run/*
RUN chmod +rwx /golem/run/imageclassifier.py
WORKDIR /golem/run
ENTRYPOINT [ "sh" ]
