FROM python:3.8-slim
VOLUME /golem/work
COPY itjustworks /golem/work/output
COPY /h5/ /golem/run/h5/
COPY imageclassifier.py /golem/run/
RUN pip install urllib3
RUN pip install sklearn
RUN pip install numpy
RUN pip install mahotas
RUN pip install opencv-python-headless
RUN pip install h5py
RUN pip install matplotlib
RUN chmod +rwx /golem/work/*
RUN chmod +rwx /golem/run/*
ENTRYPOINT [ "sh" ]
