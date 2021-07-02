FROM python:3.8-slim
COPY itjustworks /golem/work/output
COPY dataset /golem/work/dataset/
COPY imageclassifier.py /golem/work/
RUN pip install urllib3
RUN pip install sklearn
RUN pip install numpy
RUN pip install mahotas
RUN pip install opencv-python-headless
RUN pip install h5py
RUN pip install matplotlib
RUN chmod +x /golem/work/*
ENTRYPOINT [ "sh" ]
