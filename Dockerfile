FROM python:3.8-slim-buster
#VOLUME golem/input
RUN mkdir output
RUN mkdir dataset
COPY dataset dataset/
COPY imageclassifier.py .
RUN pip install urllib3
RUN pip install sklearn
RUN pip install numpy
RUN pip install mahotas
RUN pip install opencv-python-headless
RUN pip install h5py
RUN pip install matplotlib
RUN chmod +x imageclassifier.py
ENTRYPOINT [ "sh" ]
