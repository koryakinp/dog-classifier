FROM tensorflow/tensorflow
RUN apt-get -y update
RUN apt-get -y install git
RUN apt-get install git-lfs
RUN pip install pillow
RUN pip install scipy
RUN git clone https://github.com/koryakinp/dog-classifier.git
WORKDIR "dog-classifier"
RUN git lfs pull
ENTRYPOINT ["python", "/dog-classifier/classifier.py"]
