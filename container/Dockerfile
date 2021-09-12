FROM continuumio/miniconda3:latest
COPY . /usr/app/
EXPOSE 5000:5000
WORKDIR /usr/app/
RUN conda install python=3.8
RUN pip install -r requirements.txt
CMD python app.py