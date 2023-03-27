FROM python:3.6
ADD . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN python -m spacy download en
CMD [ "python", "server.py" ]
#CMD [ "gunicorn --workers=4 --bind 0.0.0.0:5000 server:app"]
#CMD ["gunicorn" , "--workers", "2", "-b", "0.0.0.0:5000", "server:app"]
