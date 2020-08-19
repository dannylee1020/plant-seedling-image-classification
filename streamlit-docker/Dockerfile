FROM python:3.7

# define working directory within docker container image
WORKDIR /opt/streamlit_app

# installing requirements
COPY requirements.txt .
RUN pip install -r requirements.txt


# docker COPY method copies files not directory -> we need to create directories within docker image 
COPY app.py .
COPY test_image /opt/streamlit_app/test_image/
COPY model /opt/streamlit_app/model/

# for local testing
EXPOSE 8501

# for local testing
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]


# for dynamic port assigned by Heroku server
#CMD streamlit run app.py --server.port $PORT



