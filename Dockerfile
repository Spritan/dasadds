# app/Dockerfile

FROM python:3.9-slim
FROM ubuntu

WORKDIR /app

COPY . /app

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     software-properties-common \
#     git \
#     && rm -rf /var/lib/apt/lists/*
RUN add-apt-repository ppa:opencpn/opencpn
RUN apt-get update
RUN apt-get install opencpn
RUN apt-get install -y libgl1-mesa-dev, freeglut3-dev

RUN pip3 install -r requirements.txt

EXPOSE 8000

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8000", "--server.address=0.0.0.0"]