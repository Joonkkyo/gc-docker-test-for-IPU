FROM graphcore/pytorch:latest

WORKDIR /worker

RUN apt-get update && apt-get install -y \
  && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

COPY . /worker

RUN pip install --no-cache -r requirements.txt

ENV LC_ALL C.UTF-8 

ENV LANG C.UTF-8

ENTRYPOINT ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]
