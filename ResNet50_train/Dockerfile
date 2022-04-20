FROM graphcore/pytorch:latest

WORKDIR /root

RUN apt-get update && apt-get install -y \
  ssh \
  && rm -rf /var/lib/apt/lists/*

COPY . /root

SHELL ["/bin/bash", "-c"]

RUN pip install -r /root/requirements.txt
# RUN echo "source /root/venv/bin/activate" >> .bashrc

CMD ["/bin/bash", "-c", "/root/train/rn50_pod16.sh"]
