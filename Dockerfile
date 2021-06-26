FROM python:3.8
WORKDIR /usr/src/app

COPY ./ ./

RUN apt update \
  && apt-get install -y libsqlite3-dev libbz2-dev libncurses5-dev libgdbm-dev liblzma-dev libssl-dev tcl-dev tk-dev libreadline-dev \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED 1
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD bash