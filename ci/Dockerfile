FROM nvcr.io/nvidia/tensorflow:24.09-tf2-py3

# Set python environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY . .

# Install deps
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r ./ci/requirements.txt

# Download the Kaggle datasetTBD
#RUN kaggle datasets download -d kazanova/sentiment140 -p /app/data/twitter \
#    && unzip /app/data/twitter/*.zip -d /app/data/twitter \
#    && rm /app/data/twitter/*.zip

EXPOSE 5000
CMD ["python", "./app/main.py"]
