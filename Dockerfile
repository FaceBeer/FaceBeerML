FROM tensorflow/tensorflow:latest-gpu

RUN pip install matplotlib numpy pandas scipy

COPY src .

ENTRYPOINT ["python", "main.py"]