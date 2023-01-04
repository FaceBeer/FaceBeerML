FROM tensorflow/tensorflow:latest-gpu

COPY src .

ENTRYPOINT ["python", "main.py"]