FROM python:3.8-slim
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY models/model.pkl /models/model.pkl
COPY models/transformer.pkl /models/transformer.pkl
COPY app.py /app.py
COPY src /src

WORKDIR .

ENV PATH_TO_MODEL="models/model.pkl"
ENV PATH_TO_TRANSFORMER="models/transformer.pkl"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]