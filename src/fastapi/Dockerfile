FROM tiangolo/uvicorn-gunicorn:python3.10-slim

RUN mkdir /fastapi

COPY . /fastapi

WORKDIR /fastapi

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]