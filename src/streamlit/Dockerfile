FROM python:3.10-slim

RUN mkdir /streamlit

COPY . /streamlit

WORKDIR /streamlit

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "main.py"]