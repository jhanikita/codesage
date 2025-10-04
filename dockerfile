FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install streamlit streamlit-chat uvicorn

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port 8000 & streamlit run src/frontend.py --server.port 8501 --server.headless true"]
