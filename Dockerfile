FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

EXPOSE 7860

# App runs internally on 8080
ENV PORT=8501

CMD ["streamlit", "run", "app.py"]