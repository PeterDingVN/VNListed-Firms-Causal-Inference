FROM python:3.11

WORKDIR /app

COPY src/ ./src
COPY exe.py .
COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

EXPOSE 8501

CMD ["python", "exe.py", "--server.address=0.0.0.0"]