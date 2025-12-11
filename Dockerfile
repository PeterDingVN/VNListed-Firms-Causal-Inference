FROM python:3.11

WORKDIR /src

COPY src/ .
COPY exe.py .
RUN pip install --no-cache-dir -r requirement.txt

EXPOSE 3000

CMD ["python", "exe.py"]