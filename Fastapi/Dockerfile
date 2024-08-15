FROM python:3.9.12

WORKDIR /the/workdir/path

COPY ./requirment2.txt /code/requirment2.txt

RUN pip install --no-cache-dir -r /code/requirment2.txt

COPY ./app_fast /code/app

EXPOSE 8000

CMD ["uvicorn", "app.app_fast:app", "--host", "0.0.0.0", "--port","8000"]
