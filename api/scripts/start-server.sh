python src/manage.py collectstatic
python src/manage.py migrate
python -m gunicorn src.api.asgi:application -k uvicorn.workers.UvicornWorker --max-worker=8
