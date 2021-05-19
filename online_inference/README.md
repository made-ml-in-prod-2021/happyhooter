Docker build: ```docker build -t happyhooter/online_inference:v2 .```

Docker run: ```docker run -p 8000:8000 happyhooter/online_inference:v2```

Docker pull: ```docker pull happyhooter/online_inference:v2```

Сделать запрос: ```python -m src.make_request```

Тесты: ```python -m pytest -v```

Для сокращения размера docker image использовал python:3.8-slim вместо python:3.8, удалось добиться уменьшения размера с 1.2Gb до 600Mb