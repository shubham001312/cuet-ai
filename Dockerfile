FROM python:3.11-slim

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=7860

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

EXPOSE 7860

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
