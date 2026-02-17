FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY pixel_mosaic/ pixel_mosaic/
COPY main.py .

RUN pip install --no-cache-dir .

# Create directories
RUN mkdir -p images output

ENTRYPOINT ["pixel-mosaic"]
CMD ["batch"]
