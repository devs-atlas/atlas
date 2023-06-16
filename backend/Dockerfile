FROM python:3.11.4-slim-buster
WORKDIR /api

ENV PYTHONUNBUFFERED True

# Install core dependencies.
RUN apt-get update && apt-get install -y libpq-dev build-essential
RUN python -m pip install --upgrade pip

#Copy over requirements to docker system
COPY ./requirements.txt /api/requirements.txt
# Install Python dependencies.
RUN pip install --no-cache --upgrade  -r requirements.txt
# Copy over the rest of the code
COPY . /api
# Expose port 8000
CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]