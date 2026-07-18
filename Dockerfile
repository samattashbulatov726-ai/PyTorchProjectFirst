FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import torchvision.models as models; models.resnet18(pretrained=True)"
COPY . .
CMD ["python", "main.py"]