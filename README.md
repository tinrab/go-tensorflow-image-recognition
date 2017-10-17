# Image Recognition API in Go using TensorFlow

<p align="center">
  <img src="./cover.jpg"/>
</p>

This is the underlying code for article [Build an Image Recognition API with Go and TensorFlow](https://outcrawl.com/image-recognition-api-go-tensorflow/).

## Running the service

Build the container.

```
$ docker-compose -f docker-compose.yaml up -d --build
```

Call the service.

```
$ curl localhost:8080/recognize -F 'image=@./cat.jpg'
{
  "filename": "cat.jpg",
  "labels": [
    { "label": "Egyptian cat", "probability": 0.39229771 },
    { "label": "weasel", "probability": 0.19872947 },
    { "label": "Arctic fox", "probability": 0.14527217 },
    { "label": "tabby", "probability": 0.062454574 },
    { "label": "kit fox", "probability": 0.043656528 }
  ]
}
```
