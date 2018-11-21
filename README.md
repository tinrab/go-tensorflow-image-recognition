# Image Recognition API in Go using TensorFlow

<p align="center">
  <img src="./cover.jpg"/>
</p>

This is the underlying code for article [Build an Image Recognition API with Go and TensorFlow](https://outcrawl.com/image-recognition-api-go-tensorflow).

## Running the service

Build the image.

```
$ docker build -t localhost/recognition .
```

Run servicve in a container.

```
$ docker run -p 8080:8080 --rm localhost/recognition
```

Call the service.

```
$ curl localhost:8080/recognize -F 'image=@./cat.jpg'
{
  "filename": "cat.jpg",
  "labels": [
    { "label": "tabby", "probability": 0.45087516 },
    { "label": "Egyptian cat", "probability": 0.26096493 },
    { "label": "tiger cat", "probability": 0.23208225 },
    { "label": "lynx", "probability": 0.050698064 },
    { "label": "grey fox", "probability": 0.0019019963 }
  ]
}
```
