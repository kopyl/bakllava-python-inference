I basically Dockerized this [Hugginface model](https://huggingface.co/SkunkworksAI/BakLLaVA-1)
into a serverless Runpod API.
You can find more info on this project in [Twitter](https://twitter.com/skunkworks_ai/status/1713372586225156392?s=46&t=RPmlEju3ShjhNkXFzQLRPQ&fbclid=IwAR2n2i9KMgn8e3ofB3-u6mZ94YhGfi7izuQY5aigVaOZode_iT4UKsJnR8A)

[Docker image](https://hub.docker.com/layers/kopyl/bakllava/latest/images/sha256-59f9bed9bd6b5e4593391c69823647891b235bb335dfc90644867e097dbcc139?context=repo): (kopyl/bakllava)

In order to build you'd need to ackquire a model cache (/root/.cache/huggingface). You can do it by running this Docker image once and then copying the cache from the container to your host machine. Then you can build the image with the cache.
