FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt update && apt install -y python3 wget python-is-python3
RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py

ENV PYTHONUNBUFFERED=1

RUN apt update && apt install --no-install-recommends -y git
RUN git clone https://github.com/SkunkworksAI/BakLLaVA.git
WORKDIR /BakLLaVA

COPY cache-files/huggingface /root/.cache/huggingface

RUN \
    pip install -e . && \
    pip uninstall transformers -y && \
    pip install transformers==4.34.0 && \
    pip install runpod==1.3.0 && \
    pip install pydantic==1.10.11 && \
    pip cache purge

COPY handler.py /handler.py

CMD ["python", "/handler.py"]


# sudo depot build --project xxx . -t kopyl/bakllava --push

# TODO:
# - I see `Downloading shards:`. Even though it's almost instant, it's still better to cache it.
# - Find a way to reduce Docker image size. (https://github.com/huggingface/transformers/issues/23904#issuecomment-1571553010)