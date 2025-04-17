FROM semitechnologies/transformers-inference:custom
RUN MODEL_NAME=ipipan/silver-retriever-base-v1.1 USE_SENTENCE_TRANSFORMERS_VECTORIZER=true ONNX_RUNTIME=true ./download.py
