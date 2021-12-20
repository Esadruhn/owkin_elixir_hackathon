# this base image works in both CPU and GPU enabled environments
FROM esadruhn/elixir_connect:0.9.0

# install dependencies
RUN pip3 install scikit-learn==0.24

# add your metrics script to docker image
ADD metrics.py .

# define how script is run
ENTRYPOINT ["python3", "metrics.py"]
