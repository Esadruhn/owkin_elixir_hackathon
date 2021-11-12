# this base image works in both CPU and GPU enabled environments
FROM esadruhn/elixir_connect:0.9.0

# install dependencies
RUN pip3 install pandas==1.1.5 numpy==1.19.5 tensorflow==2.6.2 scikit-learn==0.24

# add your algorithm script to docker image
ADD algo.py .

# define how script is run
ENTRYPOINT ["python3", "algo.py"]
