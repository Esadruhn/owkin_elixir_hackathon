# this base image works in both CPU and GPU enabled environments
FROM europe-west1-docker.pkg.dev/elixir-hackathon-project-30/connect-tools/connect-tools:0.9.0

# install dependencies
RUN pip3 install pandas==1.3.4 numpy==1.21.4 tensorflow==2.7.0 scikit-learn==1.0.1

# add your algorithm script to docker image
ADD algo.py .

# define how script is run
ENTRYPOINT ["python3", "algo.py"]
