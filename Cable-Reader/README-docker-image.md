# Setting the project
> docker build -t sacamos -f Dockerfile .

# Container creation
> docker run --name sacamos-python -it -d -v ./:/home/repository sacamos

## Run the containers
> docker start sacamos-python

## Access containers
> docker exec -it sacamos-python bash

## Executing python files
> python3 assertsTests.py

## Stop the containers
> docker stop sacamos-python
