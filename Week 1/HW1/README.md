# Train a model to classify mnist dataset in docker container pytorch/pytorch
>* ### Image: 
>>[pytorch/pytorch](https://hub.docker.com/r/pytorch/pytorch/tags)
>
>
>* ### Build Container:
>> docker run -it --name pytorch_container -v "/home/user/Desktop/SummerCourse/Week 1/HW1":/workspace pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel
>
>
>* ### Start and get in container
>> docker start pytorch_container
>> docker exec -it pytorch_container bash
>> python /workspace/train.py
>
>
>* ### Result: ![result](/Images/HW1.jpg)