# Train a model to classify mnist dataset in docker container pytorch/pytorch
>* Image: [pytorch/pytorch](https://hub.docker.com/r/pytorch/pytorch/tags)
>* Container management:
>> <font color="#F7A004">Building</font> docker run -it --name pytorch_container -v "/home/user/Desktop/SummerCourse/Week 1/HW1":/workspace pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel
>> <font color="#F7A004">Starting</font> docker start pytorch_container
>> <font color="#F7A004">Get in</font> docker exec -it pytorch_container bash
>> <font color="#F7A004">Run file</font> python /workspace/train.py
>* Result: [result]()