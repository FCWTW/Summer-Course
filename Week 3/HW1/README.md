# ROS Practise

## Build environment

> ### 1. Install ROS Noetic on Ubuntu
>>
>> - ### Setup your sources.list
>>
> > ```bash
> > sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
> > ```
>>
>> - ### Set up your keys
>>
> > ```bash
> > sudo apt install curl # if you haven't already installed curl
> > curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
> > ```
>>
>> - ### Installation
>>
> > ```bash
> > sudo apt update
> > sudo apt install ros-noetic-desktop-full
> > ```
>>
>> - ### Environment setup
>>
> > ```bash
> > echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
> > source ~/.bashrc
> > ```
>>
>> - ### Dependencies for building packages
>>
> > ```bash
> > sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
> > ```
> >
> > ```bash
> > sudo apt install python3-rosdep
> > ```
> >
> > ```bash
> > sudo rosdep init
> > rosdep update
> > ```
>
> ### 2. Build ROS workspace
>
> ```bash
> mkdir -p ~/catkin_ws/src
> cd ~/catkin_ws/
> catkin_make
> source ~/catkin_ws/devel/setup.bash
> ```
>
> ### 3. Create ROS package
>
> ```bash
> cd ~/catkin_ws/src
> catkin_create_pkg my_package std_msgs roscpp
> ```
>
> ```bash
> cd ..
> catkin_make
> ```
>
> <details>
>  <summary>參考資料</summary>
>    1. <a href="https://wiki.ros.org/noetic/Installation/Ubuntu">Ubuntu install of ROS Noetic</a><br>
>    2. <a href="https://www.youtube.com/watch?v=9Ia4vQYXEc4">建置影片</a><br>
>    3. <a href="http://wiki.ros.org/turtlesim">Turtlesim</a><br>
> </details>

---

## How to run

> ### 1. Add code file to ~catkin_ws/src/my_package/src
>
> ### 2. Edit CMakeLists.txt in my_package and add following content
>
> ```cmake
> add_executable(node1 src/Node1.cpp)
> target_link_libraries(node1 ${catkin_LIBRARIES})
>
> add_executable(node2 src/Node2.cpp)
> target_link_libraries(node2 ${catkin_LIBRARIES})
>
> add_executable(node3 src/Node3.cpp)
> target_link_libraries(node3 ${catkin_LIBRARIES})
>
> add_executable(node4 src/Node4.cpp)
> target_link_libraries(node4 ${catkin_LIBRARIES})
> ```
>
> ### 3. Compile
>
> ```bash
> cd ~/catkin_ws
> catkin_make
> ```
>
> ### 4. Set environment variables
>
> ```bash
> echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
> source ~/.bashrc
> ```
>
> ### 5. Run
>
> ```bash
> roscore
> ```
>
> ```bash
> rosrun my_package node1
> ```
