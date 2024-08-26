# OpenCV practice: Active Contour

## How to run:
> ### 1. Prerequisites:
> * install cmake, pkg-config, lib* with following commands
> ```bash
> sudo apt-get install gcc g++ cmake pkg-config build-essential
> sudo apt-get install libgtk2.0-dev libavcodec-dev libavformat-dev libtiff5-dev libswscale-dev
> ```
>
> * download opencv from github
> ```bash
> git clone https://github.com/opencv/opencv.git
> ```
>
> * compile and install opencv
> ```bash
> cd opencv 
> mkdir build
> cd build
> cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local ..
> make -j$(nproc)
> sudo make install
> ```
>
> * update system library cache
> ```bash
> echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/opencv.conf
> sudo ldconfig
> ```
>
> ### 2. Compilation: use the following command in terminal
> ```bash
> g++ hw1.cpp -o hw1 `pkg-config --cflags --libs opencv4`
> ```
>
> ### 3. Execution: use the following command in terminal
> ```bash
> ./hw1
> ```
>
## Example

> <details>
>  <summary>參考資料</summary>
>    1. <a href="https://hackmd.io/@xpg109/H1wZiRUNF">Linux環境下安裝OpenCV 4.5.3 (C++)</a><br>
>    2. <a href="https://www.cnblogs.com/chenzhen0530/p/14660498.html">OpenCV-C++ Sobel算子使用</a><br>
>    3. <a href="https://stackoverflow.com/questions/21079758/opencv-c-drawing-on-image">Opencv c++ drawing on image</a><br>
>    4. <a href="https://www.cs.ait.ac.th/~mdailey/cvreadings/Kass-Snakes.pdf">Snakes: Active Contour Models</a><br>
>    5. <a href="https://github.com/webzhuce07/Digital-Image-Processing/blob/master/Segment/CvSnakeImage/snakeimage.cpp#L451">用OpenCV3.x重新實現cvSnakeImage函數</a><br>
>    6. <a href="https://github.com/bosley/Active-Contours/blob/master/src/Vision/Algorithms/customsnake.cpp">custom snake algorithm</a><br>
> </details>

