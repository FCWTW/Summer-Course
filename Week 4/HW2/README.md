# OpenCV practice: Image Stitching

## How to run:
> ### 1. Prerequisites:
> * make sure you have [Anaconda](https://www.anaconda.com/download) installed on your system.
>
> ### 2. Create environment:
> * open VScode from anaconda prompt and paste the following command
> ```bash
> conda create -n HW2 python=3.8
> ```
>
> * press **ctrl+shift+P** to open command palette, and check the **Python: Select Interpreter** option to select environment.
>
> * download dependencies with the following command
> ```bash
> pip install opencv-python
> pip install opencv-contrib-python
> ```
>
> ### 3. Execution: 
> * make sure images are in the correct path and click your run button : )
>
> <details>
>  <summary>參考資料</summary>
>    1. <a href="https://code.visualstudio.com/docs/python/environments#_create-a-conda-environment-in-the-terminal">Python environments in VS Code</a><br>
>    2. <a href="https://stackoverflow.com/questions/67750857/how-to-activate-conda-environment-in-vs-code">How to activate conda environment in vs code</a><br>
> </details>

## Example:
> ### 1. Find feature points with SIFT
> ![right](/Week%204/HW2/image/right_feature.jpg)
> ![left](/Week%204/HW2/image/left_feature.jpg)
>
> ### 2. Matching features with KNN match
> ![knn](/Week%204/HW2/image/feature_matching.jpg)
>
> ### 3. Combine images
> ![result](/Week%204/HW2/image/result.jpg)
>
> <details>
>  <summary>參考資料</summary>
>    1. <a href="https://opencv-python-tutorials.readthedocs.io/zh/latest/5.%20%E7%89%B9%E5%BE%81%E6%A3%80%E6%B5%8B%E5%92%8C%E6%8F%8F%E8%BF%B0/5.4.%20SIFT(Scale-Invariant%20Feature%20Transform)%E7%AE%80%E4%BB%8B/">SIFT(Scale-Invariant Feature Transform)簡介</a><br>
>    2. <a href="https://blog.csdn.net/zhangziju/article/details/79754652">應用OpenCV和Python進行SIFT演算法的實現</a><br>
>    3. <a href="https://towardsdatascience.com/image-stitching-using-opencv-817779c86a83">Image Stitching Using OpenCV</a><br>
>    4. <a href="https://blog.csdn.net/weixin_43810267/article/details/112643580">opencv-python 小白筆記（23）</a><br>
> </details>
