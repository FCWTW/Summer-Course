# Problem 1 solution:
>* ## **Method**:
>>### use binary search twice to find the starting and ending position of a given target value.
>
>
>* ## **How to run**:
>>### 1.	Prerequisites: make sure you have g++ installed on your system.
>>### 2.	Compilation: use the following command in terminal.
>>```bash
>>g++ -o Problem_1 Problem_1.cpp
>>```
>>### 3.	**Excution**: After compiling the code, you can run with the following command.
>>```bash
>>./Problem_1
>>```
>>### 4.	**Dependencies**: This program does not require any additional libraries.
>
>
>*	## **Example**:
>>### Problem_1_sol 的 Input 為 int vector 和指定的數字， output 會直接列印出結果
>> ![P1]{/Images/Problem_1_sol.jpg}
>
>
>*	## **Time complexity**:
>>### The recursion time function of binary search is T(N)=T(N/2)+1, according to Master theorem, its time complexity is O(logN).
>>### This solution use binary search twice, so its time complexity is also O(logN).

---
# Problem 2 solution:
>* ## **Method**:
>>### Just Brute Force, use two for loop to update the longest substring.
>
>
>* ## **How to run**:
>>### 1.	Prerequisites: make sure you have g++ installed on your system.
>>### 2.	Compilation: use the following command in terminal.
>>```bash
>>g++ -o Problem_2 Problem_2.cpp
>>```
>>### 3.	**Excution**: After compiling the code, you can run with the following command.
>>```bash
>>./Problem_2
>>```
>>### 4.	**Dependencies**: This program does not require any additional libraries.
>
>
>*	## **Example**:
>>### Problem_2_sol 的 Input 為 string ， output 會直接列印出結果
>> ![P1]{/Images/Problem_1_sol.jpg}
>
>
>*	## **Time complexity**:
>>### This solution use two for loop and one while loop, so its time complexity is O(N^3).

---
# Problem 3 solution:
>* ## **Method**:
>>### Use "merge" of merge sort to combine two sorted array into a big array, however, we only need to combine first half of big array and the problem will be solved.
>
>
>* ## **How to run**:
>>### 1.	Prerequisites: make sure you have g++ installed on your system.
>>### 2.	Compilation: use the following command in terminal.
>>```bash
>>g++ -o Problem_3 Problem_3.cpp
>>```
>>### 3.	**Excution**: After compiling the code, you can run with the following command.
>>```bash
>>./Problem_3
>>```
>>### 4.	**Dependencies**: This program does not require any additional libraries.
>
>
>*	## **Example**:
>>### Problem_3_sol 的 Input 為兩個 int array 和他們的 size ， output 會直接列印出結果
>> ![P1]{/Images/Problem_1_sol.jpg}
>
>
>*	## **Time complexity**:
>>### This solution scans the first half of combined array, so its time complexity is O(m+n), m and n are sizes of input array.
