# Problem 1 solution:
>### 1. Use binary search twice to find the starting and ending position of a given target value.
>### 2. 有定義好class Problem_1，其中Problem_1_sol function的Input為int vector和指定的數字，output會直接列印出結果
>### 3. The recursive time function of binary reach is T(N)=T(N/2)+1, according to Master theorem, its time complexity is O(logN). This solution use binary search twice, so its time complexity is also O(logN).

---
# Problem 2 solution:
>### 1. Just Brute Force, use two for loop to update the longest substring.
>### 2. 有定義好class Problem_2，其中Problem_2_sol function的Input為string，output會直接列印出結果
>### 3. Time complexity is O(N^3)

---
# Problem 3 solution:
>### 1. Use "merge" of merge sort to combine two sorted array into a big array, however, we only need to combine first half of big array and the problem will be solved.
>### 2. 有定義好class Problem_3，其中Problem_3_sol function的Input為兩個int array和他們的size，output會直接列印出結果
>### 3. This solution scans the first half of combined array, so its time complexity is O(m+n).
