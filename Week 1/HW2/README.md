# Problem 1 solution:
>## 1. Use binary search twice to find the starting and ending position of a given target value.
>## 2. The recursive time function of binary reach is T(N)=T(N/2)+1, according to Master theorem, its time complexity is O(logN). This solution use binary search twice, so its time complexity is also O(logN).

---
# Problem 2 solution:
>## 1. Just Brute Force, its time complexity is O(N^3)

---
# Problem 3 solution:
>## 1. Use "merge" of merge sort to combine two sorted array into a big array, however, we only need to combine first half of big array and the problem will be solved.
>## 2. This solution scans the first half of combined array, so its time complexity is O(m+n).