#include <iostream>
#include <vector>

using namespace std;

class Problem_1 {
    public:
        void Problem_1_sol(const vector<int>& nums, int target) {
            int len = nums.size();
            
            if(len == 0)
                cout << "Output: [ " << begin << ", " << end << " ]\n";

            else {
                begin = BinarySearchLeft(nums, 0, len-1, target);
                end = BinarySearchRight(nums, 0, len-1, target);

                cout << "Output: [ " << begin << ", " << end << " ]\n";
            }
        }

    private:
        int begin = -1;
        int end = -1;

        // customized binary search which only go left
        int BinarySearchLeft(const vector<int>& arr, int left, int right, int x)
        {
            if(right >= left) {
                int mid = left + (right - left) / 2;

                if(arr[mid] == x) {
                    int tmp = BinarySearchLeft(arr, left, mid - 1, x);
                    if(tmp == -1)
                        return mid;
                    else
                        return tmp;
                }
                else if(arr[mid] > x)
                    return BinarySearchLeft(arr, left, mid - 1, x);
                else
                    return BinarySearchLeft(arr, mid + 1, right, x);
            }
            return -1;
        }

        // customized binary search which only go right
        int BinarySearchRight(const vector<int>& arr, int left, int right, int x)
        {
            if(right >= left) {
                int mid = left + (right - left) / 2;

                if(arr[mid] == x) {
                    int tmp = BinarySearchRight(arr, mid + 1, right, x);
                    if(tmp == -1)
                        return mid;
                    else
                        return tmp;
                }
                else if(arr[mid] > x)
                    return BinarySearchRight(arr, left, mid - 1, x);
                else
                    return BinarySearchRight(arr, mid + 1, right, x);
            }
            return -1;
        }
};

// test
int main()
{
    Problem_1 test;
    vector<int> nums = {5, 7, 7, 8, 8, 10};

    test.Problem_1_sol(nums, 8);
    test.Problem_1_sol(nums, 12);

    return 0;
}