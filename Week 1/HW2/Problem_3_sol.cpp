#include <iostream>
#include <iomanip>

using namespace std;

class Problem_3 {
  	public:
    	void Problem_3_sol(const int nums1[], const int nums2[], int m, int n) {
			int mid = (m+n)/2;
			int merge[mid+1];
			int index1 = 0;
			int index2 = 0;

			// merge array
			for(int i=0; i<mid+1; i++) {
				if(index1 == m) {
					merge[i] = nums2[index2];
					index2++;
				}
				else if(index2 == n) {
					merge[i] = nums1[index1];
					index1++;
				}
				else if(nums1[index1] < nums2[index2]) {
					merge[i] = nums1[index1];
					index1++;
				}
				else {
					merge[i] = nums2[index2];
					index2++;
				}
			}
			
			//	convert result from int to float
			if((m+n)%2 == 0) {
				float tmp1 = static_cast<float>(merge[mid]);
				float tmp2 = static_cast<float>(merge[mid-1]);

				cout << "Output: " << fixed << setprecision(3) << (tmp1+tmp2)/2 << endl;
			}
			else {
				float tmp1 = static_cast<float>(merge[mid]);

				cout << "Output: " << fixed << setprecision(3) << tmp1 << endl;
			}
    }
};

// test
int main() {
  	Problem_3 test;
  	int nums1[] = {1,3};
  	int nums2[] = {2,8};

  	test.Problem_3_sol(nums1, nums2, sizeof(nums1)/sizeof(nums1[0]), sizeof(nums2)/sizeof(nums2[0]));

  	return 0;
}