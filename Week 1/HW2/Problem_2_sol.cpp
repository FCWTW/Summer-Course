#include <iostream>
#include <string>

using namespace std;

class Problem_2 {
	public:
		void Problem_2_sol(string s) {
			int size = s.size();
			int start = 0;
			int length = 0;

			for(int i=0; i<size; i++) {
				for(int j=i; j<size; j++) {
					if(check(s, i, j)) {
					if(j-i+1 > length) {
						// update length of substring
						length = j-i+1;
						start = i;
						}
					}
				}
			}

			cout << "Output: \"";
			for(int i=start; i<(start+length); i++)
			cout << s[i];
			cout << "\"" << endl;
		}

	private:
		bool check(string &s, int index1, int index2) {
			while(index1 < index2) {
				if(s[index1] != s[index2])
					return false;
				index1++;
				index2--;
			}

			return true;
		}
};

//	test
int main() {
	Problem_2 test;
	string s = "babad";

	test.Problem_2_sol(s);

	return 0;
}