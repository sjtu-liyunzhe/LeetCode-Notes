### LeetCode 761

##### 日期

2022.08.08

##### 题目描述

![image-20220808182413140](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220808182413140.png)

##### 思路

- 特殊二进制序列条件等价于有效括号；如(())()(())

- 定义的操作相当于相邻同级括号之间可以交换

  - 任意交换得到字典序最大  =>  相当于冒泡排序，排序即可

  - 排序：参照剑指Offer 45，a + b > b + a

- 不同的括号内部交换相互独立，可以分治，用递归来做
- 分割出同级的字符串，对同级的字符串进行排序，每个字符串内部递归

##### 代码

```cpp
class Solution {
public:
    string makeLargestSpecial(string s) {
        if(s.size() <= 2) return s;
        int cnt = 0;
        vector<string> vstr;
        string str;
        for(auto& ch : s)
        {
            str += ch;
            if(ch == '1') ++cnt;
            else
            {
                if(--cnt == 0)
                {
                    vstr.push_back('1' + makeLargestSpecial(str.substr(1, str.size() - 2)) + '0');
                    str.clear();
                }
            }
        }
        sort(vstr.begin(), vstr.end(), [](string& a, string& b){
            return a + b > b + a;
        });
        string res;
        for(auto& str : vstr) res += str;
        return res;
    }
};
```

### LeetCode 636

##### 日期

2022.08.07

##### 题目描述

![image-20220808185738073](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220808185738073.png)

##### 测试样例

![image-20220808185804382](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220808185804382.png)

##### 思路

- 用栈来模拟函数的执行
- 函数开始执行，“start”，将其压入栈中；结束执行，“end”，弹出；
- 入栈和出栈时更新栈顶元素开始执行的时间戳

##### 代码

```cpp
class Solution {
private:
    vector<string> split(string& str, char ch)
    {
        int n = str.size();
        vector<string> res;
        int start = 0;
        for(int i = 0; i < n; ++i)
        {
            if(str[i] == ch)
            {
                res.push_back(str.substr(start, i - start));
                while(str[i] == ch) ++i;
                start = i;
            }
        }
        if(start < n) res.push_back(str.substr(start));
        return res;
    }
public:
    vector<int> exclusiveTime(int n, vector<string>& logs) {
        vector<int> res(n);
        stack<pair<int,int>> stk;       // pair: id, 函数开始执行时间戳
        for(auto& str : logs)
        {
            vector<string> tem = split(str, ':');
            int id = stoi(tem[0]), time = stoi(tem[2]);
            string flag = tem[1];
            if(flag[0] == 's')
            {
                if(!stk.empty())
                {
                    auto& index = stk.top();
                    res[index.first] += time - index.second;
                    index.second = time;
                }
                stk.push({id, time});
            }
            else if(flag[0] == 'e')
            {
                res[id] += time - stk.top().second + 1;
                stk.pop();
                if(!stk.empty())
                    stk.top().second = time + 1;
            }
        }
        return res;
    }
};
```

### LeetCode 673

##### 日期

2022.08.09

##### 题目描述

![image-20220809111836443](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220809111836443.png)

##### 测试样例

![image-20220809111905498](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220809111905498.png)

##### 思路

- 找最长递增子序列长度：动态规划，参考LC 300
  - dp[i]表示以i下标元素结尾（以nums[i]结尾）的递增子序列的最长长度

- <font color = 'red'>如何维护最长长度的个数 </font>
  - 用数组cnt[i]表示以nums[i]结尾的最长递增子序列的个数
  - nums[i] > nums[j]前提下，在[0, i-1]的范围内，找到了j，使得dp[j] + 1 > dp[i]，说明找到了新的的更长的递增子序列;
    - <font color = 'blue'>cnt[i] = cnt[j]；</font>
  
  - nums[i] > nums[j]前提下，在[0, i-1]的范围内，找到了j，使得dp[j] + 1 == dp[i]，说明找到了两个相同长度的递增子序列；
    - <font color = 'blue'>cnt[i] += cnt[j];</font>

##### 代码

```cpp
class Solution {
public:
    int findNumberOfLIS(vector<int>& nums) {
        int n = nums.size();
        if(n <= 1) return n;
        vector<int> dp(n, 1), cnt(n, 1);      // dp[i]: 以下标i元素结尾的递增子序列最长长度，个数
        int maxLen = 0, res = 0;
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < i; ++j)
            {
                if(nums[j] < nums[i])
                {
                    if(dp[i] < dp[j] + 1)
                    {
                        dp[i] = dp[j] + 1;
                        cnt[i] = cnt[j];
                    }
                    else if(dp[i] == dp[j] + 1)
                        cnt[i] += cnt[j];
                }
            }
            if(dp[i] > maxLen)
            {
                maxLen = dp[i];
                res = cnt[i];
            }
            else if(dp[i] == maxLen)
                res += cnt[i];
        }
        return res;
    }
};
```

------



### LeetCode 202

##### 日期

2022.08.13

##### 题目描述

![image-20220813224638926](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220813224638926.png)

##### 测试样例

![image-20220813224729531](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220813224729531.png)

##### 思路

- 哈希表判断重复

##### 代码

```c++
class Solution {
private:
    int helper(int& n)
    {
        int res = 0;
        while(n)
        {
            res += pow((n % 10), 2);
            n /= 10;
        }
        return res;
    }
public:
    bool isHappy(int n) {
        if(n == 1) return true;
        unordered_map<int, int> hash;
        while(true)
        {
            n = helper(n);
            if(n == 1) return true;
            if(++hash[n] > 1) return false;
        }
    }
};
```

------



### LeetCode 768

##### 题目描述

![image-20220813224850606](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220813224850606.png)

##### 测试样例

![image-20220813224915909](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220813224915909.png)

##### 思路1

- 排序+前缀和 OR 排序+哈希表
  - 排序后的前缀和与原数组前缀和相同，可以划分

##### 代码1

```cpp
typedef long long LL;
class Solution {
public:
    int maxChunksToSorted(vector<int>& arr) {
        vector<int> sorted = arr;
        sort(sorted.begin(), sorted.end());
        int n = arr.size(), res = 0;
        LL sum_1 = 0, sum_2 = 0;
        for(int i = 0; i < n; ++i)
        {
            sum_1 += arr[i];
            sum_2 += sorted[i];
            if(sum_1 == sum_2) ++res;
        }
        return res;
    }
};

// 哈希表
typedef long long LL;
class Solution {
public:
    int maxChunksToSorted(vector<int>& arr) {
        vector<int> sorted = arr;
        sort(sorted.begin(), sorted.end());
        int n = arr.size();
        unordered_map<int, int> hash;
        int cnt = 0, res = 0;
        for(int i = 0; i < n; ++i)
        {
            ++hash[arr[i]];
            if(hash[arr[i]] == 0) --cnt;
            else if(hash[arr[i]] == 1) ++cnt;
            --hash[sorted[i]];
            if(hash[sorted[i]] == 0) --cnt;
            else if(hash[sorted[i]] == -1) ++cnt;
            if(cnt == 0) ++res;
        }
        return res;
    }
};
```

##### 思路2

- 单调栈





------



### LeetCode 201

##### 题目描述

![image-20220814222424940](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220814222424940.png)

##### 标签

- 位运算

##### 测试样例

![image-20220814222453601](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220814222453601.png)

##### 思路

- 对所有数字执行按位与运算的结果是所有对应二进制字符串的公共前缀再用零补上后面的剩余位
  - 即找到两个数的公共前缀
    - 1、移位判断是否相等
    - 2、Brian Kernighan 算法，清除最右边的1：每次对 *num* 和 *num - 1* 之间进行按位与运算

##### 代码

```cpp
class Solution {
public:
    int rangeBitwiseAnd(int left, int right) {
        int shift = 0, res = 0;
        while(left != right)
        {
            left >>= 1;
            right >>= 1;
            ++shift;
        }
        res = left << shift;
        return res;
    }
};

// Brian Kernighan
class Solution {
public:
    int rangeBitwiseAnd(int m, int n) {
        while (m < n) {
            // 抹去最右边的 1
            n = n & (n - 1);
        }
        return n;
    }
};
```

***

### AcWing 4405

##### 题目描述

![image-20220815163552910](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220815163552910.png)

##### 测试样例

![image-20220815163611715](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220815163611715.png)

##### 思路

- 暴力枚举时间复杂度*O(n^4 / 2) * ，超时；需要降到*O(n^3 / 2)*
- 枚举上下边界，转化为1维问题，求上下边界内满足要求的子矩阵
  - 需要求每一列区间和，先求前缀和，s[i] [j] 表示第j列中前i行的前缀和
  - 双指针求两行之间满足要求的矩阵个数
    - 对于右边界right，左边界left表示<font color = 'red'>最左边边界的满足不超过K的位置</font>
    - 双指针的应用场景：left = f(right)单调，right往右走，left也往右走

##### 代码

```cpp
#include <iostream>
#include <vector>
using namespace std;

typedef long long LL;
const int N = 510, M = 510;
int arr[N][M], s[N][M];
int n, m, k;
int main()
{
    cin >> n >> m >> k;
    for(int i = 1; i <= n; ++i)
    {
        for(int j = 1; j <= m; ++j)
        {
            cin >> arr[i][j];
            s[i][j] = s[i - 1][j] + arr[i][j];
        }
    }
    LL res = 0;
    for(int i = 1; i <= n; ++i)
    {
        for(int j = i; j <= n; ++j)
        {
            int sum = 0;
            for(int l = 1, r = 1; r <= m; ++r)
            {
                sum += s[j][r] - s[i - 1][r];
                while(sum > k)
                {
                    sum -= s[j][l] - s[i - 1][l];
                    ++l;
                }
                res += r - l + 1;
            }
        }
    }
    cout << res;
    return 0;
}
```

***

### LeetCode 56

##### 题目描述

![image-20220826153816232](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220826153816232.png)

##### 测试样例

![image-20220826153843754](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220826153843754.png)

##### 思路

- 排序
  - 固定左端点，更新右端点；
  - 相邻两区间不相交，插入到答案中

##### 代码

```cpp
class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        int n = intervals.size();
        if(n <= 1) return intervals;
        sort(intervals.begin(), intervals.end());
        vector<vector<int>> res;
        for(int i = 0; i < n; ++i)
        {
            int left = intervals[i][0], right = intervals[i][1];
            if(res.size() == 0 || res.back()[1] < left) res.push_back({left, right});
            if(res.back()[1] >= left) res.back()[1] = max(res.back()[1], right);
        }
        return res;
    }
};
```

***

### LeetCode 763

##### 题目描述

![image-20220826154814240](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220826154814240.png)

##### 测试用例

![image-20220826154835290](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220826154835290.png)

##### 思路

- 哈希表记录每个字母的最后出现位置
- 从前向后遍历，更新end，当i == end时分割

##### 代码

```cpp
class Solution {
private:
    int pos[26];
public:
    vector<int> partitionLabels(string s) {
        int n = s.size();
        for(int i = 0; i < n; ++i) pos[s[i] - 'a'] = i;
        int start = 0, end = 0;
        vector<int> res;
        for(int i = 0; i < n; ++i)
        {
            end = max(end, pos[s[i] - 'a']);
            if(i == end)
            {
                res.push_back(end - start + 1);
                start = end + 1;
            }
        }
        return res;
    }
};
```

### LeetCode 1345

##### 日期

2022.08.27

##### 题目描述

![image-20220827221537543](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220827221537543.png)

##### 测试用例

![image-20220827221602608](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220827221602608.png)

##### 思路

- 可以由相同数值更新最小步数，无法用DP
- 转换为图论单源最短路问题（BFS）
  - 将相同数值的节点加到哈希表里
  - 时间复杂度O(V + E)；顶点数 + 边数；V = n；E有可能n^2；常规BFS会超时；

- 优化
  - 每次访问过相同数值的节点后，从哈希表中删除，即便再次遍历到也是visited，防止再次访问

##### 代码

```cpp
class Solution {
public:
    int minJumps(vector<int>& arr) {
        int n = arr.size();
        if(n <= 1) return 0;
        unordered_map<int, vector<int>> hash;
        for(int i = 0; i < n; ++i) hash[arr[i]].push_back(i);
        vector<int> visited(n), dist(n);
        queue<int> que;
        que.push(0);
        visited[0] = 1;
        while(!que.empty())
        {
            auto tmp = que.front();
            que.pop();
            for(auto& x : hash[arr[tmp]])
            {
                if(x != tmp && !visited[x])
                {
                    que.push(x);
                    dist[x] = dist[tmp] + 1;
                    visited[x] = 1;
                    if(x == n - 1) return dist[n - 1];
                }
            }
            hash.erase(arr[tmp]);
            if(tmp + 1 < n && !visited[tmp + 1])
            {
                que.push(tmp + 1);
                dist[tmp + 1] = dist[tmp] + 1;
                visited[tmp + 1] = 1;
                if(tmp + 1 == n - 1) return dist[n - 1];
            }
            if(tmp - 1 >= 0 && !visited[tmp - 1])
            {
                que.push(tmp - 1);
                dist[tmp - 1] = dist[tmp] + 1;
                visited[tmp - 1] = 1;
            }
        }
        return dist[n - 1];
    }
};
```

***

### LeetCode 2390

##### 日期

2022.08.29

##### 题目描述

![image-20220829191905645](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220829191905645.png)

##### 测试用例

![image-20220829191924805](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220829191924805.png)

##### 思路

- 用栈来模拟
  - 遇到 '*' 从res中弹出末尾元素
  - 其余元素加到末尾

##### 代码

```cpp
class Solution {
public:
    string removeStars(string s) {
        string res;
        for(auto& ch : s)
        {
            if(ch == '*') res.pop_back();
            else res.push_back(ch);
        }
        return res;
    }
};
```

***

### LeetCode 793

##### 日期

2022.08.29

##### 题目描述

![image-20220829192238334](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220829192238334.png)

##### 测试用例

![image-20220829192257931](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220829192257931.png)

##### 思路

- 求阶乘后有几个0
  - 看贡献了几个5；
  - $\operatorname{zeta}(x)=\sum_{k=1}^{\infty}\left\lfloor\frac{x}{5^{k}}\right\rfloor$
  
  - 代码实现
  
    ```cpp
    int zeta(long x) {
            int res = 0;
            while (x) {
                res += x / 5;
                x /= 5;
            }
            return res;
        }
    ```

- 找阶乘末尾0个数为k的数

  - 如果存在，有5个；不存在即为0；

  - $zeta(x)$具有单调性，用二分查找；
  - 左边界$left = 0$，右边界$right = 5k$，因为$zeta(x) >= \frac{x}{5}$,$zeta(5x) >= x$,

  - 或者找末尾0个数大于等于k的数字，返回$n_{k+1} - n_{k}$；

##### 代码

```cpp
typedef long long LL;

class Solution {
    int getZeroNums(LL n)
    {
        int res = 0;
        while(n)
        {
            res += n / 5;
            n /= 5;
        }
        return res;
    }
public:
    int preimageSizeFZF(int k) {
        LL left = 0, right = 5LL * k;
        while(left <= right)
        {
            LL mid = (left + right) >> 1;
            if(getZeroNums(mid) < k) left = mid + 1;
            else if(getZeroNums(mid) == k) return 5;
            else right = mid - 1;
        }
        return 0;
    }
};
```

***

### LeetCode 2400

##### 日期

2022.09.04

##### 题目描述

![image-20220904173524548](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220904173524548.png)

##### 测试用例

![image-20220904173459037](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220904173459037.png)

##### 思路

- 数学解法
  - 正向移动a步，逆向移动b步，$a - b = d, a + b = k$
  - 求组合数$C^{r}_{k}$，其中$r = \frac{d + k}{2}$；

- 组合数
  - 计算公式：$C^{m}_{n} = \frac{n!}{m!(n - m)!}$;
  - 递推公式：$C^{m}_{n} = C^{m - 1}_{n - 1} + C^{m - 1}_{n}$;

- 组合数
  - 快速幂求m^k mod p，时间复杂度O(log k)；
  - 费马小定理求逆元：p为质数，则$a^{p - 1} = 1 (mod\ p)$,即$b * b^{p - 2} = 1(mod\ p)$



##### 代码

```cpp
typedef long long LL;
const int MOD = 1e9 + 7;

class Solution {
private:
    int qmi(int m, int k, int p)
    {
        int res = 1 % p, t = m;
        while(k)
        {
            if(k & 1) res = (LL)res * t % p;
            t = (LL)t * t % p;
            k >>= 1;
        }
        return res;
    }
public:
    int numberOfWays(int startPos, int endPos, int k) {
        int d = abs(startPos - endPos);
        if((k + d) % 2 || k < d) return 0;
        int r = (k + d) / 2;
        LL res = 1;
        for(int i = k; i > k - r; --i)
            res = res * i % MOD;
        for(int i = 1; i <= r; ++i)
            res = res * qmi(i, MOD - 2, MOD) % MOD;
        return (int)res;
    }
};
```

***

### LeetCode 1371

##### 日期

2022.09.05

##### 题目描述

![image-20220905134425248](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220905134425248.png)

##### 测试用例

![image-20220905134439341](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220905134439341.png)

##### 思路

- 前缀和
  - 维护a e i o u的奇偶状态；
- 状态压缩
  - 5位2进制数表示状态，0表示偶数，1表示奇数；
  - 开一个长度为32的数组表示当前状态出现的最早位置；

- 偶数次子字符串
  - 前$i$的状态与前j状态相同，则说明$j + 1$到$i$的子字符串是偶数次

##### 代码

```cpp
class Solution {
public:
    int findTheLongestSubstring(string s) {
        vector<int> hash(32, INT_MAX);
        hash[0] = -1;
        int n = s.size();
        string str = "aeiou";
        int state = 0, res = 0;
        for(int i = 0; i < n; ++i)
        {
            int k = str.find(s[i]);
            if(k != -1) state ^= (1 << k);
            if(hash[state] < INT_MAX) res = max(res, i - hash[state]);
            else hash[state] = i;
        }
        return res;
    }
};
```

***

### LeetCode 652

##### 日期

2022.09.05

##### 题目描述

![image-20220906004606744](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220906004606744.png)

##### 测试用例

![image-20220906004629284](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220906004629284.png)

##### 思路

- 方法一：哈希表
  - 序列化树，前序遍历或后序遍历，空节点标记为#;
  - 递归，返回值为以当前节点为根节点的子树的序列化字符串
  - 哈希表存入2个时收集当前节点，防止重复push；

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
private:
    unordered_map<string, int> hash;
    vector<TreeNode*> res;
    string DFS(TreeNode* node)
    {
        if(node == nullptr) return "#";     // return "";也对，相当于对空节点标记空子字符串
        string str = to_string(node->val) + " " + DFS(node->left) + " " + DFS(node->right);
        if(++hash[str] == 2) res.push_back(node);
        return str;
    }
public:
    vector<TreeNode*> findDuplicateSubtrees(TreeNode* root) {
        DFS(root);
        return res;
    }
};
```

- 方法二：树 <-------->数建立一一映射（防止字符串哈希过不了）
  - 对每个子树对应的字符串建立一个唯一的id；以id为键值建立哈希表
  - 递归返回值为以当前节点为根节点的子树的id；
  - 哈希表存入2个时收集当前节点，防止重复push；

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
private:
    unordered_map<string, int> ids;
    unordered_map<int, int> hash;
    int cnt = 0;
    vector<TreeNode*> res;
    int DFS(TreeNode* node)
    {
        if(node == nullptr) return 0;
        int l_id = DFS(node->left);
        int r_id = DFS(node->right);
        string str = to_string(node->val) + " " + to_string(l_id) + " " + to_string(r_id);
        if(ids.find(str) == ids.end())
            ids[str] = ++cnt;
        if(++hash[ids[str]] == 2) res.push_back(node);
        return ids[str];
    }
public:
    vector<TreeNode*> findDuplicateSubtrees(TreeNode* root) {
        DFS(root);
        return res;
    }
};
```

### LeetCode 850

##### 题目描述

![image-20220916231839531](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220916231839531.png)

##### 测试用例

![image-20220916232015479](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220916232015479.png)

##### 思路

- 扫描线
  - 将二维问题转化为一维

- 区间合并
  - 先排序
  - 维护一段区间长度，根据区间末尾与下一个区间首端点的关系更新区间；

##### 代码

```cpp
typedef long long LL;
typedef pair<int, int> PII;

class Solution {
public:
    LL merge(vector<vector<int>>& rectangles, int a, int b)
    {
        vector<PII> segs;
        for(auto& arr : rectangles)
            if(arr[0] <= a && arr[2] >= b)
                segs.emplace_back(arr[1], arr[3]);
        sort(segs.begin(), segs.end());
        int st = -1, ed =-1;
        LL res = 0;
        for(auto& seg : segs)
        {
            if(ed < seg.first)
            {
                res += ed - st;
                st = seg.first, ed = seg.second;
            }
            else ed = max(ed, seg.second);
        }
        res += ed - st;
        return res;
    }
    int rectangleArea(vector<vector<int>>& rectangles) {
        vector<int> lines;
        for(auto& arr : rectangles)
        {
            lines.push_back(arr[0]);
            lines.push_back(arr[2]);
        }
        sort(lines.begin(), lines.end());
        LL res = 0;
        for(int i = 0; i + 1 < lines.size(); ++i)
            res += merge(rectangles, lines[i], lines[i + 1]) * (lines[i + 1] - lines[i]);
        return res % 1000000007;
    }
};
```

### LeetCode 698

##### 日期

2022.09.20

##### 题目描述

![image-20220921000202698](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220921000202698.png)

 ##### 测试用例

![image-20220921000332396](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220921000332396.png)

##### 思路

- 回溯+剪枝
  - 先从大到小排序
  - 递归处理每个子集，dfs()求当前位置开始搜索，当前子集和，需搜索几组子集，是否成立
- 剪枝操作（4步）
  - 从大到小枚举；
  - 当前搜的数和前一个数相等且失败，一定失败；
  - 当前数是某一组第一个数且失败，后面一定失败；
  - 当前数是某一组最后一个数且失败，后面一定失败；

- 代码

```cpp
class Solution {
private:
    vector<int> nums;
    vector<bool> visited;
    int m;
    bool dfs(int start, int cur, int k)
    {
        if(!k) return true;
        if(cur == m) return dfs(0, 0, k - 1);
        for(int i = start; i < nums.size(); ++i)
        {
            if(visited[i]) continue;
            if(cur + nums[i] <= m)
            {
                visited[i] = true;
                if(dfs(i + 1, cur + nums[i], k)) return true;
                visited[i] = false;
            }
            while(i + 1 < nums.size() && nums[i + 1] == nums[i]) ++i;
            if(cur == 0) return false;
        }
        return false;
    }
public:
    bool canPartitionKSubsets(vector<int>& _nums, int k) {
        nums = _nums;
        visited = vector<bool>(nums.size(), false);
        int sum = 0;
        for(auto& x : nums) sum += x;
        if(sum % k) return false;
        m = sum / k;
        int n = nums.size();
        sort(nums.begin(), nums.end(), greater<int>());
        return dfs(0, 0, k);
    }
};
```

### LeetCode 854

##### 日期

2022.09.21

##### 题目描述

![image-20220921220835275](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220921220835275.png)

##### 测试样例

![image-20220921220904460](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220921220904460.png)

##### 思路

- BFS+剪枝；暴搜每种交换的方案
  - BFS，找每一个可以交换的字符交换，放进队列
  - 剪枝：
    - 哈希表去重
    - 位置已经对的字符跳过处理

##### 代码

```cpp
class Solution {
public:
    int kSimilarity(string s1, string s2) {
        int n = s1.size();
        queue<pair<string, int>> que;
        unordered_set<string> hash;
        que.emplace(s1, 0);
        hash.insert(s1);
        int res = 0;
        while(!que.empty())
        {
            int size = que.size();
            for(int i = 0; i < size; ++i)
            {
                // auto& [str, index] = que.front();        不能写成引用，弹出后队列front已经改变，可能为空
                auto [str, index] = que.front();
                que.pop();
                if(str == s2) return res;
                while(index < n && str[index] == s2[index]) ++index;
                for(int j = index + 1; j < n; ++j)
                {
                    if(str[j] != s2[j] && str[j] == s2[index])     // 剪枝
                    {
                        swap(str[index], str[j]);
                        if(!hash.count(str))            // 剪枝
                        {
                            que.emplace(str, index + 1);
                            hash.insert(str);
                        }
                        swap(str[index], str[j]);
                    }
                }
            }
            ++res;
        }
        return res;
    }
};
```

### LeetCode 面试题 17.19

##### 日期

2022.09.27

##### 题目描述

![image-20220927132330394](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220927132330394.png)

##### 测试用例

![image-20220927132411229](C:\Users\liyunzhe\AppData\Roaming\Typora\typora-user-images\image-20220927132411229.png)

##### 思路

- 设消失的两个数字是$a$和$b$，再补上1~N；
- 所有数异或起来，得到a^b，此时不能把a和b区分开；
- 由x & -x求最低位是1的位lsb；则a的lsb是1，b的lsb是0，即将a和b分在两类中；
- 每一类分别异或，即可分别得到a和b；

##### 代码

```cpp
class Solution {
public:
    vector<int> missingTwo(vector<int>& nums) {
        int N = nums.size() + 2;
        int OR = 0;
        for(int i = 1; i <= N; ++i) nums.push_back(i);
        for(auto i : nums) OR ^= i;
        int low = OR & (-OR);
        int a = 0, b = 0;
        for(auto i : nums)
        {
            if(low & i) a ^= i;
            else b ^= i;
        }
        return {a, b};
    }
};
```



