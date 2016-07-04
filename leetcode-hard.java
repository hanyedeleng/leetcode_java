296	Best Meeting Point
A group of two or more people wants to meet and minimize the total travel distance. 
You are given a 2D grid of values 0 or 1, where each 1 marks the home of someone in the group. 
The distance is calculated using Manhattan Distance, where distance(p1, p2) = |p2.x - p1.x| + |p2.y - p1.y|.
For example, given three people living at (0,0), (0,4), and (2,2):
1 - 0 - 0 - 0 - 1
|   |   |   |   |
0 - 0 - 0 - 0 - 0
|   |   |   |   |
0 - 0 - 1 - 0 - 0
The point (0,2) is an ideal meeting point, as the total travel
distance of 2+2+2=6 is minimal. So return 6.
//二维的等于一维的相加, 一维的最小点必在median点(用反证法可以证明).
public class Solution {
    public int minTotalDistance(int[][] grid) {
        List<Integer> ipos = new ArrayList<Integer>();
        List<Integer> jpos = new ArrayList<Integer>();
        // 统计出有哪些横纵坐标
        for(int i = 0; i < grid.length; i++){
            for(int j = 0; j < grid[0].length; j++){
                if(grid[i][j] == 1){
                    ipos.add(i);
                    jpos.add(j);
                }
            }
        }
        int sum = 0;
        // 计算纵坐标到纵坐标中点的距离，这里不需要排序，因为之前统计时是按照i的顺序
        for(Integer pos : ipos){
            sum += Math.abs(pos - ipos.get(ipos.size() / 2));
        }
        // 计算横坐标到横坐标中点的距离，这里需要排序，因为统计不是按照j的顺序
        Collections.sort(jpos);
        for(Integer pos : jpos){
            sum += Math.abs(pos - jpos.get(jpos.size() / 2));
        }
        return sum;
    }
}

52	N-Queens II
Follow up for N-Queens problem.
Now, instead outputting board configurations, return the total number of distinct solutions.
public class Solution {
    public List<List<String>> solveNQueens(int n) {
        boolean[] 
            //ocp0 = new boolean[n], //whether there's a queen ocupying nth row, I don't need it
        	// 表示斜线，一共有2*n－1条斜线，45 左低右高，从左上到右下编号0-6，135，左高右低从左下到右上编号0-6
            ocp90 = new boolean[n], //whether there's a queen ocupying nth column
            ocp45 = new boolean[2 * n - 1], // mark 45 degree occupation
            ocp135 = new boolean[2 * n - 1]; // mark 135 degree occupation
        List<List<String>> ans = new ArrayList<List<String>>();
        char[][] map = new char[n][n];
        for (char[] tmp : map) Arrays.fill(tmp, '.'); //init

        solve(0, n, map, ans, ocp45, ocp90, ocp135);
        return ans;
    }

    private void solve(int depth, int n, char[][] map, List<List<String>> ans, 
    boolean[] ocp45, boolean[] ocp90, boolean[] ocp135) {
        if (depth == n) {
            addSolution(ans, map);
            return;
        }

        for (int j = 0; j < n; j++)
            if (!ocp90[j] && !ocp45[depth + j] && !ocp135[j - depth + n - 1]) {
                ocp90[j] = true;
                ocp45[depth + j] = true;
                ocp135[j - depth + n - 1] = true;
                map[depth][j] = 'Q';
                solve(depth + 1, n, map, ans, ocp45, ocp90, ocp135);
                ocp90[j] = false;
                ocp45[depth + j] = false;
                ocp135[j - depth + n - 1] = false;
                map[depth][j] = '.';
            }
    }

    private void addSolution(List<List<String>> ans, char[][] map) {
        List<String> cur = new ArrayList<String>();
        for (char[] i : map) cur.add(String.valueOf(i));
        ans.add(cur);
    }
}


count n queen number:
/*
    常规n-queens解法, 数答案个数.
    用column标记此行之前的哪些column已经放置了queen. 棋盘坐标(row, col)对应column的第col位(LSB --> MSB, 下同).
    用diag标记此位置之前的哪些主对角线已经放置了queen. 棋盘坐标(row, col)对应diag的第(n - 1 + row - col)位.
    用antiDiag标记此位置之前的哪些副对角线已经放置了queen. 棋盘坐标(row, col)对应antiDiag的第(row + col)位.
*/
public class Solution {
    int count = 0;

    public int totalNQueens(int n) {
        dfs(0, n, 0, 0, 0);
        return count;
    }

    private void dfs(int row, int n, int column, int diag, int antiDiag) {
        if (row == n) {
            ++count;
            return;
        }
        for (int i = 0; i < n; ++i) {
            boolean isColSafe = ((1 << i) & column) == 0;
            boolean isDiagSafe = ((1 << (n - 1 + row - i)) & diag) == 0;
            boolean isAntiDiagSafe = ((1 << (row + i)) & antiDiag) == 0;
            if (isColSafe && isDiagSafe && isAntiDiagSafe) {
                dfs(row + 1, n, (1 << i) | column, (1 << (n - 1 + row - i)) | diag, (1 << (row + i)) | antiDiag);
            }
        }
    }
}

302	Smallest Rectangle Enclosing Black Pixels
An image is represented by a binary matrix with 0 as a white pixel and 1 as a black pixel.
The black pixels are connected, i.e., there is only one black region. Pixels are connected
horizontally and vertically. Given the location (x, y) of one of the black pixels, return
the area of the smallest (axis-aligned) rectangle that encloses all black pixels.

For example, given the following image:

[
  "0010",
  "0110",
  "0100"
]
and x = 0, y = 2,
Return 6.

Time Complexity - O(mlogn + nlogm)， Space Complexity - O(1)

public class Solution {
    public int minArea(char[][] image, int x, int y) {
        if(image == null || image.length == 0) {
            return 0;
        }
        int rowNum = image.length, colNum = image[0].length;
        int left = binarySearch(image, 0, y, 0, rowNum, true, true);
        int right = binarySearch(image, y + 1, colNum, 0, rowNum, true, false);
        int top = binarySearch(image, 0, x, left, right, false, true);
        int bot = binarySearch(image, x + 1, rowNum, left, right, false, false);
        
        return (right - left) * (bot - top);
    }
    
    private int binarySearch(char[][] image, int lo, int hi, int min, int max, boolean searchHorizontal, 
    	boolean searchLo) {
        while(lo < hi) {
            int mid = lo + (hi - lo) / 2;
            boolean hasBlackPixel = false;
            for(int i = min; i < max; i++) {
                if((searchHorizontal ? image[i][mid] : image[mid][i]) == '1') {
                    hasBlackPixel = true;
                    break;
                }
            }
            if(hasBlackPixel == searchLo) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }
}

287	Find the Duplicate Number
Given an array nums containing n + 1 integers where each integer is between 1 and n (inclusive), prove that at least one duplicate number must exist. Assume that there is only one duplicate number, find the duplicate one.

Note:
You must not modify the array (assume the array is read only).
You must use only constant, O(1) extra space.
Your runtime complexity should be less than O(n2).
There is only one duplicate number in the array, but it could be repeated more than once.!!!!!!!

Java O(n) time and O(1) space solution. Similar to find loop in linkedlist.
public int findDuplicate(int[] nums) {
    int slow = 0, fast = 0;
    do{
        slow = nums[slow];
        fast = nums[nums[fast]];
    }while(slow != fast);
    slow = 0;
    while(slow != fast){
        slow = nums[slow];
        fast = nums[fast];
    }
    return slow;
}

"******************************************************"
340	Longest Substring with At Most K Distinct Characters 
Given a string, find the length of the longest substring T that contains at most k 
distinct characters.

For example, Given s = “eceba” and k = 2,
T is "ece" which its length is 3.

public int lengthOfLongestSubstringKDistinct(String s, int k) {
    int[] count = new int[256];
    int num = 0, i = 0, res = 0;
    for (int j = 0; j < s.length(); j++) {
        if (count[s.charAt(j)]++ == 0) num++;
        if (num > k) { 
            while (--count[s.charAt(i++)] > 0);//??
            num--;
        }
        res = Math.max(res, j - i + 1);
    }
    return res;
}

public int lengthOfLongestSubstringKDistinct(String s, int k) {
    Map<Character, Integer> map = new HashMap<>();
    int left = 0;
    int best = 0;
    for(int i = 0; i < s.length(); i++) {
        // character at the right pointer
        char c = s.charAt(i);
        map.put(c, map.getOrDefault(c, 0) + 1);
        // make sure map size is valid, no need to check left pointer less than s.length()
        while (map.size() > k) {
            char leftChar = s.charAt(left);
            if (map.containsKey(leftChar)) {
                map.put(leftChar, map.get(leftChar) - 1);                     
                if (map.get(leftChar) == 0) { 
                    map.remove(leftChar);
                }
            }
            left++;
        }
        best = Math.max(best, i - left + 1);
    }
    return best;
}


312	Burst Balloons
Given n balloons, indexed from 0 to n-1. Each balloon is painted with a number on it represented by array
nums. You are asked to burst all the balloons. If the you burst balloon i you will get 
nums[left] * nums[i] * nums[right] coins. Here left and right are adjacent indices of i. After the burst, 
the left and right then becomes adjacent.
Find the maximum coins you can collect by bursting the balloons wisely.

Note: 
(1) You may imagine nums[-1] = nums[n] = 1. They are not real therefore you can not burst them.
(2) 0 ≤ n ≤ 500, 0 ≤ nums[i] ≤ 100

Example:
Given [3, 1, 5, 8]
Return 167

    nums = [3,1,5,8] --> [3,5,8] -->   [3,8]   -->  [8]  --> []
   coins =  3*1*5      +  3*5*8    +  1*3*8      + 1*8*1   = 167

public int maxCoins(int[] iNums) {
    int[] nums = new int[iNums.length + 2];
    int n = 1;
    for (int x : iNums) if (x > 0) nums[n++] = x;
    nums[0] = nums[n++] = 1;


    int[][] memo = new int[n][n];
    return burst(memo, nums, 0, n - 1);
}

public int burst(int[][] memo, int[] nums, int left, int right) {
    if (left + 1 == right) return 0;
    if (memo[left][right] > 0) return memo[left][right];
    int ans = 0;
    for (int i = left + 1; i < right; ++i)
        ans = Math.max(ans, nums[left] * nums[i] * nums[right] 
        + burst(memo, nums, left, i) + burst(memo, nums, i, right));
    memo[left][right] = ans;
    return ans;
}

Dp[i][j] 表示打破的气球介于i和 j 之间得到的最大硬币数。显然我们要求的是Dp[0][n-1].

145	Binary Tree Postorder Traversal
Given a binary tree, return the postorder traversal of its nodes' values.'

For example:
Given binary tree {1,#,2,3},
   1
    \
     2
    /
   3
return [3,2,1].

public class Solution {
    public List<Integer> postorderTraversal(TreeNode root) {
        ArrayList<Integer> list = new ArrayList<>();
        if (root == null) return list;
        
        helper(list, root);
        return list;
    }
    
    public void helper(List<Integer> list, TreeNode root) {
        if (root == null) return;
        helper(list, root.left);
        helper(list, root.right);
        list.add(root.val);
    }
}

265	Paint House II 
There are a row of n houses, each house can be painted with one of the k colors. The
cost of painting each house with a certain color is different. You have to paint all
the houses such that no two adjacent houses have the same color.

The cost of painting each house with a certain color is represented by a n x k cost matrix.
For example, costs[0][0] is the cost of painting house 0 with color 0; costs[1][2] is the
cost of painting house 1 with color 2, and so on... Find the minimum cost to paint all houses.

Note:
All costs are positive integers.
Follow up:
Could you solve it in O(nk) runtime?

public class Solution {
    public int minCostII(int[][] costs) {
        if (costs == null)
            throw new IllegalArgumentException("costs is null");
        if (costs.length == 0)
            return 0;
        int len = costs.length;
        int k = costs[0].length;
        int min_1 = 0, min_2 = 0;
        int pre_min_1, pre_min_2;
        int[] dp = new int[k];
        for (int i = 0; i < len; i++) {
            pre_min_1 = min_1;
            pre_min_2 = min_2;
            min_1 = Integer.MAX_VALUE;
            min_2 = Integer.MAX_VALUE;
            for (int j = 0; j < k; j++) {
                if (dp[j] != pre_min_1 || pre_min_1 == pre_min_2) {
                    dp[j] = pre_min_1 + costs[i][j];
                } else{
                    dp[j] = pre_min_2 + costs[i][j];
                }
                if (dp[j] <= min_1) {
                    min_2 = min_1;
                    min_1 = dp[j];
                } else if (dp[j] < min_2){
                    min_2 = dp[j];
                }
            }
        }
        return min_1;
    }
}


159	Longest Substring with At Most Two Distinct Characters
Given a string S, find the length of the longest substring T that contains at most two distinct characters.
For example,
Given S = “eceba”,
T is “ece” which its length is 3.
public int lengthOfLongestSubstringTwoDistinct(String s) {  
    int left = 0, second = -1;  
    int n = s.length();  
    int len = 0;  
    for(int i=1; i < n; i++) {  
        if(s.charAt(i) == s.charAt(i-1)) continue;  
        if(second >= 0 && s.charAt(i) != s.charAt(second)) {  
            len = Math.max(len, i-left);  
            left = second+1;  
        }  
        second = i-1;  
    }  
    return Math.max(len, n-left);  
}  
最优的解法应该是维护一个sliding window，指针变量i指向sliding window的起始位置，j指向另个一个字符在sliding window的最后一个，
用于定位i的下一个跳转位置。内部逻辑就是
1）如果当前字符跟前一个字符是一样的，直接继续。
2）如果不一样，则需要判断当前字符跟j是不是一样的
a）一样的话sliding window左边不变，右边继续增加，但是j的位置需要调整到k-1。
b）不一样的话，sliding window的左侧变为j的下一个字符（也就是去掉包含j指向的字符的区间），j的位置也需要调整到k-1。

在对i进行调整的时候（1.a），需要更新maxLen。

[注意事项]
1）在最后返回的时候，注意考虑s.length()-i这种情况，也就是字符串读取到最后而没有触发（1.a）
2）讲解清楚sliding window的更新
3）该题目有个follow-up，就是如果是k个distinct characters怎么办。这样的话就只能对所有可能的字符用一个数组去做counting，
而且只能假设ASIC字符集256。Unicode太大了

352	Data Stream as Disjoint Intervals
Given a data stream input of non-negative integers a1, a2, ..., an, ..., summarize the numbers seen so far 
as a list of disjoint intervals.

For example, suppose the integers from the data stream are 1, 3, 7, 2, 6, ..., then the summary will be:
[1, 1]
[1, 1], [3, 3]
[1, 1], [3, 3], [7, 7]
[1, 3], [7, 7]
[1, 3], [6, 7]
Follow up:
What if there are lots of merges and the number of disjoint intervals are small compared to the data 
stream's size?'

Use TreeMap to easily find the lower and higher keys, the key is the start of the interval. Merge the 
lower and higher intervals when necessary. The time complexity for adding is O(logN) since lowerKey(), 
higherKey(), put() and remove() are all O(logN). It would be O(N) if you use an ArrayList and remove an 
interval from it.

public class SummaryRanges {
    TreeMap<Integer, Interval> tree;

    public SummaryRanges() {
        tree = new TreeMap<>();
    }

    public void addNum(int val) {
        if(tree.containsKey(val)) return;
        Integer l = tree.lowerKey(val);
        Integer h = tree.higherKey(val);
        if(l != null && h != null && tree.get(l).end + 1 == val && h == val + 1) {
            tree.get(l).end = tree.get(h).end;
            tree.remove(h);
        } else if(l != null && tree.get(l).end + 1 >= val) {
            tree.get(l).end = Math.max(tree.get(l).end, val);
        } else if(h != null && h == val + 1) {
            tree.put(val, new Interval(val, tree.get(h).end));
            tree.remove(h);
        } else {
            tree.put(val, new Interval(val, val));
        }
    }

    public List<Interval> getIntervals() {
        return new ArrayList<>(tree.values());
    }
}


291 Word Pattern II
Given a pattern and a string str, find if str follows the same pattern.
Here follow means a full match, such that there is a bijection between a letter in pattern and a non-empty 
substring in str.

Examples:
pattern = "abab", str = "redblueredblue" should return true.
pattern = "aaaa", str = "asdasdasdasd" should return true.
pattern = "aabb", str = "xyzabcxzyabc" should return false. 
Notes:
You may assume both pattern and str contains only lowercase letters.

A typical backtracking. I use two hashmap to guarantee one pattern only map to exact one string. Note we 
need to remove new added element in hashmap if current splitted string is illegal

public class Solution {
Map<Character,String> map =new HashMap();
Set<String> set =new HashSet();
public boolean wordPatternMatch(String pattern, String str) {
    if(pattern.isEmpty()) return str.isEmpty();
    if(map.containsKey(pattern.charAt(0))){
        String value= map.get(pattern.charAt(0));
        if(str.length()<value.length() || !str.substring(0,value.length()).equals(value)) return false;
        if(wordPatternMatch(pattern.substring(1),str.substring(value.length()))) return true;
    }else{
        for(int i=1;i<=str.length();i++){
            if(set.contains(str.substring(0,i))) continue;
            map.put(pattern.charAt(0),str.substring(0,i));
            set.add(str.substring(0,i));
            if(wordPatternMatch(pattern.substring(1),str.substring(i))) return true;
            set.remove(str.substring(0,i));
            map.remove(pattern.charAt(0));
        }
    }
    return false;
}


public boolean wordPatternMatch(String pattern, String str) {
        HashMap map = new HashMap();
        return dfs(pattern, 0, str, 0, map);
    }
    private boolean dfs(String pattern, int i, String str, int j, HashMap map){
        if(i == pattern.length() && j == str.length()){// 如果刚好搜完. 返回true
            return true;
        }
        if(i == pattern.length() || j == str.length()){// 如果一个完了, 另一个没完, 返回false
            return false;
        }
        char c = pattern.charAt(i);
        for(int k = j; k < str.length(); k++){
            if(map.get(c) == map.get(str.substring(j, k+1))){//如果map中的i对应的值(可以是null) 和 sbustring对应的值相同(也可以是null)
                Integer val = (Integer)map.get(c);
                if(val == null){//如果是null
                    map.put(pattern.charAt(i), i);//把pattern的<char,integer>放map中
                    map.put(str.substring(j, k+1), i);//把string的<string,integer>放map中
                }
                if(dfs(pattern, i+1, str, k+1, map)){//dfs
                    return true;
                }
                if(val == null){// backtracking
                    map.remove(pattern.charAt(i));
                    map.remove(str.substring(j, k+1));
                }
            }
        }
        return false;
    }

305 Number of Islands II 
A 2d grid map of m rows and n columns is initially filled with water. We may perform an addLand operation 
which turns the water at position (row, col) into a land. Given a list of positions to operate, count the 
number of islands after each addLand operation. An island is surrounded by water and is formed by 
connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all 
surrounded by water.

Example:

Given m = 3, n = 3, positions = [[0,0], [0,1], [1,2], [2,1]].
Initially, the 2d grid grid is filled with water. (Assume 0 represents water and 1 represents land).
0 0 0
0 0 0
0 0 0
Operation #1: addLand(0, 0) turns the water at grid[0][0] into a land.
1 0 0
0 0 0   Number of islands = 1
0 0 0
Operation #2: addLand(0, 1) turns the water at grid[0][1] into a land.
1 1 0
0 0 0   Number of islands = 1
0 0 0
Operation #3: addLand(1, 2) turns the water at grid[1][2] into a land.
1 1 0
0 0 1   Number of islands = 2
0 0 0
Operation #4: addLand(2, 1) turns the water at grid[2][1] into a land.
1 1 0
0 0 1   Number of islands = 3
0 1 0
We return the result as an array: [1, 1, 2, 3]

Challenge:
Can you do it in time complexity O(k log mn), where k is the length of the positions?


public class Solution {
    public List<Integer> numIslands2(int m, int n, int[][] positions) {
        int[] id = new int[m * n]; // 表示各个index对应的root
        
        List<Integer> res = new ArrayList<>();
        Arrays.fill(id, -1); // 初始化root为-1，用来标记water, 非-1表示land
        int count = 0; // 记录island的数量
        
        int[][] dirs = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
        for (int i = 0; i < positions.length; i++) {
            count++;
            int index = positions[i][0] * n + positions[i][1];           
            id[index] = index; // root初始化
            
            for (int j = 0; j < dirs.length; j++) {
                int x = positions[i][0] + dirs[j][0];
                int y = positions[i][1] + dirs[j][1];
                if (x >= 0 && x < m && y >= 0 && y < n && id[x * n + y] != -1) {
                    int root = root(id, x * n + y);

                    // 发现root不等的情况下，才union, 同时减小count
                    if (root != index) {
                        id[root] = index;
                        count--;
                    }
                }
            }
            res.add(count);
        }
        return res;
    }
    
    public int root(int[] id, int i) {
        while (i != id[i]) {
            id[i] = id[id[i]]; // 优化，为了减小树的高度                
            i = id[i];
        }
        return i;
    }
}


154 Find Minimum in Rotated Sorted Array II
Suppose a sorted array is rotated at some pivot unknown to you beforehand.
(i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
Find the minimum element.

The array may contain duplicates.

public class Solution {
    public int findMin(int[] nums) {
        if (nums.length == 0) return 0;
        
        int lo = 0;
        int hi = nums.length - 1;
        while (lo < hi) {
            int m = lo + (hi-lo)/2;
            if (nums[m] > nums[hi]) {
                lo = m + 1;
            }else if (nums[m] < nums[hi]) {
                hi = m;
            }else {
                hi--;
            }
        }
        return nums[lo];
    }
}
When num[mid] == num[hi], we couldn't sure the position of minimum in mid's left or right, 
so just let upper bound reduce one.

272 Closest Binary Search Tree Value II
Given a non-empty binary search tree and a target value, find k values in the BST that are closest to the
target.
Note:

Given target value is a floating point.
You may assume k is always valid, that is: k ≤ total nodes.
You are guaranteed to have only one unique set of k values in the BST that are closest to the target.

一开始思路非常不明确，看了不少discuss也不明白为什么。在午饭时间从头仔细想了一下，像Closest Binary Search Tree Value I一样，
追求O(logn)的解法可能比较困难，但O(n)的解法应该不难实现。我们可以使用in-order的原理，从最左边的元素开始，维护一个Deque或者
doubly linked list，将这个元素的值从后端加入到Deque中，然后继续遍历下一个元素。当Deque的大小为k时， 比较当前元素和队首元素
与target的差来尝试更新deque。循环结束条件是队首元素与target的差更小或者遍历完全部元素。这样的话时间复杂度是O(n)， 
空间复杂度应该是O(k)。

public class Solution {
    public List<Integer> closestKValues(TreeNode root, double target, int k) {
        LinkedList<Integer> res = new LinkedList<>();
        inOrder(root, target, k, res);
        return res;
    }
    
    private void inOrder(TreeNode root, double target, int k, LinkedList<Integer> res) {
        if(root == null) {
            return;
        }
        inOrder(root.left, target, k, res);
        if(res.size() == k) {
            if(Math.abs(res.get(0) - target) >= Math.abs(root.val - target)) {
                res.removeFirst();
                res.add(root.val);
            } else {
                return;
            }
        } else {
            res.add(root.val);
        }
        inOrder(root.right, target, k, res);
    }
}


117 Populating Next Right Pointers in Each Node II
Follow up for problem "Populating Next Right Pointers in Each Node".

What if the given tree could be any binary tree? Would your previous solution still work?

Note:

You may only use constant extra space.
For example,
Given the following binary tree,
         1
       /  \
      2    3
     / \    \
    4   5    7
After calling your function, the tree should look like:
         1 -> NULL
       /  \
      2 -> 3 -> NULL
     / \    \
    4-> 5 -> 7 -> NULL

public class Solution {
    public void connect(TreeLinkNode root) {
        if (root == null) {
            return;
        }
        
        Queue<TreeLinkNode> dq = new LinkedList<>();
        dq.add(root);
        while (!dq.isEmpty()) {
            Queue<TreeLinkNode> tmpQ = new LinkedList<>();
            
            while (!dq.isEmpty()) {
                TreeLinkNode tmp = dq.remove();
                if (tmp.left != null) tmpQ.add(tmp.left);
                if (tmp.right != null) tmpQ.add(tmp.right);
                if (!dq.isEmpty()) {
                    tmp.next = dq.peek();
                }else {
                    tmp.next = null;
                }
            }
            dq = tmpQ;
        }
        return;
    }
}
// pay attention to add, remove(linked list method)

public class Solution {
    //based on level order traversal
    public void connect(TreeLinkNode root) {

        TreeLinkNode head = null; //head of the next level
        TreeLinkNode prev = null; //the leading node on the next level
        TreeLinkNode cur = root;  //current node of current level

        while (cur != null) {

            while (cur != null) { //iterate on the current level
                //left child
                if (cur.left != null) {
                    if (prev != null) {
                        prev.next = cur.left;
                    } else {
                        head = cur.left;
                    }
                    prev = cur.left;
                }
                //right child
                if (cur.right != null) {
                    if (prev != null) {
                        prev.next = cur.right;
                    } else {
                        head = cur.right;
                    }
                    prev = cur.right;
                }
                //move to next node
                cur = cur.next;
            }

            //move to next level
            cur = head;
            head = null;
            prev = null;
        }

    }
}

42  Trapping Rain Water
Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how 
much water it is able to trap after raining.

For example, 
Given [0,1,0,2,1,0,1,3,2,1,2,1], return 6.

public class Solution {
    public int trap(int[] height) {
        if (height.length == 0) {
            return 0;
        }
        
        int lo = 0;
        int hi = height.length - 1;
        int sum = 0;
        int plank = 0;
        while (lo < hi) {
            int min = Math.min(height[lo], height[hi]);
            plank = plank < min?min:plank;
            if (height[lo] <= height[hi]) {
                sum += plank-height[lo];
                lo++;
            }else {
                sum += plank-height[hi];
                hi--;
            }
        }
        return sum;
    }
}
Basically this solution runs two pointers from two sides to the middle, and the plank is used to record 
the height of the elevation within a certain range, plank height can only increase (or remain the same) 
from two sides to the middle. If the current pointer is pointing at a number that is less than the current 
plank height, the difference between plank height and the number would be the amount of water trapped. 
Otherwise, A[i] == plank, no water is trapped.}

128 Longest Consecutive Sequence
Given an unsorted array of integers, find the length of the longest consecutive elements sequence.
For example,
Given [100, 4, 200, 1, 3, 2],
The longest consecutive elements sequence is [1, 2, 3, 4]. Return its length: 4.
Your algorithm should run in O(n) complexity.

I have used a hashmap in my solution. The keyset of the map stores the number in the given array. The 
entryset stores the upper/lower end of sequence if the key is an lower/upper end of an existing consecutive
sequence.
For a new number ,we have four conditions 
1) It will be a new lower end.-->Refresh both upper and lower end
2) It will be a new upper end.-->Refresh both upper and lower end 
3) Neither-->It is both upper and lower end by itself-->Add the number to the keyset with the value as 
itself. 
4) Both-->It connects two existing sequence.Its own value is not important-->Refresh both upper and lower 
end.Add the number to the keyset with the value as itself.

public class Solution {
    public int longestConsecutive(int[] num) {
        Map<Integer, Integer> seq = new HashMap<Integer, Integer>();
        int longest = 0;
        for (int i = 0; i < num.length; i++) {
            if (seq.containsKey(num[i])) continue;

            int low= num[i],upp=num[i];

            if (seq.containsKey(num[i] - 1)) // Get the lowerbound if existed
                low = seq.get(num[i] - 1);
            if (seq.containsKey(num[i] + 1)) // Get the upperbound if existed
                upp = seq.get(num[i] + 1);

            longest = Math.max(longest, (upp - low)+ 1);

            seq.put(num[i],num[i]);          //Handle   3 & 4. See Beginning
            seq.put(low, upp);               //Handle 1 2 & 4 
            seq.put(upp, low);               //Handle 1 2 & 4 
        }
        return longest;
    }
}


301 Remove Invalid Parentheses
Remove the minimum number of invalid parentheses in order to make the input string valid. 
Return all possible results.
Note: The input string may contain letters other than the parentheses ( and ).

Examples:
"()())()" -> ["()()()", "(())()"]
"(a)())()" -> ["(a)()()", "(a())()"]
")(" -> [""]

public class Solution {
    public List<String> removeInvalidParentheses(String s) {
        List<String> ans = new ArrayList<>();
        remove(s, ans, 0, 0, new char[]{'(', ')'});
        return ans;
    }
    
    public void remove(String s, List<String> ans, int last_i, int last_j,  char[] par) {
        for (int stack = 0, i = last_i; i < s.length(); ++i) {
            if (s.charAt(i) == par[0]) stack++;
            if (s.charAt(i) == par[1]) stack--;
            if (stack >= 0) continue;
            for (int j = last_j; j <= i; ++j)
                if (s.charAt(j) == par[1] && (j == last_j || s.charAt(j - 1) != par[1]))
                    remove(s.substring(0, j) + s.substring(j + 1, s.length()), ans, i, j, par);
            return;
        }
        String reversed = new StringBuilder(s).reverse().toString();
        if (par[0] == '(') // finished left to right
            remove(reversed, ans, 0, 0, new char[]{')', '('});
        else // finished right to left
            ans.add(reversed);
    }
}


329 Longest Increasing Path in a Matrix
Given an integer matrix, find the length of the longest increasing path.
From each cell, you can either move to four directions: left, right, up or down. You may NOT move 
diagonally or move outside of the boundary (i.e. wrap-around is not allowed).}

Example 1:
nums = [
  [9,9,4],
  [6,6,8],
  [2,1,1]
]
Return 4
The longest increasing path is [1, 2, 6, 9].

Example 2:

nums = [
  [3,4,5],
  [3,2,6],
  [2,2,1]
]
Return 4
The longest increasing path is [3, 4, 5, 6]. Moving diagonally is not allowed.

public class Solution {
    int[][] dis = {{1,0},{-1,0},{0,1},{0,-1}};
    public int longestIncreasingPath(int[][] matrix) {
      if(matrix.length == 0 ){
            return 0;
      }
      int[][] state = new int[matrix.length][matrix[0].length];
      int res = 0;
      for(int i = 0; i < matrix.length; i++){
          for(int j = 0; j < matrix[0].length; j++){
             res = Math.max(res,dfs(i,j,matrix,state));
          }
      }
      return res;
    }
      public int dfs(int i, int j, int[][] matrix,int[][] state){
          if(state[i][j] > 0) return state[i][j];
          int max = 0;
          for(int m = 0; m < dis.length; m++){
              if(i + dis[m][0] >= 0 && i + dis[m][0] < matrix.length 
                && j + dis[m][1] >= 0 && j + dis[m][1] < matrix[0].length 
                && matrix[i+dis[m][0]][j+dis[m][1]] > matrix[i][j]){
                  max = Math.max(max,dfs(i + dis[m][0],j + dis[m][1],matrix,state));
              }
          }
          state[i][j] = 1 + max;
          return state[i][j];

      }
}

317 Shortest Distance from All Buildings
You want to build a house on an empty land which reaches all buildings in the shortest amount of distance. 
You can only move up, down, left and right. You are given a 2D grid of values 0, 1 or 2, where:
Each 0 marks an empty land which you can pass by freely.
Each 1 marks a building which you cannot pass through.
Each 2 marks an obstacle which you cannot pass through.
For example, given three buildings at (0,0), (0,4), (2,2), and an obstacle at (0,2):
1 - 0 - 2 - 0 - 1
|   |   |   |   |
0 - 0 - 0 - 0 - 0
|   |   |   |   |
0 - 0 - 1 - 0 - 0

The point (1,2) is an ideal empty land to build a house, as the total travel distance of 3+3+1=7 is 
minimal. So return 7.
Note:
There will be at least one building. If it is not possible to build such house according to the above 
rules, return -1.

A BFS problem. Search from each building and calculate the distance to the building. One thing to note is 
an empty land must be reachable by all buildings. To achieve this, maintain an array of counters. Each 
time we reach a empty land from a building, increase the counter. Finally, a reachable point must have the
counter equaling to the number of buildings. 

public class Solution {
    public int shortestDistance(int[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }
         
        int m = grid.length;
        int n = grid[0].length;
         
        int[][] dist = new int[m][n];
        int[][] reach = new int[m][n];
        // step 1: BFS and calcualte the min dist from each building
        int numBuildings = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {
                    boolean[][] visited = new boolean[m][n];
                    Queue<Integer> queue = new LinkedList<>();
                    shortestDistanceHelper(i, j, 0, dist, reach, grid, visited, queue);
                    numBuildings++;
                }
            }
        }
         
        // step 2: caluclate the minimum distance
        int minDist = Integer.MAX_VALUE;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0 && reach[i][j] == numBuildings) {
                    minDist = Math.min(minDist, dist[i][j]);
                }
            }
        }
         
        return minDist == Integer.MAX_VALUE ? -1 : minDist;
    }
     
    private void shortestDistanceHelper(int x, int y, int currDist, 
                                        int[][] dist, int[][] reach, int[][] grid,
                                        boolean[][] visited, Queue<Integer> queue) {
        fill(x, y, x, y, currDist, dist, reach, grid, visited, queue);
         
        int m = grid.length;
        int n = grid[0].length;
         
        while (!queue.isEmpty()) {
            int size = queue.size();
            currDist++;
            for (int sz = 0; sz < size; sz++) {
                int cord = queue.poll();
                int i = cord / n;
                int j = cord % n;
                 
                fill(x, y, i - 1, j, currDist, dist, reach, grid, visited, queue);
                fill(x, y, i + 1, j, currDist, dist, reach, grid, visited, queue);
                fill(x, y, i, j - 1, currDist, dist, reach, grid, visited, queue);
                fill(x, y, i, j + 1, currDist, dist, reach, grid, visited, queue);
            }
 
        }
    }
     
    private void fill(int origX, int origY, int x, int y, int currDist, 
                      int[][] dist, int[][] reach,  
                      int[][] grid, boolean[][] visited, Queue<Integer> queue) {
         
        int m = dist.length;
        int n = dist[0].length;
        if (x < 0 || x >= m || y < 0 || y >= n || visited[x][y]) {
            return;
        }
         
        if ((x != origX || y != origY) && grid[x][y] != 0) {
            return;
        }
         
        visited[x][y] = true;
         
        dist[x][y] += currDist;
        reach[x][y]++;
         
        queue.offer(x * n + y);
    }
}


public class Solution {
    public int shortestDistance(int[][] grid) {
        int rows = grid.length;
        if (rows == 0) {
            return -1;
        }
        int cols = grid[0].length;
 
        // 记录到各个building距离和
        int[][] dist = new int[rows][cols];
        
        // 记录到能到达的building的数量
        int[][] nums = new int[rows][cols];            
        int buildingNum = 0;
        
        // 从每个building开始BFS
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == 1) {
                    buildingNum++;
                    bfs(grid, i, j, dist, nums);
                }
            }
        }
        
        int min = Integer.MAX_VALUE;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == 0 && dist[i][j] != 0 && nums[i][j] == buildingNum)
                    min = Math.min(min, dist[i][j]);
            }
        }
        if (min < Integer.MAX_VALUE)
            return min;
        return -1;
    }
    
    public void bfs(int[][] grid, int row, int col, int[][] dist, int[][] nums) {
        int rows = grid.length;
        int cols = grid[0].length;
        
        Queue<int[]> q = new LinkedList<>();
        q.add(new int[]{row, col});
        int[][] dirs = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
        
        // 记录访问过的点
        boolean[][] visited = new boolean[rows][cols];
        int level = 0;
        while (!q.isEmpty()) {
            level++;
            int size = q.size();
            for (int i = 0; i < size; i++) {
                int[] coords = q.remove();
                for (int k = 0; k < dirs.length; k++) {
                    int x = coords[0] + dirs[k][0];
                    int y = coords[1] + dirs[k][1];
                    if (x >= 0 && x < rows && y >= 0 && y < cols && !visited[x][y] && grid[x][y] == 0) {
                        visited[x][y] = true;
                        dist[x][y] += level;
                        nums[x][y]++;
                        q.add(new int[]{x, y});
                    }
                }
            }
        }
    }
}

315 Count of Smaller Numbers After Self
You are given an integer array nums and you have to return a new counts array. The counts array has the 
property where counts[i] is the number of smaller elements to the right of nums[i].

Example:
Given nums = [5, 2, 6, 1]

To the right of 5 there are 2 smaller elements (2 and 1).
To the right of 2 there is only 1 smaller element (1).
To the right of 6 there is 1 smaller element (1).
To the right of 1 there is 0 smaller element.
Return the array [2, 1, 1, 0].

The basic idea is to do merge sort to nums[]. To record the result, we need to keep the index of each 
number in the original array. So instead of sort the number in nums, we sort the indexes of each number. 
Example: nums = [5,2,6,1], indexes = [0,1,2,3] After sort: indexes = [3,1,0,2]
While doing the merge part, say that we are merging left[] and right[], left[] and right[] are already 
sorted.
We keep a rightcount to record how many numbers from right[] we have added and keep an array count[] to 
record the result.
When we move a number from right[] into the new sorted array, we increase rightcount by 1.
When we move a number from left[] into the new sorted array, we increase count[ index of the number ] by 
rightcount.

public class Solution {
    int[] count;
    public List<Integer> countSmaller(int[] nums) {
        List<Integer> res = new ArrayList<Integer>();     
    
        count = new int[nums.length];
        int[] indexes = new int[nums.length];
        for(int i = 0; i < nums.length; i++){
            indexes[i] = i;
        }
        mergesort(nums, indexes, 0, nums.length - 1);
        for(int i = 0; i < count.length; i++){
            res.add(count[i]);
        }
        return res;
    }
    private void mergesort(int[] nums, int[] indexes, int start, int end){
        if(end <= start){
            return;
        }
        int mid = (start + end) / 2;
        mergesort(nums, indexes, start, mid);
        mergesort(nums, indexes, mid + 1, end);
    
        merge(nums, indexes, start, end);
    }
    private void merge(int[] nums, int[] indexes, int start, int end){
        int mid = (start + end) / 2;
        int left_index = start;
        int right_index = mid+1;
        int rightcount = 0;     
        int[] new_indexes = new int[end - start + 1];
    
        int sort_index = 0;
        while(left_index <= mid && right_index <= end){
            if(nums[indexes[right_index]] < nums[indexes[left_index]]){
                new_indexes[sort_index] = indexes[right_index];
                rightcount++;
                right_index++;
            }else{
                new_indexes[sort_index] = indexes[left_index];
                count[indexes[left_index]] += rightcount;
                left_index++;
            }
            sort_index++;
        }
        while(left_index <= mid){
            new_indexes[sort_index] = indexes[left_index];
            count[indexes[left_index]] += rightcount;
            left_index++;
            sort_index++;
        }
        while(right_index <= end){
            new_indexes[sort_index++] = indexes[right_index++];
        }
        for(int i = start; i <= end; i++){
            indexes[i] = new_indexes[i - start];
        }
    }
}

327 Count of Range Sum
Given an integer array nums, return the number of range sums that lie in [lower, upper] inclusive.
Range sum S(i, j) is defined as the sum of the elements in nums between indices i and j (i ≤ j), 
inclusive.

Note:
A naive algorithm of O(n2) is trivial. You MUST do better than that.

Example:
Given nums = [-2, 5, -1], lower = -2, upper = 2,
Return 3.
The three ranges are : [0, 0], [2, 2], [0, 2] and their respective sums are: -2, -1, 2.
public int countRangeSum(int[] nums, int lower, int upper) {
    int n = nums.length;
    long[] sums = new long[n + 1];
    for (int i = 0; i < n; ++i)
        sums[i + 1] = sums[i] + nums[i];
    return countWhileMergeSort(sums, 0, n + 1, lower, upper);
}

private int countWhileMergeSort(long[] sums, int start, int end, int lower, int upper) {
    if (end - start <= 1) return 0;
    int mid = (start + end) / 2;
    int count = countWhileMergeSort(sums, start, mid, lower, upper) 
              + countWhileMergeSort(sums, mid, end, lower, upper);
    int j = mid, k = mid, t = mid;
    long[] cache = new long[end - start];
    for (int i = start, r = 0; i < mid; ++i, ++r) {
        while (k < end && sums[k] - sums[i] < lower) k++;
        while (j < end && sums[j] - sums[i] <= upper) j++;
        while (t < end && sums[t] < sums[i]) cache[r++] = sums[t++];
        cache[r] = sums[i];
        count += j - k;
    }
    System.arraycopy(cache, 0, sums, start, t - start);
    return count;
}



33  Search in Rotated Sorted Array
Suppose a sorted array is rotated at some pivot unknown to you beforehand.
(i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
You are given a target value to search. If found in the array return its index, otherwise return -1.
You may assume no duplicate exists in the array.

public int search(int[] A, int target) {
    // check if the target is in the sorted part, if so keep doing the binary search
    // otherwise throw away the sorted part and do the same on the other part of the array
    int start = 0;
    int end = A.length-1;

    while (start <= end) {
        int mid = (start + end) / 2;
        if (A[mid] == target) return mid;
        if (A[start] <= A[mid]) {
            // situation 1, red line
            if (A[start] <= target && target <= A[mid]) {
                end = mid-1;
            }
            else {
                start = mid+1;
            }
        }
        else {
            // situation 2, green line
            if (A[mid] <= target && target <= A[end]) {
                start = mid+1;
            }
            else {
                end = mid-1;
            }
        }
    }
    return -1;      
}


330 Patching Array
Given a sorted positive integer array nums and an integer n, add/patch elements to the array such that 
any number in range [1, n] inclusive can be formed by the sum of some elements in the array. Return 
the minimum number of patches required.

Example 1:
nums = [1, 3], n = 6
Return 1.

Combinations of nums are [1], [3], [1,3], which form possible sums of: 1, 3, 4.
Now if we add/patch 2 to nums, the combinations are: [1], [2], [3], [1,3], [2,3], [1,2,3].
Possible sums are 1, 2, 3, 4, 5, 6, which now covers the range [1, 6].
So we only need 1 patch.

Example 2:
nums = [1, 5, 10], n = 20
Return 2.
The two patches can be [2, 4].

Example 3:
nums = [1, 2, 2], n = 5
Return 0.

public class Solution {
    public int minPatches(int[] nums, int n) {
        long missing = 1;
        int patches = 0, i = 0;
    
        while (missing <= n) {
            if (i < nums.length && nums[i] <= missing) {
                missing += nums[i++];
            } else {
                missing += missing;
                patches++;
            }
        }
        return patches;
    }
}

115 Distinct Subsequences
Given a string S and a string T, count the number of distinct subsequences of T in S.
A subsequence of a string is a new string which is formed from the original string by deleting 
some (can be none) of the characters without disturbing the relative positions of the remaining 
characters. (ie, "ACE" is a subsequence of "ABCDE" while "AEC" is not).}

Here is an example:
S = "rabbbit", T = "rabbit"
Return 3.

给定字符串S和T，S通过删除某些位置的字符得到T的话，就记作一种subSequence。返回总共有几种。
思路二：这样的题都是可以用动态规划解决的。

用dp[i][j]记录S的前i个和T的前j个的符合个数，那么最后目标就是dp[S.size()][T.size()];

初始化，j = 0 时候，dp[i][0] = 1，因为所有的都可以通过删除所有变成空字符，并且只有一种。
递推式子如下了：
i和j都从1开始，且j不能大于i，因为匹配的长度至少为1开始，j大于i无意义
如果 i == j  那么 dp[i][j] = S.substr(0, i) == T.substr(0, j);
如果 i != j 分两种情况

S[i-1] != T[j-1] 时，也就是加入不加入i的影响是一样的，那么 dp[i][j] = dp[i - 1][j];
S[i-1] == T[j-1] 时，那么当前字符可选择匹配或者是不匹配，所以dp[i][j] = dp[i - 1][j -1] + dp[i - 1][j];
/**
 * Solution (DP):
 * We keep a m*n matrix and scanning through string S, while
 * m = T.length() + 1 and n = S.length() + 1
 * and each cell in matrix Path[i][j] means the number of distinct subsequences of 
 * T.substr(1...i) in S(1...j)
 * 
 * Path[i][j] = Path[i][j-1]            (discard S[j])
 *              +     Path[i-1][j-1]    (S[j] == T[i] and we are going to use S[j])
 *                 or 0                 (S[j] != T[i] so we could not use S[j])
 * while Path[0][j] = 1 and Path[i][0] = 0.
 */
int numDistinct(string S, string T) {
    int m = T.length();
    int n = S.length();
    if (m > n) return 0;    // impossible for subsequence
    vector<vector<int>> path(m+1, vector<int>(n+1, 0));
    for (int k = 0; k <= n; k++) path[0][k] = 1;    // initialization

    for (int j = 1; j <= n; j++) {
        for (int i = 1; i <= m; i++) {
            path[i][j] = path[i][j-1] + (T[i-1] == S[j-1] ? path[i-1][j-1] : 0);
        }
    }

    return path[m][n];
}

72  Edit Distance
Given two words word1 and word2, find the minimum number of steps required to convert word1 to word2. 
(each operation is counted as 1 step.)

You have the following 3 operations permitted on a word:
a) Insert a character
b) Delete a character
c) Replace a characte

Use f[i][j] to represent the shortest edit distance between word1[0,i) and word2[0, j). Then compare the
last character of word1[0,i) and word2[0,j), which are c and d respectively (c == word1[i-1], 
d == word2[j-1]):
if c == d, then : f[i][j] = f[i-1][j-1]

Otherwise we can use three operations to convert word1 to word2:
(a) if we replaced c with d: f[i][j] = f[i-1][j-1] + 1;
(b) if we added d after c: f[i][j] = f[i][j-1] + 1;
(c) if we deleted c: f[i][j] = f[i-1][j] + 1;

Note that f[i][j] only depends on f[i-1][j-1], f[i-1][j] and f[i][j-1], therefore we can reduce the space
to O(n) by using only the (i-1)th array and previous updated element(f[i][j-1]).

Actually at first glance I thought this question was similar to Word Ladder and I tried to solve it 
using BFS(pretty stupid huh?). But in fact, the main difference is that there's a strict restriction 
on the intermediate words in Word Ladder problem, while there's no restriction in this problem. If we 
added some restriction on intermediate words for this question, I don't think this DP solution would 
still work. 
public class Solution {
    public int minDistance(String word1, String word2) {
        int len1 = word1.length();
        int len2 = word2.length();

        //distance[i][j] is the distance converse word1(1~ith) to word2(1~jth)
        int[][] distance = new int[len1 + 1][len2 + 1]; 
        for (int j = 0; j <= len2; j++)
            {distance[0][j] = j;} //delete all characters in word2
        for (int i = 0; i <= len1; i++)
            {distance[i][0] = i;}

        for (int i = 1; i <= len1; i++) {
            for (int j = 1; j <= len2; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) { //ith & jth
                    distance[i][j] = distance[i - 1][j - 1];
                } else {
                    distance[i][j] = 
                    Math.min(Math.min(distance[i][j - 1], distance[i - 1][j]), distance[i - 1][j - 1]) + 1;
                }
            }
        }
        return distance[len1][len2];        
    }
}

297 Serialize and Deserialize Binary Tree
Serialization is the process of converting a data structure or object into a sequence of bits so that it 
can be stored in a file or memory buffer, or transmitted across a network connection link to be 
reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your 
serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be 
serialized to a string and this string can be deserialized to the original tree structure.

For example, you may serialize the following tree

    1
   / \
  2   3
     / \
    4   5
as "[1,2,3,null,null,4,5]", just the same as how LeetCode OJ serializes a binary tree. You do not 
necessarily need to follow this format, so please be creative and come up with different approaches 
yourself.
Note: Do not use class member/global/static variables to store states. Your serialize and deserialize 
algorithms should be stateless.

The idea is simple: print the tree in pre-order traversal and use "X" to denote null node and split node 
with ",". We can use a StringBuilder for building the string on the fly. For deserializing, we use a 
Queue to store the pre-order traversal and since we have "X" as null node, we know exactly how to where 
to end building subtress.
public class Codec {
    private static final String spliter = ",";
    private static final String NN = "X";

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        buildString(root, sb);
        return sb.toString();
    }

    private void buildString(TreeNode node, StringBuilder sb) {
        if (node == null) {
            sb.append(NN).append(spliter);
        } else {
            sb.append(node.val).append(spliter);
            buildString(node.left, sb);
            buildString(node.right,sb);
        }
    }
    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        Deque<String> nodes = new LinkedList<>();
        nodes.addAll(Arrays.asList(data.split(spliter)));
        return buildTree(nodes);
    }

    private TreeNode buildTree(Deque<String> nodes) {
        String val = nodes.remove();
        if (val.equals(NN)) return null;
        else {
            TreeNode node = new TreeNode(Integer.valueOf(val));
            node.left = buildTree(nodes);
            node.right = buildTree(nodes);
            return node;
        }
    }
}

239 Sliding Window Maximum
Given an array nums, there is a sliding window of size k which is moving from the very left of the array 
to the very right. You can only see the k numbers in the window. Each time the sliding window moves right 
by one position.

For example,
Given nums = [1,3,-1,-3,5,3,6,7], and k = 3.

Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
Therefore, return the max sliding window as [3,3,5,5,6,7]

public class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        int n = nums.length;
        if (n == 0) {
            return nums;
        }
        int[] result = new int[n - k + 1];
        LinkedList<Integer> dq = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            if (!dq.isEmpty() && dq.peek() < i - k + 1) {
                dq.poll();
            }
            while (!dq.isEmpty() && nums[i] >= nums[dq.peekLast()]) {
                dq.pollLast();
            }
            dq.offer(i);
            if (i - k + 1 >= 0) {
                result[i - k + 1] = nums[dq.peek()];
            }
        }
        return result;
    }
}

25  Reverse Nodes in k-Group
Given a linked list, reverse the nodes of a linked list k at a time and return its modified list.
If the number of nodes is not a multiple of k then left-out nodes in the end should remain as it is.
You may not alter the values in the nodes, only nodes itself may be changed.
Only constant memory is allowed.

For example,
Given this linked list: 1->2->3->4->5
For k = 2, you should return: 2->1->4->3->5
For k = 3, you should return: 3->2->1->4->5

public ListNode reverseKGroup(ListNode head, int k) {
    ListNode curr = head;
    int count = 0;
    while (curr != null && count != k) { // find the k+1 node
        curr = curr.next;
        count++;
    }
    if (count == k) { // if k+1 node is found
        curr = reverseKGroup(curr, k); // reverse list with k+1 node as head
        // head - head-pointer to direct part, 
        // curr - head-pointer to reversed part;
        while (count-- > 0) { // reverse current k-group: 
            ListNode tmp = head.next; // tmp - next head in direct part
            head.next = curr; // preappending "direct" head to the reversed list 
            curr = head; // move head of reversed part to a new node
            head = tmp; // move "direct" head to the next node in direct part
        }
        head = curr;
    }
    return head;
}


354 Russian Doll Envelopes
You have a number of envelopes with widths and heights given as a pair of integers (w, h). One envelope 
can fit into another if and only if both the width and height of one envelope is greater than the width 
and height of the other envelope.

What is the maximum number of envelopes can you Russian doll? (put one inside other)
Example:
Given envelopes = [[5,4],[6,4],[6,7],[2,3]], the maximum number of envelopes you can Russian doll is 3 
([2,3] => [5,4] => [6,7]).}

Sort the array. Ascend on width and descend on height if width are same.
Find the longest increasing subsequence based on height.
Since the width is increasing, we only need to consider height.
[3, 4] cannot contains [3, 3], so we need to put [3, 4] before [3, 3] when sorting otherwise it will be 
counted as an increasing number if the order is [3, 3], [3, 4]

public static int binarySearch(Object[] a, int fromIndex, int toIndex, Object key)
This method returns index of the search key, if it is contained in the array, else it 
returns (-(insertion point) - 1)

public class Solution {
    public int maxEnvelopes(int[][] envelopes) {
        if(envelopes == null || envelopes.length == 0 
           || envelopes[0] == null || envelopes[0].length != 2)
            return 0;
        Arrays.sort(envelopes, new Comparator<int[]>(){
            public int compare(int[] arr1, int[] arr2){
                if(arr1[0] == arr2[0])
                    return arr2[1] - arr1[1];
                else
                    return arr1[0] - arr2[0];
           } 
        });
        int dp[] = new int[envelopes.length];
        int len = 0;
        for(int[] envelope : envelopes){
            int index = Arrays.binarySearch(dp, 0, len, envelope[1]);
            if(index < 0)
                index = -(index + 1);
            dp[index] = envelope[1];
            if(index == len)
                len++;
        }
        return len;
    }
}
index: -1
envelope: 3
index: -2
envelope: 4
index: -3
envelope: 7
index: 1
envelope: 4

248 Strobogrammatic Number III
A strobogrammatic number is a number that looks the same when rotated 180 degrees (looked at upside down).
Write a function to count the total strobogrammatic numbers that exist in the range of low <= num <= high.
For example,
Given low = "50", high = "100", return 3. Because 69, 88, and 96 are three strobogrammatic numbers.
Note:
Because the range might be a large number, the low and high numbers are represented as string.}

Construct char array from lenLow to lenHigh and increase count when s is between low and high. 
Add the stro pairs from outside to inside until left > right.

public class Solution {
    char[][] pairs = {{'0', '0'}, {'1', '1'}, {'6', '9'}, {'8', '8'}, {'9', '6'}};
    int count = 0;

    public int strobogrammaticInRange(String low, String high) {
        for(int len = low.length(); len <= high.length(); len++) {
            dfs(low, high, new char[len], 0, len - 1);
        }
        return count;
    }

    public void dfs(String low, String high, char[] c, int left, int right) {
        if(left > right) {
            String s = new String(c);
            if((s.length() == low.length() && s.compareTo(low) < 0) || 
               (s.length() == high.length() && s.compareTo(high) > 0)) return;
            count++; 
            return;
        }

        for(char[] p : pairs) {
            c[left] = p[0]; 
            c[right] = p[1];
            if(c.length != 1 && c[0] == '0') continue;
            if(left < right || left == right && p[0] == p[1]) dfs(low, high, c, left + 1, right - 1);
        }
    }
｝


164 Maximum Gap
Given an unsorted array, find the maximum difference between the successive elements in its sorted form.
Try to solve it in linear time/space.

Return 0 if the array contains less than 2 elements.
You may assume all elements in the array are non-negative integers and fit in the 32-bit signed integer
range.

public class Solution {
    public int maximumGap(int[] nums) {
        int n = nums.length;
        if(n < 2) return 0;
        int min = nums[0];
        int max = nums[0];
        for(int i = 1;i < n;i++){
            if(min > nums[i]) min = nums[i];
            if(max < nums[i]) max = nums[i];
        }

        int gap = (max-min)/(n-1);
        if(gap == 0) gap++;
        int len = (max-min)/gap+1;
        int [] tmax = new int [len];
        int [] tmin = new int [len];

        for(int i = 0;i < n;i++){
            int index = (nums[i]-min)/gap;
            if(nums[i] > tmax[index]) tmax[index] = nums[i];
            if(tmin[index] == 0 || nums[i] < tmin[index]) tmin[index] = nums[i];
        }
        int myMax = 0;
        for(int i = 0;i < len;i++){
            if(myMax < tmin[i]-min) myMax = tmin[i]-min;
            if(tmax[i] != 0) min = tmax[i];
        }
        return myMax;
    }
}

99  Recover Binary Search Tree
Two elements of a binary search tree (BST) are swapped by mistake.
Recover the tree without changing its structure.
public void recoverTree(TreeNode root) {
    TreeNode pre = null;
    TreeNode first = null, second = null;
    // Morris Traversal
    TreeNode temp = null;
    while(root!=null){
        if(root.left!=null){
            // connect threading for root
            temp = root.left;
            while(temp.right!=null && temp.right != root)
                temp = temp.right;
            // the threading already exists
            if(temp.right!=null){
                if(pre!=null && pre.val > root.val){
                    if(first==null){first = pre;second = root;}
                    else{second = root;}
                }
                pre = root;

                temp.right = null;
                root = root.right;
            }else{
                // construct the threading
                temp.right = root;
                root = root.left;
            }
        }else{
            if(pre!=null && pre.val > root.val){
                if(first==null){first = pre;second = root;}
                else{second = root;}
            }
            pre = root;
            root = root.right;
        }
    }
    // swap two node values;
    if(first!= null && second != null){
        int t = first.val;
        first.val = second.val;
        second.val = t;
    }
}

87  Scramble String
Given a string s1, we may represent it as a binary tree by partitioning it to two non-empty substrings 
recursively.

Below is one possible representation of s1 = "great":
    great
   /    \
  gr    eat
 / \    /  \
g   r  e   at
           / \
          a   t
To scramble the string, we may choose any non-leaf node and swap its two children.
For example, if we choose the node "gr" and swap its two children, it produces a scrambled string 
"rgeat".
    rgeat
   /    \
  rg    eat
 / \    /  \
r   g  e   at
           / \
          a   t
We say that "rgeat" is a scrambled string of "great".
Similarly, if we continue to swap the children of nodes "eat" and "at", it produces a scrambled 
string "rgtae".
    rgtae
   /    \
  rg    tae
 / \    /  \
r   g  ta  e
       / \
      t   a
We say that "rgtae" is a scrambled string of "great".
Given two strings s1 and s2 of the same length, determine if s2 is a scrambled string of s1.

The main idea is:
separate s1 into two parts, namely --s11--, --------s12--------
separate s2 into two parts, namely --s21--, --------s22--------, 
and test the corresponding part (s11 and s21 && s12 and s22) with isScramble.
separate s2 into two parts, namely --------s23--------, --s24--, 
and test the corresponding part (s11 and s24 && s12 and s23) with isScramble.
Note that before testing each sub-part with isScramble, anagram is used first to test if 
the corresponding parts are anagrams. If not, skip directly.

public class Solution {
    public boolean isScramble(String s1, String s2) {
        if(s1==null || s2==null || s1.length()!=s2.length()) return false;
        if(s1.equals(s2)) return true;
        char[] c1 = s1.toCharArray();
        char[] c2 = s2.toCharArray();
        Arrays.sort(c1);
        Arrays.sort(c2);
        if(!Arrays.equals(c1, c2)) return false;
        for(int i=1; i<s1.length(); i++)
        {
            if(isScramble(s1.substring(0,i), s2.substring(0,i)) 
                && isScramble(s1.substring(i), s2.substring(i))) return true;
            if(isScramble(s1.substring(0,i), s2.substring(s2.length()-i)) 
                && isScramble(s1.substring(i), s2.substring(0, s2.length()-i))) return true;
        }
        return false;
    }
}


123 Best Time to Buy and Sell Stock III
Say you have an array for which the ith element is the price of a given stock on day i.
Design an algorithm to find the maximum profit. You may complete at most two transactions.

Note:
You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy 
again).

First assume that we have no money, so buy1 means that we have to borrow money from others, we want to 
borrow less so that we have to make our balance as max as we can(because this is negative).
sell1 means we decide to sell the stock, after selling it we have price[i] money and we have to give 
back the money we owed, so we have price[i] - |buy1| = prices[i ] + buy1, we want to make this max.
buy2 means we want to buy another stock, we already have sell1 money, so after buying stock2 we have 
buy2 = sell1 - price[i] money left, we want more money left, so we make it max
sell2 means we want to sell stock2, we can have price[i] money after selling it, and we have buy2 money 
left before, so sell2 = buy2 + prices[i], we make this max.

So sell2 is the most money we can have.
Hope it is helpful and welcome quesions!

public class Solution {
    public int maxProfit(int[] prices) {
        int sell1 = 0, sell2 = 0, buy1 = Integer.MIN_VALUE, buy2 = Integer.MIN_VALUE;
        for (int i = 0; i < prices.length; i++) {
            buy1 = Math.max(buy1, -prices[i]);
            sell1 = Math.max(sell1, buy1 + prices[i]);
            buy2 = Math.max(buy2, sell1 - prices[i]);
            sell2 = Math.max(sell2, buy2 + prices[i]);
        }
        return sell2;
    }
}


138 Copy List with Random Pointer
A linked list is given such that each node contains an additional random pointer which could point to 
any node in the list or null.
Return a deep copy of the list.

The idea is: 
Step 1: create a new node for each existing node and join them together eg: A->B->C will be 
A->A'->B->B'->C->C'
Step2: copy the random links: for each new node n', n'.random = n.random.next
Step3: detach the list: basically n.next = n.next.next; n'.next = n'.next.next

Here is the code:

/**
 * Definition for singly-linked list with a random pointer.
 * class RandomListNode {
 *     int label;
 *     RandomListNode next, random;
 *     RandomListNode(int x) { this.label = x; }
 * };
 */
public class Solution {
    public RandomListNode copyRandomList(RandomListNode head) {
        if(head==null){
            return null;
        }
        RandomListNode n = head;
        while (n!=null){
            RandomListNode n2 = new RandomListNode(n.label);
            RandomListNode tmp = n.next;
            n.next = n2;
            n2.next = tmp;
            n = tmp;
        }

        n=head;
        while(n != null){
            RandomListNode n2 = n.next;
            if(n.random != null)
                n2.random = n.random.next;
            else
                n2.random = null;
            n = n.next.next;
        }

        //detach list
        RandomListNode n2 = head.next;
        n = head;
        RandomListNode head2 = head.next;
        while(n2 != null && n != null){
            n.next = n.next.next;
            if (n2.next == null){
                break;
            }
            n2.next = n2.next.next;

            n2 = n2.next;
            n = n.next;
        }
        return head2;

    }
}


56  Merge Intervals
Given a collection of intervals, merge all overlapping intervals.

For example,
Given [1,3],[2,6],[8,10],[15,18],
return [1,6],[8,10],[15,18].

public class Solution {
    public List<Interval> merge(List<Interval> intervals) {
        if (intervals == null || intervals.size() == 1) {
            return intervals;
        }
        
        Collections.sort(intervals, new Comparator<Interval>(){
            public int compare(Interval i1, Interval i2) {
                return Integer.compare(i1.start,i2.start);
            }
        });
        
        ArrayList<Interval> list = new ArrayList<>();
        for (int i = 0; i < intervals.size(); i++) {
            Interval tmp = intervals.get(i);
            while (i + 1 < intervals.size() && intervals.get(i+1).start <= tmp.end) {
                tmp.end = Math.max(tmp.end,intervals.get(i+1).end);
                i++;
            }
            list.add(tmp);
        }
        return list;
    }
}

// pay attention to Collections.sort(), list.size(), end = Math.max(tmp.end, new end)

316 Remove Duplicate Letters
Given a string which contains only lowercase letters, remove duplicate letters so that every letter 
appear once and only once. You must make sure your result is the smallest in lexicographical order 
among all possible results.

Example:
Given "bcabc"
Return "abc"

Given "cbacdcbc"
Return "acdb"

First, given "bcabc", the solution should be "abc". If we think about this problem intuitively, you would
sort of go from the beginning of the string and start removing one if there is still the same character 
left and a smaller character is after it. Given "bcabc", when you see a 'b', keep it and continue with 
the search, then keep the following 'c', then we see an 'a'. Now we get a chance to get a smaller lexi 
order, you can check if after 'a', there is still 'b' and 'c' or not. We indeed have them and "abc" will 
be our result.
Come to the implementation, we need some data structure to store the previous characters 'b' and 'c', and 
we need to compare the current character with previous saved ones, and if there are multiple same 
characters, we prefer left ones. This calls for a stack.
After we decided to use stack, the implementation becomes clearer. From the intuition, we know that we 
need to know if there are still remaining characters left or not. So we need to iterate the array and 
save how many each characters are there. A visited array is also required since we want unique character 
in the solution. The line while(!stack.isEmpty() && stack.peek() > c && count[stack.peek()-'a'] > 0) 
checks that the queued character should be removed or not, like the 'b' and 'c' in the previous example. 
After removing the previous characters, push in the new char and mark the visited array.
Time complexity: O(n), n is the number of chars in string.
Space complexity: O(n) worst case.

The basic idea is to go through the given string char by char. If the current char has been used in the 
solution string, continue our loop to next char; If not, keep replacing the last char of current solution 
string with our current char being considered if the current character is smaller, then add current char 
to solution string.

The process requires an int array and a Boolean array to store the appearances and status(used or not) 
of each letter. And a stack is used to conveniently push and pop chars.

public String removeDuplicateLetters(String s) {
    Stack<Character> stack = new Stack<>();
    int[] count = new int[26];
    char[] arr = s.toCharArray();
    for(char c : arr) {
        count[c-'a']++;
    }
    boolean[] visited = new boolean[26];
    for(char c : arr) {
        count[c-'a']--;
        if(visited[c-'a']) {
            continue;
        }
        while(!stack.isEmpty() && stack.peek() > c && count[stack.peek()-'a'] > 0) {
            visited[stack.peek()-'a'] = false;
            stack.pop();
        }
        stack.push(c);
        visited[c-'a'] = true;
    }
    StringBuilder sb = new StringBuilder();
    for(char c : stack) {
        sb.append(c);
    }
    return sb.toString();
}


282 Expression Add Operators
Given a string that contains only digits 0-9 and a target value, return all possibilities to add 
binary operators (not unary) +, -, or * between the digits so they evaluate to the target value.

Examples: 
"123", 6 -> ["1+2+3", "1*2*3"] 
"232", 8 -> ["2*3+2", "2+3*2"]
"105", 5 -> ["1*0+5","10-5"]
"00", 0 -> ["0+0", "0-0", "0*0"]
"3456237490", 9191 -> []

overflow: we use a long type once it is larger than Integer.MAX_VALUE or minimum, we get over it.
0 sequence: because we can't have numbers with multiple digits started with zero, we have to deal '
with it too.
a little trick is that we should save the value that is to be multiplied in the next recursion.

public class Solution {
    public List<String> addOperators(String num, int target) {
        List<String> rst = new ArrayList<String>();
        if(num == null || num.length() == 0) return rst;
        helper(rst, "", num, target, 0, 0, 0);
        return rst;
    }
    public void helper(List<String> rst, String path, String num, int target, int pos, long eval, long multed){
        if(pos == num.length()){
            if(target == eval)
                rst.add(path);
            return;
        }
        for(int i = pos; i < num.length(); i++){
            if(i != pos && num.charAt(pos) == '0') break;
            long cur = Long.parseLong(num.substring(pos, i + 1));
            if(pos == 0){
                helper(rst, path + cur, num, target, i + 1, cur, cur);
            }
            else{
                helper(rst, path + "+" + cur, num, target, i + 1, eval + cur , cur);

                helper(rst, path + "-" + cur, num, target, i + 1, eval -cur, -cur);

                helper(rst, path + "*" + cur, num, target, i + 1, eval - multed + multed * cur, multed * cur );
            }
        }
    }
}


37  Sudoku Solver
Write a program to solve a Sudoku puzzle by filling the empty cells.
Empty cells are indicated by the character '.'.
You may assume that there will be only one unique solution.

public class Solution {
    public void solveSudoku(char[][] board) {
        boolean[][] status = new boolean[3*9][10];
        int index;

        for(int i = 0; i < 9; i++){//record the board status
            for(int j = 0; j < 9; j++){
                if(board[i][j] == '.'){
                    continue;
                }
                index = board[i][j] - '0';
                status[i][index] = true;
                status[9+j][index] = true;
                status[2*9+i/3*3+j/3][index] = true;
            }
        }
        helper(0, 0, status, board);
    }

    private boolean helper(int i, int j, boolean[][] status, char[][] board){
        if(j >= 9){
            i++;
            if(i >= 9){
                return true;
            }
            j = 0;
        }
        if(board[i][j] == '.'){
            int m;
            for(m = 1; m <= 9; m++){
                if(status[i][m] || status[9+j][m] || status[2*9 + i/3*3 + j/3][m]){// check which number to put
                    continue;
                }else{
                    board[i][j] = (char)(m+'0');
                    status[i][m] = status[9+j][m] = status[2*9 + i/3*3 + j/3][m] = true;
                    if(!helper(i, j+1, status, board)){// reverse the changes
                        board[i][j] = '.';
                        status[i][m] = status[9+j][m] = status[2*9 + i/3*3 + j/3][m] = false;
                    }else{
                        return true;
                    }
                }
            }
            if(m > 9){
                return false;
            }else{
                return true;
            }
        }else{
            return helper(i, j+1, status, board);
        }
}

358 Rearrange String k Distance Apart
Given a non-empty string str and an integer k, rearrange the string such that the same characters are at 
least distance k from each other.

All input strings are given in lowercase letters. If it is not possible to rearrange the string, return 
an empty string "".

Example 1:
str = "aabbcc", k = 3
Result: "abcabc"

The same letters are at least distance 3 from each other.
Example 2:
str = "aaabc", k = 3 
Answer: ""

It is not possible to rearrange the string.
Example 3:
str = "aaadbbcc", k = 2
Answer: "abacabcd"
Another possible answer is: "abcabcda"
The same letters are at least distance 2 from each other.
这道题给了我们一个字符串str，和一个整数k，让我们对字符串str重新排序，使得其中相同的字符之间的距离不小于k，这道题的难度标为Hard，
看来不是省油的灯。的确，这道题的解法用到了哈希表，堆，和贪婪算法。这道题我最开始想的算法没有通过OJ的大集合超时了，下面的方法是参
考网上大神的解法，发现十分的巧妙。我们需要一个哈希表来建立字符和其出现次数之间的映射，然后需要一个堆来保存这每一堆映射，按照出现
次数来排序。然后如果堆不为空我们就开始循环，我们找出k和str长度之间的较小值，然后从0遍历到这个较小值，对于每个遍历到的值，如果此时
堆为空了，说明此位置没法填入字符了，返回空字符串，否则我们从堆顶取出一对映射，然后把字母加入结果res中，此时映射的个数减1，如果
减1后的个数仍大于0，则我们将此映射加入临时集合v中，同时str的个数len减1，遍历完一次，我们把临时集合中的映射对由加入堆}]

public string rearrangeString(string str, int k) {
    if (k == 0) return str;
    string res;
    int len = (int)str.size();
    unordered_map<char, int> m;
    priority_queue<pair<int, char>> q;
    for (auto a : str) ++m[a];
    for (auto it = m.begin(); it != m.end(); ++it) {
        q.push({it->second, it->first});
    }
    while (!q.empty()) {
        vector<pair<int, int>> v;
        int cnt = min(k, len);
        for (int i = 0; i < cnt; ++i) {
            if (q.empty()) return "";
            auto t = q.top(); q.pop();
            res.push_back(t.second);
            if (--t.first > 0) v.push_back(t);
            --len;
        }
        for (auto a : v) q.push(a);
    }
    return res;
}

方法：根据出现频率将字母从大到小排列，以k为间隔进行重排。
public class Solution {  
    public String rearrangeString(String str, int k) {  
        if (k <= 0) return str;  
        int[] f = new int[26];  
        char[] sa = str.toCharArray();  
        for(char c: sa) f[c-'a']++;  
        int r = sa.length / k;  
        int m = sa.length % k;  
        int c = 0;  
        for(int g: f) {  
            if (g-r>1) return "";  
            if (g-r==1) c++;  
        }  
        if (c>m) return "";  
        Integer[] pos = new Integer[26];  
        for(int i=0; i<pos.length; i++) pos[i] = i;  
        Arrays.sort(pos, new Comparator<Integer>() {  
           @Override  
           public int compare(Integer i1, Integer i2) {  
               return f[pos[i2]] - f[pos[i1]];  
           }  
        });  
        char[] result = new char[sa.length];  
        for(int i=0, j=0, p=0; i<sa.length; i++) {  
            result[j] = (char)(pos[p]+'a');  
            if (-- f[pos[p]] == 0) p++;  
            j += k;  
            if (j >= sa.length) {  
                j %= k;  
                j++;  
            }  
        }  
        return new String(result);  
    }  
}  

45  Jump Game II
Given an array of non-negative integers, you are initially positioned at the first index of the array.
Each element in the array represents your maximum jump length at that position.
Your goal is to reach the last index in the minimum number of jumps.

For example:
Given array A = [2,3,1,1,4]
The minimum number of jumps to reach the last index is 2. (Jump 1 step from index 0 to 1, then 3 steps 
to the last index.)

I try to change this problem to a BFS problem, where nodes in level i are all the nodes that can be 
reached in i-1th jump. for example. 2 3 1 1 4 , is 2|| 3 1|| 1 4 ||
clearly, the minimum jump of 4 is 2 since 4 is in level 3. my ac code.

int jump(int A[], int n) {
 if(n<2)return 0;
 int level=0,currentMax=0,i=0,nextMax=0;

 while(currentMax-i+1>0){       //nodes count of current level>0
     level++;
     for(;i<=currentMax;i++){   //traverse current level , and update the max reach of next level
        nextMax=max(nextMax,A[i]+i);
        if(nextMax>=n-1)return level;   // if last element is in level+1,  then the min jump=level 
     }
     currentMax=nextMax;
 }
 return 0;
}


public int jump(int[] A) {
    int count = 0, max = 0;
    for (int i = 0, nextMax = 0; i <= max && i < A.length - 1; i++) {
        nextMax = Math.max(nextMax, i + A[i]);
        if (i == max) {
            max = nextMax;
            count++;
        }
    }
    // if there is no way to get to the end, return -1
    return max >= A.length - 1 ? count : -1;
}


233 Number of Digit One
Given an integer n, count the total number of digit 1 appearing in all non-negative integers less than 
or equal to n.

For example:
Given n = 13,
Return 6, because digit 1 occurred in the following numbers: 1, 10, 11, 12, 13.

https://leetcode.com/discuss/44281/4-lines-o-log-n-c-java-python
Go through the digit positions by using position multiplier m with values 1, 10, 100, 1000, etc.

For each position, split the decimal representation into two parts, for example split n=3141592 into 
a=31415 and b=92 when we're' at m=100 for analyzing the hundreds-digit. And then we know that the 
hundreds-digit of n is 1 for prefixes "" to "3141", i.e., 3142 times. Each of those times is a streak, 
though. Because it's the hundreds-digit, each streak is 100 long. So (a / 10 + 1) * 100 times, the '
hundreds-digit is 1.

Consider the thousands-digit, i.e., when m=1000. Then a=3141 and b=592. The thousands-digit is 1 for 
prefixes "" to "314", so 315 times. And each time is a streak of 1000 numbers. However, since the 
thousands-digit is a 1, the very last streak isn't 1000 numbers but only 593 numbers, for the suffixes 
"000" to "592". So (a / 10 * 1000) + (b + 1) times, the thousands-digit is 1.'

The case distincton between the current digit/position being 0, 1 and >=2 can easily be done in one 
expression. With (a + 8) / 10 you get the number of full streaks, and a % 10 == 1 tells you whether 
to add a partial streak

public int countDigitOne(int n) {
    int ones = 0;
    for (long m = 1; m <= n; m *= 10)
        ones += (n/m + 8) / 10 * m + (n/m % 10 == 1 ? n%m + 1 : 0);
    return ones;
}


public int countDigitOne(int n) {
    int ones = 0, m = 1, r = 1;
    while (n > 0) {
        ones += (n + 8) / 10 * m + (n % 10 == 1 ? r : 0);
        r += n % 10 * m;
        m *= 10;
        n /= 10;
    }
    return ones;
}

    
84  Largest Rectangle in Histogram
Given n non-negative integers representing the histogram's' bar height where the width of each bar is 1, 
find the area of largest rectangle in the histogram.
Above is a histogram where width of each bar is 1, given height = [2,1,5,6,2,3].
The largest rectangle is shown in the shaded area, which has area = 10 unit.

For example,
Given heights = [2,1,5,6,2,3],
return 10.
public class Solution {
    public int largestRectangleArea(int[] height) {
        int len = height.length;
        Stack<Integer> s = new Stack<Integer>();
        int maxArea = 0;
        for(int i = 0; i <= len; i++){
            int h = (i == len ? 0 : height[i]);
            if(s.isEmpty() || h >= height[s.peek()]){
                s.push(i);
            }else{
                int tp = s.pop();
                maxArea = Math.max(maxArea, height[tp] * (s.isEmpty() ? i : i - 1 - s.peek()));
                i--;
            }
        }
        return maxArea;
    }
}


57  Insert Interval
Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).
You may assume that the intervals were initially sorted according to their start times.

Example 1:
Given intervals [1,3],[6,9], insert and merge [2,5] in as [1,5],[6,9].

Example 2:
Given [1,2],[3,5],[6,7],[8,10],[12,16], insert and merge [4,9] in as [1,2],[3,10],[12,16].
This is because the new interval [4,9] overlaps with [3,5],[6,7],[8,10].

public List<Interval> insert(List<Interval> intervals, Interval newInterval) {
    List<Interval> result = new LinkedList<>();
    int i = 0;
    // add all the intervals ending before newInterval starts
    while (i < intervals.size() && intervals.get(i).end < newInterval.start)
        result.add(intervals.get(i++));
    // merge all overlapping intervals to one considering newInterval
    while (i < intervals.size() && intervals.get(i).start <= newInterval.end) {
        newInterval = new Interval( // we could mutate newInterval here also
                Math.min(newInterval.start, intervals.get(i).start),
                Math.max(newInterval.end, intervals.get(i).end));
        i++;
    }
    result.add(newInterval); // add the union of intervals we got
    // add all the rest
    while (i < intervals.size()) result.add(intervals.get(i++)); 
    return result;
}


41  First Missing Positive
Given an unsorted integer array, find the first missing positive integer.
For example,
Given [1,2,0] return 3,
and [3,4,-1,1] return 2.

Your algorithm should run in O(n) time and uses constant space.

The basic idea is for any k positive numbers (duplicates allowed), the first missing positive number must 
e within [1,k+1]. The reason is like you put k balls into k+1 bins, there must be a bin empty, the empty 
bin can be viewed as the missing number.

Unfortunately, there are 0 and negative numbers in the array, so firstly I think of using partition 
technique (used in quick sort) to put all positive numbers together in one side. This can be finished 
in O(n) time, O(1) space.

After partition step, you get all the positive numbers lying within A[0,k-1]. Now, According to the basic 
idea, I infer the first missing number must be within [1,k+1]. I decide to use A[i] (0<=i<=k-1) to 
indicate whether the number (i+1) exists. But here I still have to main the original information A[i] 
holds. Fortunately, A[i] are all positive numbers, so I can set them to negative to indicate the 
existence of (i+1) and I can still use abs(A[i]) to get the original information A[i] holds.
After step 2, I can again scan all elements between A[0,k-1] to find the first positive element A[i], 
that means (i+1) doesn't exist, which is what I want.'


 public int firstMissingPositive(int[] A) {
    int n=A.length;
    if(n==0)
        return 1;
    int k=partition(A)+1;
    int temp=0;
    int first_missing_Index=k;
    for(int i=0;i<k;i++){
        temp=Math.abs(A[i]);
        if(temp<=k)
            A[temp-1]=(A[temp-1]<0)?A[temp-1]:-A[temp-1];
    }
    for(int i=0;i<k;i++){
        if(A[i]>0){
            first_missing_Index=i;
            break;
        }
    }
    return first_missing_Index+1;
}

public int partition(int[] A){
    int n=A.length;
    int q=-1;
    for(int i=0;i<n;i++){
        if(A[i]>0){
            q++;
            swap(A,q,i);
        }
    }
    return q;
}

public void swap(int[] A, int i, int j){
    if(i!=j){
        A[i]^=A[j];
        A[j]^=A[i];
        A[i]^=A[j];
    }
}

----
public int firstMissingPositive(int[] nums) {
    int start = 0;
    int end = nums.length - 1;
    while (start <= end) {
        int index = nums[start] - 1;
        if (index == start)
            start++;
        else if (index < 0 || index > end || nums[start] == nums[index])
            nums[start] = nums[end--];
        else {
            nums[start] = nums[index];
            nums[index] = index + 1;
        }
    }
    return start + 1;
}



