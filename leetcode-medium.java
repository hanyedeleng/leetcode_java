338	Counting Bits
Given a non negative integer number num. For every numbers i in the range 0 ≤ i ≤ num calculate the number
of 1's in their binary representation and return them as an array.'

Example:
For num = 5 you should return [0,1,1,2,1,2].

public class Solution {
    public int[] countBits(int num) {
        int[] ans = new int[num+1];
        
        for (int i = 0; i < num+1; i++) {
            ans[i] = Integer.toBinaryString(i).replace("0","").length();
        }
        return ans;
    }
}

136	Single Number	
Given an array of integers, every element appears twice except for one. Find that single one.
Note:
Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?

public class Solution {
    public int singleNumber(int[] nums) {
        int ans = nums[0];
        for (int i = 1; i < nums.length; i++) {
            ans = ans ^ nums[i];
        }
        return ans;
    }
}
// Java bitwise operator
//a = 0011 1100
//b = 0000 1101
//a^b = 0011 0001
// exclusive-or ("xor") operator

'*************************'
280	Wiggle Sort
Wiggle Sort
Given an unsorted array nums, reorder it in-place such that nums[0] <= nums[1] >= nums[2] <= nums[3]....
For example, given nums = [3, 5, 2, 1, 6, 4], one possible answer is [1, 6, 2, 5, 3, 4].

public class Solution {
    public void wiggleSort(int[] nums) {
        if (nums == null || nums.length <= 0)
            return;
        for (int i = 1; i < nums.length; i++) {
            if (i % 2 == 1) {
                if (nums[i-1] > nums[i])
                    swap(nums, i, i-1);
            } else{
                if (nums[i] > nums[i-1])
                    swap(nums, i, i-1);
            }
        }
    }
    
    
    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}


324 Wiggle Sort II
Given an unsorted array nums, reorder it such that nums[0] < nums[1] > nums[2] < nums[3]....
Example:
(1) Given nums = [1, 5, 1, 1, 6, 4], one possible answer is [1, 4, 1, 5, 1, 6]. 
(2) Given nums = [1, 3, 2, 2, 3, 1], one possible answer is [2, 3, 1, 3, 1, 2].

Note:
You may assume all input has valid answer.

Follow Up:
Can you do it in O(n) time and/or in-place with O(1) extra space?
public void wiggleSort(int[] nums) {
    Arrays.sort(nums);
    int[] temp = new int[nums.length];
    int mid = nums.length%2==0?nums.length/2-1:nums.length/2;
    int index = 0;
    for(int i=0;i<=mid;i++){
        temp[index] = nums[mid-i];
        if(index+1<nums.length)
            temp[index+1] = nums[nums.length-i-1];
        index = index+2;
    }
    for(int i=0;i<nums.length;i++){
        nums[i] = temp[i];
    }
}

other solution:
public class Solution {
       public void wiggleSort(int[] nums) {
        int median = findKthLargest(nums, (nums.length + 1) / 2);
        int n = nums.length;

        int left = 0, i = 0, right = n - 1;

        while (i <= right) {

            if (nums[newIndex(i,n)] > median) {
                swap(nums, newIndex(left++,n), newIndex(i++,n));
            }
            else if (nums[newIndex(i,n)] < median) {
                swap(nums, newIndex(right--,n), newIndex(i,n));
            }
            else {
                i++;
            }
        }


    }

    private int newIndex(int index, int n) {
        return (1 + 2*index) % (n | 1);
    }
}



167	Two Sum II - Input array is sorted
Given an array of integers that is already sorted in ascending order, find two numbers such that they add up
to a specific target number.
The function twoSum should return indices of the two numbers such that they add up to the target, where
index1 must be less than index2.
Please note that your returned answers (both index1 and index2) are not zero-based.
You may assume that each input would have exactly one solution.
Input: numbers={2, 7, 11, 15}, target=9
Output: index1=1, index2=2

public class Solution {
	public int[] twoSum(int[] numbers, int target) {
		int ans[] = new int[2];

		int lo = 0;
		int hi = numbers.length-1;
		while (lo < hi) {
			if (numbers[lo] + numbers[hi] == target) {
				ans[0] = lo + 1;
				ans[1] = hi + 1;
				break;
			} else if (numbers[lo] + numbers[hi] > target) {
				hi--;
			} else if (numbers[lo] + numbers[hi] < target) {
				lo++;
			}
		}
		return ans;
	}
}

other's solution':
public class Solution {
    public int[] twoSum(int[] nums, int target) {
    	int[] rst = new int[2];
        if (nums == null || nums.length <= 1) {
        	return rst;
        }
        int start = 0;
        int end = nums.length - 1;
        while(start < end) {
        	long sum = (long)(nums[start] + nums[end]); //****
        	if (target == sum) {
        		rst[0] = start + 1;
        		rst[1] = end + 1;
        		break; //******
        	} else if (target > sum) {
        		start++;
        	} else {
        		end--;
        	}
        }//END while
        return rst;
    }
}

// 1. use long for sum.
// 2. remember to break;

'*********************************'
311	Sparse Matrix Multiplication
Problem Description:
Given two sparse matrices A and B, return the result of AB.
You may assume that A's column number is equal to B's row number.
Example:
A = [
  [ 1, 0, 0],
  [-1, 0, 3]
]
B = [
  [ 7, 0, 0 ],
  [ 0, 0, 0 ],
  [ 0, 0, 1 ]
]
     |  1 0 0 |   | 7 0 0 |   |  7 0 0 |
AB = | -1 0 3 | x | 0 0 0 | = | -7 0 3 |
                  | 0 0 1 |

public class Solution {
    public int[][] multiply(int[][] A, int[][] B) {
      int rowLength = A.length; //row
      int commonLength = A[0].length;
      int colLength = B[0].length; // column
      int[][] result = new int[rowLength][colLength];

      for (int i=0; i<rowLength; i++) {
        for (int j=0; j<colLength; j++) {
          for (int k=0; k<commonLength; k++) {
            result[i][j] += A[i][k]*B[k][j]; 
          }
        }
      }
      
      return result;
    }
}

for the loop of the above solution, someone has the following thought;
for (int i = 0; i < m; i++)//m是A的行数  
            for (int j = 0; j < n; j++)//n是A的列数和B的行数  
                if (A[i][j])  
                    for (int k = 0; k < p; k++)//p是B的列数  
                        C[i][k] += A[i][j] * B[j][k];  
        return C;  


Sparse Matrix相乘。题目提示要用HashMap，于是我们就用HashMap， 保存A中不为0行，以及B中不为0的列，
然后遍历两个hashmap来更新结果数组。
Time Complexity - O(mnkl)，  Space Complexity - O(mn + kl)。
public class Solution {
    public int[][] multiply(int[][] A, int[][] B) {
        if(A == null || B == null || A.length == 0 || B.length == 0 || (A[0].length != B.length)) {
            return new int[][]{};
        }
        
        Map<Integer, int[]> rowInA = new HashMap<>();     // store non-zero rows in A
        Map<Integer, int[]> colInB = new HashMap<>();     // store non-zero cols in B
        
        for(int i = 0; i < A.length; i++) {
            for(int j = 0; j < A[0].length; j++) {
                if(A[i][j] != 0) {
                    rowInA.put(i, A[i]);
                    break;
                }
            }
        }
        
        for(int j = 0; j < B[0].length; j++) {
            for(int i = 0; i < B.length; i++) {
                if(B[i][j] != 0) {
                    int[] tmp = new int[B.length];
                    for(int k = 0; k <  B.length; k++) {
                        tmp[k] = B[k][j];
                    }
                    colInB.put(j, tmp);
                    break;
                }
            }
        }
        
        int[][] res = new int[A.length][B[0].length];
        
        for(int i : rowInA.keySet()) {
            for(int j : colInB.keySet()) {
                for(int k = 0; k < A[0].length; k++) {
                    res[i][j] += rowInA.get(i)[k] * colInB.get(j)[k];
                }
            }
        }
        
        return res;
    }
}

245	Shortest Word Distance III
This is a follow up of Shortest Word Distance. The only difference is now word1 could be the same as word2.
Given a list of words and two words word1 and word2, return the shortest distance between these two words
in the list.
word1 and word2 may be the same and they represent two individual words in the list.
For example,
Assume that words = ["practice", "makes", "perfect", "coding", "makes"].
Given word1 = “makes”, word2 = “coding”, return 1.
Given word1 = "makes", word2 = "makes", return 3.
Note:
You may assume word1 and word2 are both in the list.

public class Solution {
	public int shortestWordDistance (int[] words, String w1, String w2) {
		int t1 = -1;
		int t2 = -1;
		int min = words.length;

		for (int i = 0; i < words.length; i++) {
			if (w1.equals(w2)) {
				if (words[i].equals(w1)) {
					t1 = t2;
					t2 = i;
				}
			}else {
				if (words[i].equals(w1)) {
					t1 = i;
				}
				if (words[i].equals(w2)) {
					t2 = i;
				}
			}

			if (t1 != -1 && t2 != -1) {
				min = Math.min(min, Math.abs(t1-t2));
			}
		}
		return min;
	}
}
other's solution':
public class Solution {
    public int shortestWordDistance(String[] words, String word1, String word2) {
        int posA = -1;
        int posB = -1;
        int minDistance = Integer.MAX_VALUE;
         
        for (int i = 0; i < words.length; i++) {
            String word = words[i];
             
            if (word.equals(word1)) {
                posA = i;
            } else if (word.equals(word2)) {
                posB = i;
            }
             
            if (posA != -1 && posB != -1 && posA != posB) {
                minDistance = Math.min(minDistance, Math.abs(posA - posB));
            }
             
            if (word1.equals(word2)) {
                posB = posA;
            }
        }
         
        return minDistance;
    }
}

260	Single Number III
Given an array of numbers nums, in which exactly two elements appear only once and all the other elements
appear exactly twice. Find the two elements that appear only once.

For example:
Given nums = [1, 2, 1, 3, 2, 5], return [3, 5].

Note:
The order of the result is not important. So in the above example, [5, 3] is also correct.
Your algorithm should run in linear runtime complexity. Could you implement it using only constant space
complexity?

public class Solution {
    public int[] singleNumber(int[] nums) {
        // use set
        Set<Integer> s = new HashSet<>();
        int[] ans = new int[2];
        if (nums.length >= 2) {
            for (int i = 0; i < nums.length; i++) {
                if (s.add(nums[i]) == false) {
                    s.remove(nums[i]);
                }
            }
            int j = 0;
            for (int n : s) {
                ans[j++] = n;
            }
        }
        return ans;
    }
}

"***************************"
281	Zigzag Iterator
Suppose you have a Iterator class with has_next() and get_next() methods.
Please design and implement a ZigzagIterator class as a wrapper of two iterators.
For example, given two iterators:
i0 = [1,2,3,4]
i1 = [5,6]
ZigzagIterator it(i0, i1);

while(it.has_next()) {
    print(it.get_next());
}
The output of the above pseudocode would be [1,5,2,6,3,4].

public class ZigzagIterator {  
    Iterator i0, i1;  
    Iterator it;  
    public ZigzagIterator(Iterator i0, Iterator i1) {  
        this.i0 = i0; this.i1 = i1;  
        this.it = i0.hasNext()? i0:i1;  
    }  
      
    public boolean has_next() {  
        return it.hasNext();  
    }  
      
    public int get_next() {  
        int val = (Integer)it.next();  
        if(it == i0 && i1.hasNext())  
            it = i1;  
        else if(it == i1 && i0.hasNext())  
            it = i0;  
        return val;  
    }  
} 

"*********************************"
238	Product of Array Except Self
Given an array of n integers where n > 1, nums, return an array output such that output[i] is equal to the
product of all the elements of nums except nums[i].

Solve it without division and in O(n).
For example, given [1,2,3,4], return [24,12,8,6].

Follow up:
Could you solve it with constant space complexity? (Note: The output array does not count as extra space
for the purpose of space complexity analysis.)

The idea is to traverse twice. First traversal will get the product before current element. 
Second traversal will start from the end, and get the product after current element.

public int[] productExceptSelf(int[] nums) {
    int len = nums.length;
    int[] res = new int[len];
    if(len == 0 ){
        return res;
    }
    res[0] = 1;
    for(int i=1; i<len; i++){
        res[i] = res[i-1]*nums[i-1];
    }
    int rearProduct = 1;
    for(int j=len-1; j>=0; j--){
        res[j] = res[j] *rearProduct;
        rearProduct *= nums[j];
    }
    return res;
}

256	Paint House
There are a row of n houses, each house can be painted with one of the three colors: red, blue or green. 
The cost of painting each house with a certain color is different. 
You have to paint all the houses such that no two adjacent houses have the same color.
The cost of painting each house with a certain color is represented by a n x 3 cost matrix. 
For example, costs0 is the cost of painting house 0 with color red; 
costs1 is the cost of painting house 1 with color green, and so on... Find the minimum cost to paint all
houses.
Note: All costs are positive integers.

public class Solution {
	public int minCost(int[][] costs) {
	    if(costs.length==0) return 0;
	    int lastR = costs[0][0];
	    int lastG = costs[0][1];
	    int lastB = costs[0][2];
	    for(int i=1; i<costs.length; i++){
	        int curR = Math.min(lastG,lastB)+costs[i][0];
	        int curG = Math.min(lastR,lastB)+costs[i][1];
	        int curB = Math.min(lastR,lastG)+costs[i][2];
	        lastR = curR;
	        lastG = curG;
	        lastB = curB;
	    }
	    return Math.min(Math.min(lastR,lastG),lastB);
	}
}

'*****************************************'
323	Number of Connected Components in an Undirected Graph
Given n nodes labeled from 0 to n - 1 and a list of undirected edges (each edge is a pair of nodes), 
write a function to find the number of connected components in an undirected graph.

Example 1:
     0          3
     |          |
     1 --- 2    4
Given n = 5 and edges = [[0, 1], [1, 2], [3, 4]], return 2.
Example 2:
     0           4
     |           |
     1 --- 2 --- 3
Given n = 5 and edges = [[0, 1], [1, 2], [2, 3], [3, 4]], return 1.
Note:
You can assume that no duplicate edges will appear in edges. Since all edges are undirected, 
[0, 1] is the same as [1, 0] and thus will not appear together in edges.

public class Solution {
	public int countComponents(int n, int[][] edges) {
	    int[] root = new int[n];
	    for(int i = 0; i < n; i++) root[i] = i;
	    for(int[] edge : edges){
	        int root1 = findRoot(root, edge[0]), root2 = findRoot(root, edge[1]);
	        //Union
	        if(root1 != root2) root[root2] = root1;
	    }
	    //Count components
	    int count = 0;
	    for(int i = 0; i < n; i++) if(root[i] == i) count++;
	    return count;
	}

	//Find with path compression 
	private int findRoot(int[] root, int i){
	    while(root[i] != i){
	        root[i] = root[root[i]];
	        i = root[i];
	    }
	    return i;
	}
}

122	Best Time to Buy and Sell Stock II
Say you have an array for which the ith element is the price of a given stock on day i.
Design an algorithm to find the maximum profit. You may complete as many transactions as you like (ie,
buy one and sell one share of the stock multiple times). However, you may not engage in multiple
transactions at the same time (ie, you must sell the stock before you buy again).

public class Solution {
    public int maxProfit(int[] prices) {
        int ans = 0;
        if (prices.length <= 1) {
            return ans;
        }
        
        int pre = prices[0];
        
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > pre) {
                ans = ans + prices[i] - pre;
                pre = prices[i];
            }else {
                pre = prices[i];
            }
        }
        return ans;
    }
}

public class Solution {    
    public int maxProfit(int[] prices) {         
	    int result=0;         
	    for(int i=1;i<prices.length;i++){             
	        result+=Math.max(0,prices[i]-prices[i-1]);         
	    }        
	    return result;     
    } 
 }


347	Top K Frequent Elements
Given a non-empty array of integers, return the k most frequent elements.

For example,
Given [1,1,1,2,2,3] and k = 2, return [1,2].

Note: 
You may assume k is always valid, 1 ≤ k ≤ number of unique elements.
Your algorithm's time complexity must be better than O(n log n), where n is the array's size.

http://www.programcreek.com/2014/05/leetcode-top-k-frequent-elements-java/
public List<Integer> topKFrequent(int[] nums, int k) {
    //count the frequency for each element
    HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
    for(int num: nums){
        if(map.containsKey(num)){
            map.put(num, map.get(num)+1);
        }else{
            map.put(num, 1);
        }
    }
 
    //get the max frequency
    int max = 0;
    for(Map.Entry<Integer, Integer> entry: map.entrySet()){
        max = Math.max(max, entry.getValue());
    }
 
    //initialize an array of ArrayList. index is frequency, value is list of numbers
    ArrayList<Integer>[] arr = (ArrayList<Integer>[]) new ArrayList[max+1];
    for(int i=1; i<=max; i++){
        arr[i]=new ArrayList<Integer>();
    }
 
    for(Map.Entry<Integer, Integer> entry: map.entrySet()){
        int count = entry.getValue();
        int number = entry.getKey();
        arr[count].add(number);
    }
 
    List<Integer> result = new ArrayList<Integer>();
 
    //add most frequent numbers to result
    for(int j=max; j>=1; j--){
        if(arr[j].size()>0){
            for(int a: arr[j]){
                result.add(a);
            }
        }
 
        if(result.size()==k)
            break;
    }
 
    return result;
}

348	Design Tic-Tac-Toe
Design a Tic-tac-toe game that is played between two players on a n x n grid.

You may assume the following rules:

A move is guaranteed to be valid and is placed on an empty block.
Once a winning condition is reached, no more moves is allowed.
A player who succeeds in placing n of their marks in a horizontal, vertical, or diagonal row wins the game.

思路:建立两个大小为n的一维数组rows和cols,以及两个变量diagonal和antiDiagonal,分别代表对角线和反对角线,如果玩家1在某一位置
(假设为(i,j))放置了一个棋子,则rows[i]和cols[j]分别加1,如果该位置在对角线或者反对角线上,则diagonal或者antiDiagonal加1;
如果玩家2在某一位置(假设为(k,l))放置了一个棋子,则rows[k]和cols[l]分别减1,如果该位置在对角线或者反对角线上,则diagonal或者
antiDiagonal减1;那么只有当rows中某个值或者cols中某个值或者diagonal或者antiDiagonal值为n或-n的时候,表示该行、该列、该对
角线或者该反对角线上的棋子都是同一个玩家放的(值为n时代表玩家1,值为-n时代表玩家2),此时返回相应的玩家即可。

public class TicTacToe {
    private int[] rows;
    private int[] cols;
    private int diagonal;
    private int antiDiagonal;
    private int size;

    /**
     * Initialize your data structure here.
     */
    public TicTacToe(int n) {
        rows = new int[n];
        cols = new int[n];
        size = n;
    }

    /**
     * Player {player} makes a move at ({row}, {col}).
     *
     * @param row    The row of the board.
     * @param col    The column of the board.
     * @param player The player, can be either 1 or 2.
     * @return The current winning condition, can be either:
     * 0: No one wins.
     * 1: Player 1 wins.
     * 2: Player 2 wins.
     */
    public int move(int row, int col, int player) {
        int toAdd = player == 1 ? 1 : -1;

        rows[row] += toAdd;
        cols[col] += toAdd;
        if (row == col) {
            diagonal += toAdd;
        }

        if (col == (cols.length - row - 1)) {
            antiDiagonal += toAdd;
        }

        if (Math.abs(rows[row]) == size ||
                Math.abs(cols[col]) == size ||
                Math.abs(diagonal) == size ||
                Math.abs(antiDiagonal) == size) {
            return player;
        }

        return 0;
    }
}

"***********************"
recursive

294	Flip Game II
You are playing the following Flip Game with your friend: Given a string that contains only these two
characters: + and -, 
you and your friend take turns to flip two consecutive "++" into "--". 
The game ends when a person can no longer make a move and therefore the other person will be the winner.
Write a function to determine if the starting player can guarantee a win.
For example, given s = "++++", return true. The starting player can guarantee a win by flipping the middle
"++" to become "+--+".
Follow up:
Derive your algorithm's runtime complexity'

public class Solution {
     public boolean canWin(String s) {
         char[] list = s.toCharArray();
         return helper(list);
     }
     private boolean helper(char[] list) {
         for (int i = 0; i < list.length - 1; i++) {
             if (list[i] == '-' || list[i + 1] == '-') continue;
             list[i] = '-';
             list[i + 1] = '-';
             boolean otherWin = helper(list);
             //need to go back to the original state before return
             list[i] = '+';
             list[i + 1] = '+';
             if (!otherWin) return true;
         }
         return false;
     }
}

	public boolean canWin(String s) {
	    if(s == null || s.length() < 2) return false;
	    Map<String, Boolean> map = new HashMap<>();
	    return canWin(s, map);
	}

	public boolean canWin(String s, Map<String, Boolean> map){
	    if(map.containsKey(s)) return map.get(s);
	    for(int i = 0; i < s.length() - 1; i++) {
	        if(s.charAt(i) == '+' && s.charAt(i + 1) == '+') {
	            String opponent = s.substring(0, i) + "--" + s.substring(i + 2);
	            if(!canWin(opponent, map)) {
	                map.put(s, true);
	                return true;
	            }
	        }
	    }
	    map.put(s, false);
	    return false;
	}


343	Integer Break
Given a positive integer n, break it into the sum of at least two positive integers and maximize the
product of those integers. Return the maximum product you can get.

For example, given n = 2, return 1 (2 = 1 + 1); given n = 10, return 36 (10 = 3 + 3 + 4).
Note: you may assume that n is not less than 2.
public class Solution {
    public int integerBreak(int n) {
        int[] ans = new int[n+1];
        ans[0] = 0;
        ans[1] = 0;
        ans[2] = 1;
        
        for (int i = 2; i < n + 1; i++) {
            if (ans[i] == 0) {
                for (int j = 2; j <= i-1; j++) {
                    ans[i] = Math.max(ans[i], j*Math.max(i-j,ans[i-j]));
                }
            }
        }
        return ans[n];
    }
}

public class Solution {
    public int integerBreak(int n) {
        if(n==2) return 1;
        if(n==3) return 2;
        int product = 1;
        while(n>4){
            product*=3;
            n-=3;
        }
        product*=n;

        return product;
    }
}

"************************************"
320	Generalized Abbreviation
Write a function to generate the generalized abbreviations of a word.
Example:
Given word = "word", return the following list (order does not matter):
["word", "1ord", "w1rd", "wo1d", "wor1", "2rd", "w2d", "wo2", "1o1d", "1or1", "w1r1", "1o2", "2r1", "3d",
 "w3", "4"]

public class Solution {
    public List<String> generateAbbreviations(String word) {
        List<String> res = new ArrayList<String>();
        int len = word.length();
        res.add(len==0 ? "" : String.valueOf(len));
        for(int i = 0 ; i < len ; i++)
            for(String right : generateAbbreviations(word.substring(i+1))){
                String leftNum = i > 0 ? String.valueOf(i) : "";
                res.add( leftNum + word.substring(i,i + 1) + right );
            }
        return res;
    }
}

319	Bulb Switcher
There are n bulbs that are initially off. You first turn on all the bulbs. Then, you turn off every second
bulb. On the third round, you toggle every third bulb (turning on if it's off or turning off if it's on).
For the ith round, you toggle every i bulb. For the nth round, you only toggle the last bulb. Find how many
bulbs are on after n rounds.

Example:
Given n = 3. 

At first, the three bulbs are [off, off, off].
After first round, the three bulbs are [on, on, on].
After second round, the three bulbs are [on, off, on].
After third round, the three bulbs are [on, off, off]. 

So you should return 1, because there is only one bulb is on.

public class Solution {
    public int bulbSwitch(int n) {
        return (int)Math.sqrt(n);    
    }
}
other's solution:'
factor of 6: 1,2,3,6 factor of 7: 1,7 factor of 9: 1,3,9
so all number have even number of factors except square number(e.g: factor of 9:1,3,9). 
square number must turn on because of odd number of factors(9: turn on at 1st, off at 3rd, on at 9th) 
other number must turn off(6: turn on at 1st, off at 2nd, on at 3rd, off at 6th) so we only need to
compute the number of square number less equal than n

268	Missing Number
Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the one that is missing from
the array.

For example,
Given nums = [0, 1, 3] return 2.

public class Solution {
    public int missingNumber(int[] nums) {
        int sum = 0;
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
        }
        int s = nums.length * (nums.length + 1)/2;
        return s - sum;
    }
}

'*******************************'
144	Binary Tree Preorder Traversal
Given a binary tree, return the preorder traversal of its nodes' values.'

For example:
Given binary tree {1,#,2,3},
   1
    \
     2
    /
   3
return [1,2,3].

Note: Recursive solution is trivial, could you do it iteratively?

public class Solution {
    public List<Integer> preorderTraversal(TreeNode root) {
        // node first
        ArrayList<Integer> list = new ArrayList<>();
        if (root != null) {
            helper(root,list);    
        }
        return list;
    }
    public void helper(TreeNode root, List<Integer> list) {
        if (root == null) {
            return;
        }
        list.add(root.val);
        helper(root.left,list);
        helper(root.right,list);
    }
}
iteratively:
public class Solution {
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<Integer>();
        if (root == null) return result;
        Stack<TreeNode> stack = new Stack<TreeNode>();
        stack.push(root);
        while (!stack.isEmpty()){
            TreeNode node = stack.pop();
            result.add(node.val);
            if (node.right != null) stack.push(node.right);
            if (node.left != null) stack.push(node.left);
        }
        return result;
    }
}

94	Binary Tree Inorder Traversal
Given a binary tree, return the inorder traversal of its nodes' values.'

For example:
Given binary tree [1,null,2,3],
   1
    \
     2
    /
   3
return [1,3,2].
public class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        ArrayList<Integer> list = new ArrayList<>();
        if (root != null) {
            list.addAll(inorderTraversal(root.left));
            list.add(root.val);
            list.addAll(inorderTraversal(root.right));
        }
        return list;
    }
}

public List<Integer> inorderTraversal(TreeNode root) {
    List<Integer> res = new LinkedList<Integer>();
    if (root == null) return res;

    Stack<TreeNode> stack = new Stack<TreeNode>();
    TreeNode cur = root;
    while (cur != null || !stack.isEmpty()) { 
        while (cur != null) {// Travel to the left leaf
            stack.push(cur);
            cur = cur.left;             
        }        
        cur = stack.pop(); // Backtracking to higher level node A
        res.add(cur.val);  // Add the node to the result list
        cur = cur.right;   // Switch to A'right branch
    }
    return res;
}

"***********************************"
318	Maximum Product of Word Lengths
Given a string array words, find the maximum value of length(word[i]) * length(word[j]) where the two words
do not share common letters. You may assume that each word will contain only lower case letters. If no such
two words exist, return 0.

Example 1:
Given ["abcw", "baz", "foo", "bar", "xtfn", "abcdef"]
Return 16
The two words can be "abcw", "xtfn".

Example 2:
Given ["a", "ab", "abc", "d", "cd", "bcd", "abcd"]
Return 4
The two words can be "ab", "cd".

Example 3:
Given ["a", "aa", "aaa", "aaaa"]
Return 0
No such pair of words.

public class Solution {
    /**
     * @param words
     * @return
     * 
     *      The soultion is calcuated by doing a product of the length of
     *         each string to every other string. Anyhow the constraint given is
     *         that the two strings should not have any common character. This
     *         is taken care by creating a unique number for every string. Image
     *         a an 32 bit integer where 0 bit corresponds to 'a', 1st bit
     *         corresponds to 'b' and so on.
     * 
     *         Thus if two strings contain the same character when we do and
     *         "AND" the result will not be zero and we can ignore that case.
     */
    public int maxProduct(String[] words) {
        int[] checker = new int[words.length];
        int max = 0;
        // populating the checker array with their respective numbers
        for (int i = 0; i < checker.length; i++) {
            int num = 0;
            for (int j = 0; j < words[i].length(); j++) {
                num |= 1 << (words[i].charAt(j) - 'a');
            }
            checker[i] = num;
        }

        for (int i = 0; i < words.length; i++) {
            for (int j = i + 1; j < words.length; j++) {
                if ((checker[i] & checker[j]) == 0) //checking if the two strings have common character
                    max = Math.max(max, words[i].length() * words[j].length());
            }
        }
        return max;
    }

}


12	Integer to Roman
Given an integer, convert it to a roman numeral.
Input is guaranteed to be within the range from 1 to 3999.

public class Solution {
    public String intToRoman(int num) {
        // I V X L C D M
        int[] values = {1000,900,500,400,100,90,50,40,10,9,5,4,1};
        String[] strs = {"M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"};
    
        StringBuilder sb = new StringBuilder();
    
        for(int i=0;i<values.length;i++) {
            while(num >= values[i]) {
                num -= values[i];
                sb.append(strs[i]);
            }
        }
        return sb.toString();
    }
}

328	Odd Even Linked List
Given a singly linked list, group all odd nodes together followed by the even nodes. Please note here we
are talking about the node number and not the value in the nodes.

You should try to do it in place. The program should run in O(1) space complexity and O(nodes) time
complexity.

Example:
Given 1->2->3->4->5->NULL,
return 1->3->5->2->4->NULL.

Note:
The relative order inside both the even and odd groups should remain as it was in the input. 
The first node is considered odd, the second node even and so on ...

public class Solution {
    public ListNode oddEvenList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        
        ListNode odd = new ListNode(0);
        ListNode even = new ListNode(0);
        ListNode currOdd = odd;
        ListNode currEven = even;
        
        int count = 0;
        while (head != null) {
            if (count % 2 == 0) {
                currOdd.next = head;
                currOdd = currOdd.next;
            }else {
                currEven.next = head;
                currEven = currEven.next;
            }
            head = head.next;
            count++;
        }
        currOdd.next = even.next;
        currEven.next = null; // this statement is very important, or you will get memory exceed limits
        return odd.next;
    }
}

public class Solution {
    public ListNode oddEvenList(ListNode head) {
      if(head == null || head.next==null) {
        return head;
      }
      ListNode odd=head;
      ListNode even=head.next;
      ListNode evenori=head.next;
      
      while(even!=null&&even.next!=null) {
          odd.next=even.next;
          odd = even.next;
          even.next = odd.next;
          even = odd.next;
      }
      odd.next = evenori;
      
      return head;
    }
}

'*****************************'
156	Binary Tree Upside Down
Given a binary tree where all the right nodes are either leaf nodes with a sibling (a left node that
shares the same parent node) 
or empty,
flip it upside down and turn it into a tree where the original right nodes turned into left leaf nodes.
Return the new root.
For example:
Given a binary tree {1,2,3,4,5},
1
/ \
2  3
/ \
4 5
return the root of the binary tree [4,5,2,#,#,3,1].
4
/ \
5  2
  / \
 3   1
"my wrong answer"
public class Solution {
	public TreeNode upsideDown(TreeNode root) {
		if (root == null || root.left == null) {
			return root;
		}

		TreeNode newRoot = upsideDown(root.left);
		newRoot.left = root.right;
		newRoot.right = root;
		return newRoot;
	}
}

 public TreeNode UpsideDownBinaryTree(TreeNode root) {  
    if (root == null)  
        return null;  
    TreeNode parent = root, left = root.left, right = root.right;  
    if (left != null) {  
        TreeNode ret = UpsideDownBinaryTree(left);  
        left.left = right;  
        left.right = parent;  
        return ret;  
    }  
    return root;  
} 

"************************"
259	3Sum Smaller
Problem Description:
Given an array of n integers nums and a target, find the number of index triplets i, j, k 
with 0 <= i < j < k < n that 
satisfy the condition nums[i] + nums[j] + nums[k] < target.
For example, given nums = [-2, 0, 1, 3], and target = 2.
Return 2. Because there are two triplets which sums are less than 2:

[-2, 0, 1]
[-2, 0, 3]

public class Solution {
	public int threeSumSmaller(int[] nums, int target) {
	    if(nums.length < 3){return 0;}
	    int count = 0;
	    Arrays.sort(nums);
	    for(int i = 0; i < nums.length-2; i++){
	        if(nums[i]*3 >= target){break;}
	        count += find(nums, target-nums[i], i+1, nums.length-1);
	    }
	    return count;
	}


	//find number of pair that sum up smaller than target from given part of array
	public int find(int[] nums, int target, int start, int end){
	    int count = 0;
	    while(start < end){
	        if(nums[start] + nums[end] >= target){
	            end--;
	        }else{
	            count += end-start;
	            start++;
	        }
	    }
	    return count;
	}
}

230	Kth Smallest Element in a BST
Given a binary search tree, write a function kthSmallest to find the kth smallest element in it.
Note: 
You may assume k is always valid, 1 ≤ k ≤ BST's total elements.'

Follow up:
What if the BST is modified (insert/delete operations) often and you need to find the kth smallest
frequently? How would you optimize the kthSmallest routine?

public class Solution {
    public int kthSmallest(TreeNode root, int k) {
        if (count(root) == k) {
            while (root.right != null) {
                root = root.right;
            }
            return root.val;
        }
        if (count(root.left) < k) {
            k = k - count(root.left);
            if (k == 1) { // [1, null, 2] 1
                return root.val;
            }
            return kthSmallest(root.right, k-1);
        }
        return kthSmallest(root.left, k);
    }
    
    public int count(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return 1 + count(root.left) + count(root.right);
    }
}

other solution:
public int kthSmallest(TreeNode root, int k) {
    int count = countNodes(root.left);
    if (k <= count) {
        return kthSmallest(root.left, k);
    } else if (k > count + 1) {
        return kthSmallest(root.right, k-1-count); // 1 is counted as current node
    }

    return root.val;
	}

	public int countNodes(TreeNode n) {
	    if (n == null) return 0;

	    return 1 + countNodes(n.left) + countNodes(n.right);
	}
}

137	Single Number II
Given an array of integers, every element appears three times except for one. Find that single one.
Note:
Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?

public class Solution {
    public int singleNumber(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        
        for (int i = 0; i < nums.length; i++) {
            if (map.get(nums[i]) == null) {
                map.put(nums[i], 1);
            }else {
                if (map.get(nums[i]) + 1 == 3) {
                    map.remove(nums[i]);
                }else {
                    map.put(nums[i],map.get(nums[i])+1);
                }
            }
        }
        
        for (int i:map.keySet()) {
            return i;
        }
        return nums[0];
    }
}

public class Solution {
    public int singleNumber(int[] A) {
        if (A == null) return 0;
        int x0 = ~0, x1 = 0, x2 = 0, t;
        for (int i = 0; i < A.length; i++) {
            t = x2;
            x2 = (x1 & A[i]) | (x2 & ~A[i]);
            x1 = (x0 & A[i]) | (x1 & ~A[i]);
            x0 = (t & A[i]) | (x0 & ~A[i]);
        }
        return x1;
    }
}

Consider the following fact:

Write all numbers in binary form, then for any bit 1 that appeared 3*n times (n is an integer), the bit can
only present in numbers that appeared 3 times

e.g. 0010 0010 0010 1011 1011 1011 1000 (assuming 4-bit integers) 2(0010) and 11(1011) appeared 3 times, 
and digit counts are:

Digits 3 2 1 0
Counts 4 0 6 3
Counts%3 1 0 0 0
public int singleNumber(int[] nums) {
    int[] digits = new int[32];
    for(int i=0; i<nums.length; i++){
        int mask = 1;
        for(int j=31; j>=0; j--){
            if((mask & nums[i])!=0)
                digits[j]++;
            mask <<= 1;
        }
    }
    int res = 0;
    for(int i=0; i<32; i++){
        if(digits[i]%3==1)
            res += 1;
        if(i==31)
            continue;
        res <<= 1;
    }
    return res;
}


286	Walls and Gates
You are given a m x n 2D grid initialized with these three possible values.

-1 - A wall or an obstacle.
0 - A gate.
INF - Infinity means an empty room. We use the value 231 - 1 = 2147483647 to represent INF as you may
assume that the distance to a 
gate is less than 2147483647.
Fill each empty room with the distance to its nearest gate. If it is impossible to reach a gate, it should
be filled with INF.

For example, given the 2D grid:
INF -1 0 INF
INF INF INF -1
INF -1 INF -1
0 -1 INF INF
After running your function, the 2D grid should be:
3 -1 0 1
2 2 1 -1
1 -1 2 -1
0 -1 3 4

public class Solution {
    public void wallsAndGates(int[][] rooms) {
        for(int i=0;i<rooms.length;i++)
            for(int j=0;j<rooms[0].length;j++)
                if(rooms[i][j]==0)
                    helper(rooms,i,j,0);
    }
    
    public void helper(int[][] rooms,int i,int j,int d){
        if(i<0||i>=rooms.length||j<0||j>=rooms[0].length||rooms[i][j]<d)
            return;
        rooms[i][j]=d;
        helper(rooms,i-1,j,d+1);
        helper(rooms,i+1,j,d+1);
        helper(rooms,i,j+1,d+1);
        helper(rooms,i,j-1,d+1);
    }
}


96	Unique Binary Search Trees
Given n, how many structurally unique BST's (binary search trees) that store values 1...n?

For example,
Given n = 3, there are a total of 5 unique BST's.

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3

public class Solution {
    public int numTrees(int n) {
        if (n <= 2) {
            return n;
        }
        
        int[] a = new int[n+1];
        a[0] = 1;
        a[1] = 1;
        a[2] = 2;
        for (int i = 3; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                a[i] += a[j-1]*a[i-j];
            }
        }
        return a[n];
    }
}

'************************'
337	House Robber III
The thief has found himself a new place for his thievery again. There is only one entrance to this area,
called the "root." Besides the root, each house has one and only one parent house. After a tour, the smart
thief realized that "all houses in this place forms a binary tree". It will automatically contact the
police if two directly-linked houses were broken into on the same night.

Determine the maximum amount of money the thief can rob tonight without alerting the police.

Example 1:
     3
    / \
   2   3
    \   \ 
     3   1
Maximum amount of money the thief can rob = 3 + 3 + 1 = 7.
Example 2:
     3
    / \
   4   5
  / \   \ 
 1   3   1
Maximum amount of money the thief can rob = 4 + 5 = 9.

dfs all the nodes of the tree, each node return two number, int[] num, num[0] is the max value while rob
this node, num[1] is max value while not rob this value. Current node return value only depend on its
children's value. Transform function should be very easy to understand.

public class Solution {
    public int rob(TreeNode root) {
        int[] num = dfs(root);
        return Math.max(num[0], num[1]);
    }
    private int[] dfs(TreeNode x) {
        if (x == null) return new int[2];
        int[] left = dfs(x.left);
        int[] right = dfs(x.right);
        int[] res = new int[2];
        res[0] = left[1] + right[1] + x.val;
        res[1] = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);
        return res;
    }
}

108	Convert Sorted Array to Binary Search Tree
Given an array where elements are sorted in ascending order, convert it to a height balanced BST.
public class Solution {
    public TreeNode sortedArrayToBST(int[] nums) {
        if (nums.length == 0) {
            return null;
        }
        
        TreeNode root = new TreeNode(nums[nums.length/2]);
        helper(root, nums, 0, nums.length-1);
        return root;
    }
    public void helper(TreeNode root, int[] nums, int lo, int hi) {
        if (lo > hi) {
            return;
        }
        System.out.println(root.val);
        int m = lo + (hi-lo)/2;
        root.val = nums[m];
        if (lo <= m-1) {
            root.left = new TreeNode(0);
            helper(root.left, nums, lo, m-1);

        }
        if (m+1 <= hi) {
            root.right = new TreeNode(0);
            helper(root.right, nums, m+1, hi);
        }

    }
}

my another solution:
public class Solution {
    public TreeNode sortedArrayToBST(int[] nums) {
      return bst(nums, 0, nums.length-1);
    }
    
    public TreeNode bst(int[] nums, int low, int high) {
      if(nums==null || low>high) {
          return null;
      }
      int mid = low+(high-low)/2;
      TreeNode t = new TreeNode(nums[mid]);
      t.left = bst(nums, low, mid-1);
      t.right = bst(nums, mid+1, high);
      return t;
    }

}

35	Search Insert Position
Given a sorted array and a target value, return the index if the target is found. If not, return the index
where it would be if it were inserted in order.

You may assume no duplicates in the array.

Here are few examples.
[1,3,5,6], 5 → 2
[1,3,5,6], 2 → 1
[1,3,5,6], 7 → 4
[1,3,5,6], 0 → 0

public class Solution {
    public int searchInsert(int[] nums, int target) {
        if (nums.length == 0) {
            return 0;
        }
        int index = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == target) {
                index = i;
                return index;
            }
            if (nums[i] < target) {
                index++;
            }else {
                index = i;
                return index;
            }
        }
        return index;
    }
}

public class Solution {
    public int searchInsert(int[] nums, int target) {
        if (nums.length == 0) {
            return 0;
        }
        
        int lo = 0;
        int hi = nums.length - 1;
        
        while (lo <= hi) {
            int m = lo + (hi-lo)/2;
            if (nums[m] == target) {
                return m;
            }
            if (nums[m] > target) {
                hi = m - 1; 
            }else {
                lo = m + 1;
            }
        }
        return lo;
    }
}


"***************************"
22	Generate Parentheses
Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

For example, given n = 3, a solution set is:

[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]

public class Solution {
    public List<String> generateParenthesis(int n) {
        ArrayList<String> list = new ArrayList<>();
        if (n == 0) {
            return list;
        }
        Set<String> set = new HashSet<>();
        set.add("()");
        for (int i = 2; i <= n; i++) {
            Set<String> tmp = new HashSet<>();
            for (String s:set) {
                for (int j = 0; j < s.length(); j++) {
                    tmp.add(s.substring(0,j) + "()" + s.substring(j,s.length()));
                }
            }
            set = tmp;
        }
        for (String s : set) {
            list.add(s);
        }
        return list;
    }
}

"**************************************************"
255	Verify Preorder Sequence in Binary Search Tree 
Given an array of numbers, verify whether it is the correct preorder traversal sequence of a binary search
tree.
You may assume each number in the sequence is unique.

Follow up:
Could you do it using only constant space complexity?

Kinda simulate the traversal, keeping a stack of nodes (just their values) of which we're '
still in the left subtree. If the next number is smaller than the last stack value, then we're 'still in
the left subtree of all stack nodes, so just push the new one onto the stack. But before that, pop all
smaller ancestor values, as we must now be in their right subtrees (or even further, in the right subtree
of an ancestor). Also, use the popped values as a lower bound, since being in their right subtree means we
must never come across a smaller number anymore.

public boolean verifyPreorder(int[] preorder) {
    int low = Integer.MIN_VALUE;
    Stack<Integer> path = new Stack();
    for (int p : preorder) {
        if (p < low)
            return false;
        while (!path.empty() && p > path.peek())
            low = path.pop();
        path.push(p);
    }
    return true;
}
Solution 2 ... O(1) extra space

Same as above, but abusing the given array for the stack.

public boolean verifyPreorder(int[] preorder) {
    int low = Integer.MIN_VALUE, i = -1;
    for (int p : preorder) {
        if (p < low)
            return false;
        while (i >= 0 && p > preorder[i])
            low = preorder[i--];
        preorder[++i] = p;
    }
    return true;
}


298	Binary Tree Longest Consecutive Sequence
Given a binary tree, find the length of the longest consecutive sequence path.
The path refers to any sequence of nodes from some starting node to any node in the tree along the
parent-child connections.
The longest consecutive path need to be from parent to child (cannot be the reverse).

For example,

   1
    \
     3
    / \
   2   4
        \
         5
Longest consecutive sequence path is 3-4-5, so return 3.

   2
    \
     3
    /
   2
  /
 1
Longest consecutive sequence path is 2-3,not3-2-1, so return 2.

public class Solution {
    int max = 0;
    public int longestConsecutive(TreeNode root) {
        if(root==null) return 0;
        helper(root,0,root.val);
        return max;
    }
    
    public void helper(TreeNode root,int cur, int target){
        if(root==null) return;
        if(root.val==target) cur++;
        else cur=1;
        max = Math.max(cur,max);
        helper(root.left, cur, root.val+1);
        helper(root.right, cur, root.val+1);
    }
}

'************************************************'
309	Best Time to Buy and Sell Stock with Cooldown
Say you have an array for which the ith element is the price of a given stock on day i.
Design an algorithm to find the maximum profit. You may complete as many transactions as you like
(ie, buy one and sell one share of the stock multiple times) with the following restrictions:

You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy
again).
After you sell your stock, you cannot buy stock on next day. (ie, cooldown 1 day)
Example:

prices = [1, 2, 3, 0, 2]
maxProfit = 3
public class Solution {
    public int maxProfit(int[] prices) {
        if (prices.length <= 1) {
            return 0;
        }
        
        int b0 = -prices[0];
        int b1 = b0;
        int s2, s1, s0;
        s2 = s1 = s0 = 0;
        
        for (int i = 1; i < prices.length; i++) {
            b0 = Math.max(b1, s2-prices[i]);
            s0 = Math.max(s1, b1+prices[i]);
            b1 = b0;
            s2 = s1; 
            s1 = s0;
        }
        return s0;
    }
}
transactions = [buy, sell, cooldown, buy, sell]
buy[i] = Math.max(buy[i - 1], sell[i - 2] - prices[i]);   
sell[i] = Math.max(sell[i - 1], buy[i - 1] + prices[i]);
3. Optimize to O(1) Space
Let b2, b1, b0 represent buy[i - 2], buy[i - 1], buy[i]
Let s2, s1, s0 represent sell[i - 2], sell[i - 1], sell[i]

"*************************"
53	Maximum Subarray
Find the contiguous subarray within an array (containing at least one number) which has the largest sum.
For example, given the array [−2,1,−3,4,−1,2,1,−5,4],
the contiguous subarray [4,−1,2,1] has the largest sum = 6.

public class Solution {
    public int maxSubArray(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        int[] a = new int[nums.length];
        a[0] = nums[0];
        int max = a[0];
        
        for (int i = 1; i < nums.length; i++) {
            a[i] = Math.max(nums[i], a[i-1] + nums[i]);
            max = Math.max(a[i], max);
        }
        return max;
    }
}
//a[i]  the max value including current value;


250	Count Univalue Subtrees
Given a binary tree, count the number of uni-value subtrees.
A Uni-value subtree means all nodes of the subtree have the same value.

For example:
Given binary tree,
              5
             / \
            1   5
           / \   \
          5   5   5
return 4.

//3 leafs and 5-5

// my solution is wrong
other solution:
public class Solution {
    int count = 0;
    public int countUnivalSubtrees(TreeNode root) {
        helper(root,0);
        return count;
    }
    
    boolean helper(TreeNode root, int val){
        if(root==null)
            return true;
        if(!helper(root.left,root.val) | !helper(root.right,root.val))
            return false;
        count++;
        return root.val == val;
    }
}

89	Gray Code
The gray code is a binary numeral system where two successive values differ in only one bit.
Given a non-negative integer n representing the total number of bits in the code, print the sequence of
gray code. A gray code sequence must begin with 0.

For example, given n = 2, return [0,1,3,2]. Its gray code sequence is:

00 - 0
01 - 1
11 - 3
10 - 2

public class Solution {
    public List<Integer> grayCode(int n) {
        ArrayList<Integer> list = new ArrayList<>();
        ArrayList<String> bits = new ArrayList<>();
        if (n == 0) {
            list.add(0);
            return list;
        }
        bits.add("0");
        bits.add("1");
        int i = 2;
        while (i <= n) {
            ArrayList<String> tmp = new ArrayList<>();
            for (int j = 0; j < bits.size(); j++) {
                tmp.add("0"+bits.get(j));
            }
            for (int j = bits.size()-1; j >= 0; j--) {
                tmp.add("1"+bits.get(j));
            }
            bits = tmp;
            i++;
        }
        for (i = 0; i < bits.size(); i++) {
            list.add(Integer.parseInt(bits.get(i), 2));
        }
        return list;
    }
}

//Integer.parseInt("0001",2);

62	Unique Paths
A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).
The robot can only move either down or right at any point in time. The robot is trying to reach the
bottom-right corner of the grid (marked 'Finish' in the diagram below).

How many possible unique paths are there?

public class Solution {
    public int uniquePaths(int m, int n) {
        long result = 1;
        
        for(int i = 1; i <= Math.min(m-1, n-1); i++) {
            result = result * (m+n-1-i) / i;
        }
        return (int) result;
    }
}
// remember m-1, n-1, m+n-1
Here is my understanding of the code: Array dp stores the number of paths which passing this point. 
The whole algorithm is to sum up the paths from left grid and up grid. 'if (row[j] == 1) dp[j] = 0;'
means if there is an obstacle at this point. All the paths passing this point will no longer valid. 
In other words, the grid right of the obstacle can be reached only by the grid which lies up to it. 
Hope it helps.


153	Find Minimum in Rotated Sorted Array
Suppose a sorted array is rotated at some pivot unknown to you beforehand.
(i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).

Find the minimum element.
You may assume no duplicate exists in the array.
public class Solution {
    public int findMin(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        int lo = 0;
        int hi = nums.length-1;
        if (nums[lo] <= nums[hi]) {
            return nums[lo];
        }
        while (lo < hi-1) {
            int m = lo + (hi-lo)/2;
            if (nums[m] > nums[lo]) {
                lo = m;
            }else {
                hi = m;
            }
        }
        return Math.min(nums[lo], nums[hi]);
    }
}
// the key point is determine go left or right when nums[m] > nums[lo] --->>

81	Search in Rotated Sorted Array II
Follow up for "Search in Rotated Sorted Array":
What if duplicates are allowed?
Would this affect the run-time complexity? How and why?
Write a function to determine if a given target is in the array.


116	Populating Next Right Pointers in Each Node
Given a binary tree
    struct TreeLinkNode {
      TreeLinkNode *left;
      TreeLinkNode *right;
      TreeLinkNode *next;
    }
Populate each next pointer to point to its next right node. If there is no next right node, the next
pointer should be set to NULL.

Initially, all next pointers are set to NULL.

Note:

You may only use constant extra space.
You may assume that it is a perfect binary tree (ie, all leaves are at the same level, and every parent
has two children).
For example,
Given the following perfect binary tree,
         1
       /  \
      2    3
     / \  / \
    4  5  6  7
After calling your function, the tree should look like:
         1 -> NULL
       /  \
      2 -> 3 -> NULL
     / \  / \
    4->5->6->7 -> NULL

public class Solution {
    public void connect(TreeLinkNode root) {
        if (root == null || root.left == null) {
            return;
        }

        helper(root.left, root.right);
    }
    
    public void helper(TreeLinkNode left, TreeLinkNode right) {
        if (left == null) {
            return;
        }
        left.next = right;
        helper(left.left,left.right);
        helper(left.right,right.left);
        helper(right.left,right.right);
    }
}

216	Combination Sum III
Find all possible combinations of k numbers that add up to a number n, given that only numbers from 1 to 9
can be used and each combination should be a unique set of numbers.

Example 1:
Input: k = 3, n = 7

Output:
[[1,2,4]]

Example 2:
Input: k = 3, n = 9

Output:
[[1,2,6], [1,3,5], [2,3,4]]

public class Solution {
	public List<List<Integer>> combinationSum3(int k, int n) {
		ArrayList<List<Integer>> lists = new ArrayList<List<Integer>>();
		helper(lists, new ArrayList<Integer>(), 1, k, n);
		return lists;
	}

	public void helper(ArrayList<List<Integer>> lists, List<Integer> list, int start, int k, int n) {
		if (list.size() == k) {
		    if (n == 0) {
			    lists.add(new ArrayList<>(list));// why need to new ArrayList<>(list), or just get two numbers;
			    return;
		    }
		    return;
		}
		ArrayList<Integer> tmp = new ArrayList<>(list);
		for (int i = start; i <= 9; i++) {
		    tmp.add(i);
			helper(lists, tmp, i+1, k, n-i);
			tmp.remove(tmp.size()-1);
		}
	}
}

46	Permutations
Given a collection of distinct numbers, return all possible permutations.

For example,
[1,2,3] have the following permutations:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]
public class Solution {
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> lists = new ArrayList<List<Integer>>();
        if (nums.length == 0) {
            return lists;
        }
        
        List<Integer> list = new ArrayList<>();
        list.add(nums[0]);
        lists.add(list);
        for (int i = 1; i < nums.length; i++) {
            ArrayList<List<Integer>> tmps = new ArrayList<List<Integer>>();
            for (int j = 0; j <= i; j++) {
                for (List<Integer> l:lists) {
                    List<Integer> new_l = new ArrayList<Integer>(l);
                    new_l.add(j,nums[i]);
                    tmps.add(new_l);
                }    
            }
            lists = tmps;
        }
        return lists;
    }
}

'************************************'
241	Different Ways to Add Parentheses
Given a string of numbers and operators, return all possible results from computing all the different
possible ways to group numbers and operators. The valid operators are +, - and *.

Example 1
Input: "2-1-1".
((2-1)-1) = 0
(2-(1-1)) = 2
Output: [0, 2]

Example 2
Input: "2*3-4*5"

(2*(3-(4*5))) = -34
((2*3)-(4*5)) = -14
((2*(3-4))*5) = -10
(2*((3-4)*5)) = -10
(((2*3)-4)*5) = 10
Output: [-34, -14, -10, -10, 10]

public class Solution {
    public List<Integer> diffWaysToCompute(String input) {
        List<Integer> ret = new LinkedList<Integer>();
        for (int i=0; i<input.length(); i++) {
            if (input.charAt(i) == '-' ||
                input.charAt(i) == '*' ||
                input.charAt(i) == '+' ) {
                String part1 = input.substring(0, i);
                String part2 = input.substring(i+1);
                List<Integer> part1Ret = diffWaysToCompute(part1);
                List<Integer> part2Ret = diffWaysToCompute(part2);
                for (Integer p1 :   part1Ret) {
                    for (Integer p2 :   part2Ret) {
                        int c = 0;
                        switch (input.charAt(i)) {
                            case '+': c = p1+p2;
                                break;
                            case '-': c = p1-p2;
                                break;
                            case '*': c = p1*p2;
                                break;
                        }
                        ret.add(c);
                    }
                }
            }
        }
        if (ret.size() == 0) {
            ret.add(Integer.valueOf(input));
        }
        return ret;
    }
}

285	Inorder Successor in BST
Given a binary search tree and a node in it, find the in-order successor of that node in the BST.
Note: If the given node has no in-order successor in the tree, return null.
public class Solution {
    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        TreeNode res = null;
        while(root!=null){
            if(root.val > p.val){
                res = root;
                root = root.left;
            }
            else
                root = root.right;
        }
        return res;
    }
}


59	Spiral Matrix II
Given an integer n, generate a square matrix filled with elements from 1 to n2 in spiral order.

For example,
Given n = 3,

You should return the following matrix:
[
 [ 1, 2, 3 ],
 [ 8, 9, 4 ],
 [ 7, 6, 5 ]
]

public class Solution {
    public int[][] generateMatrix(int n) {
        int[][] res = new int[n][n];
        int cur = 1;
        int rb = 0;
        int re = n-1;
        int cb = 0;
        int ce = n-1;
        while(cur<=n*n){
            int i,j;
            for(j=cb;j<=ce;j++){
                res[rb][j]=cur++;
            }
            rb++;
            for(i=rb;i<=re;i++){
                res[i][ce]=cur++;
            }
            ce--;
            for(j=ce;j>=cb;j--){
                res[re][j]=cur++;
            }
            re--;
            for(i=re;i>=rb;i--){
                res[i][cb]=cur++;
            }
            cb++;
        }
        return res;
    }
}

244	Shortest Word Distance II
This is a follow up of Shortest Word Distance. The only difference is now you are given the list of words
and your method will be called repeatedly many times with different parameters. How would you optimize it?

Design a class which receives a list of words in the constructor, and implements a method that takes two
words word1 and word2 and return the shortest distance between these two words in the list.

For example,
Assume that words = ["practice", "makes", "perfect", "coding", "makes"].
Given word1 = “coding”, word2 = “practice”, return 3.
Given word1 = "makes", word2 = "coding", return 1.

Note:
You may assume that word1 does not equal to word2, and word1 and word2 are both in the list.

    HashMap<String, List<Integer>> map = new HashMap<String, List<Integer>>();

    public WordDistance(String[] words) {
        // 统计每个单词出现的下标存入哈希表中
        for(int i = 0; i < words.length; i++){
            List<Integer> cnt = map.get(words[i]);
            if(cnt == null){
                cnt = new ArrayList<Integer>();
            }
            cnt.add(i);
            map.put(words[i], cnt);
        }
    }

    public int shortest(String word1, String word2) {
        List<Integer> idx1 = map.get(word1);
        List<Integer> idx2 = map.get(word2);
        int distance = Integer.MAX_VALUE;
        int i = 0, j = 0;
        // 每次比较两个下标列表最小的下标，然后把跳过较小的那个
        while(i < idx1.size() && j < idx2.size()){
            distance = Math.min(Math.abs(idx1.get(i) - idx2.get(j)), distance);
            if(idx1.get(i) < idx2.get(j)){
                i++;
            } else {
                j++;
            }
        }
        return distance;
    }


254	Factor Combinations
Problem:
Numbers can be regarded as product of its factors. For example, 

8 = 2 x 2 x 2;
  = 2 x 4.
Write a function that takes an integer n and return all possible combinations of its factors. 

Note: 

Each combination's' factors must be sorted ascending, for example: The factors of 2 and 6 is [2, 6], 
not [6, 2]. 
You may assume that n is always positive. 
Factors should be greater than 1 and less than n.
 

Examples: 
input: 1
output: []

input: 37
output: []
input: 12
output:
[
  [2, 6],
  [2, 2, 3],
  [3, 4]
]
input: 32
output:

[
  [2, 16],
  [2, 2, 8],
  [2, 2, 2, 4],
  [2, 2, 2, 2, 2],
  [2, 4, 4],
  [4, 8]
]

public class Solution {
    public List<List<Integer>> getFactors(int n) {
        List<List<Integer>> ret = new ArrayList<List<Integer>> ();
        helper(ret, new ArrayList<Integer> (), n, 2);
        return ret;
    }
    
    private void helper(List<List<Integer>> ret, List<Integer> item, int n, int start) {
        if (n == 1) {
            if (item.size() > 1) {
                ret.add(new ArrayList<Integer> (item));
            }
            return;
        }
        for (int i = start; i <= n; i++) {
            if (n % i == 0) {
                item.add(i);
                helper(ret, item, n/i, i);
                item.remove(item.size()-1);
            }
        }
    }
}


public List<List<Integer>> getFactors(int n) {
    List<List<Integer>> result = new ArrayList<List<Integer>>();
    if (n <= 3) return result;
    helper(n, -1, result, new ArrayList<Integer>());
    return result; 
}

public void helper(int n, int lower, List<List<Integer>> result, List<Integer> cur) {
    if (lower != -1) {
        cur.add(n);
        result.add(new ArrayList<Integer>(cur));
        cur.remove(cur.size() - 1);
    }
    int upper = (int) Math.sqrt(n);
    for (int i = Math.max(2, lower); i <= upper; ++i) {
        if (n % i == 0) {
            cur.add(i);
            helper(n / i, i, result, cur);
            cur.remove(cur.size() - 1);
        }
    }
}


199	Binary Tree Right Side View
Given a binary tree, imagine yourself standing on the right side of it, return the values of the nodes
you can see ordered from top to bottom.

For example:
Given the following binary tree,
   1            <---
 /   \
2     3         <---
 \     \
  5     4       <---
You should return [1, 3, 4].

public class Solution {
    public List<Integer> rightSideView(TreeNode root) {
    // use queue();
        List<Integer> list = new ArrayList<>();
        if (root == null) {
            return list;
        }
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            Queue<TreeNode> sameLevel = new LinkedList<>();
            TreeNode tmp = null;
            while(!q.isEmpty()) {
                tmp = q.poll();
                if (tmp.left != null) {
                    sameLevel.offer(tmp.left);
                }
                if (tmp.right != null) {
                    sameLevel.offer(tmp.right);
                }
                
            }
            q = sameLevel;
            list.add(tmp.val);
        }
        return list;
    }
}

64	Minimum Path Sum
Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right which
minimizes the sum of all numbers along its path.

public class Solution {
    public int minPathSum(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 && j != 0) {
                    grid[i][j] = grid[i][j] + grid[i][j - 1];
                } else if (i != 0 && j == 0) {
                    grid[i][j] = grid[i][j] + grid[i - 1][j];
                } else if (i == 0 && j == 0) {
                    grid[i][j] = grid[i][j];
                } else {
                    grid[i][j] = Math.min(grid[i][j - 1], grid[i - 1][j])+grid[i][j];
                }
            }
        }
        return grid[m-1][n-1];
    }
}

253	Meeting Rooms II
Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei),
find the minimum number of conference rooms required.

For example, Given [[0, 30],[5, 10],[15, 20]], return 2.

public class Solution {
    public int minMeetingRooms(Interval[] intervals) {
        if(intervals == null || intervals.length == 0) return 0;
        Arrays.sort(intervals, new Comparator<Interval>(){
            public int compare(Interval i1, Interval i2){
                return i1.start - i2.start;
            }
        });
        // 用堆来管理房间的结束时间
        PriorityQueue<Integer> endTimes = new PriorityQueue<Integer>();
        endTimes.offer(intervals[0].end);
        for(int i = 1; i < intervals.length; i++){
            // 如果当前时间段的开始时间大于最早结束的时间，则可以更新这个最早的结束时间为当前时间段的结束时间，如果小于的话，就加入一个新的结束时间，表示新的房间
            if(intervals[i].start >= endTimes.peek()){
                endTimes.poll();
            }
            endTimes.offer(intervals[i].end);
        }
        // 有多少结束时间就有多少房间
        return endTimes.size();
    }
}

173	Binary Search Tree Iterator
Implement an iterator over a binary search tree (BST). Your iterator will be initialized with the root node
of a BST.
Calling next() will return the next smallest number in the BST.
Note: next() and hasNext() should run in average O(1) time and uses O(h) memory, where h is the height of
the tree.
public class BSTIterator {
    Stack<TreeNode> s = new Stack<>();
    private int small = 0;
    public BSTIterator(TreeNode root) {
        while (root != null) {
            s.push(root);
            root = root.left;
        }   
    }

    /** @return whether we have a next smallest number */
    public boolean hasNext() {
        if (s.isEmpty()) {
            return false;
        }
        TreeNode tmp = s.peek();
        s.pop();
        small = tmp.val;
        if (tmp.right != null) {
            tmp = tmp.right;
            s.push(tmp);
            while (tmp.left != null) {
                s.push(tmp.left);
                tmp = tmp.left;
            }
        }
        return true;
    }

    /** @return the next smallest number */
    public int next() {
        return small;    
    }
}

/**
 * Your BSTIterator will be called like this:
 * BSTIterator i = new BSTIterator(root);
 * while (i.hasNext()) v[f()] = i.next();
 */

48	Rotate Image
You are given an n x n 2D matrix representing an image.
Rotate the image by 90 degrees (clockwise).

Follow up:
Could you do this in-place?

* clockwise rotate
* first reverse up to down, then swap the symmetry 
* 1 2 3     7 8 9     7 4 1
* 4 5 6  => 4 5 6  => 8 5 2
* 7 8 9     1 2 3     9 6 3

* anticlockwise rotate
* first reverse left to right, then swap the symmetry
* 1 2 3     3 2 1     3 6 9
* 4 5 6  => 6 5 4  => 2 5 8
* 7 8 9     9 8 7     1 4 7

public class Solution {
    public void rotate(int[][] matrix) {
        int len = matrix.length;
        reverse(matrix);
        for(int i=0;i<len;i++)
            for(int j=i+1;j<len;j++){
                int temp;
                temp = matrix[i][j];
                matrix[i][j]=matrix[j][i];
                matrix[j][i]=temp;
            }
    }
    
    public void reverse(int[][] matrix){
        int len = matrix.length;
        for(int i=0;i<len/2;i++){
            int[] temp = new int[len];
            temp = matrix[i];
            matrix[i]=matrix[len-1-i];
            matrix[len-1-i]=temp;
        }
    }
}

"*********************************"
313	Super Ugly Number
Write a program to find the nth super ugly number.
Super ugly numbers are positive numbers whose all prime factors are in the given prime list primes of size
k. For example, [1, 2, 4, 7, 8, 13, 14, 16, 19, 26, 28, 32] is the sequence of the first 12 super ugly
numbers given primes = [2, 7, 13, 19] of size 4.

Note:
(1) 1 is a super ugly number for any given primes.
(2) The given numbers in primes are in ascending order.
(3) 0 < k ≤ 100, 0 < n ≤ 106, 0 < primes[i] < 1000.

public int nthSuperUglyNumber(int n, int[] primes) {
    int[] ret = new int[n];
          ret[0] = 1;

    int[] indexes  = new int[primes.length];

    for(int i = 1; i < n; i++){
        ret[i] = Integer.MAX_VALUE;

        for(int j = 0; j < primes.length; j++){
            ret[i] = Math.min(ret[i], primes[j] * ret[indexes[j]]);
        }

        for(int j = 0; j < indexes.length; j++){
            if(ret[i] == primes[j] * ret[indexes[j]]){
                indexes[j]++;
            }
        }
    }

    return ret[n - 1];
}
// Basic idea is same as ugly number II, new ugly number is generated by multiplying a prime with previous
// generated ugly number. One catch is need to remove duplicate

11	Container With Most Water
Given n non-negative integers a1, a2, ..., an, where each represents a point at coordinate (i, ai). 
n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). Find two lines, 
which together with x-axis forms a container, such that the container contains the most water.


                              |    ***         |                 
               |              |    ***         |                    
               |    |    |    |    ***    |    |                    
          |    |    |    |    |    ***    |    |                     
          |    |    |    |    |    ***    |    |              |       
     |    |    |    |    |    |    ***    |    |              |       
     |    |    |    |    |    |    ***    |    |              |    |   
     |    |    |    |    |    |    ***    |    |    |         |    |   
     |    |    |    |    |    |    ***    |    |    |    |    |    |   
    lo                                              @              hi

public int maxArea(int[] height) {
    int L = height.length, lo = 0, hi = L-1;
    int max = 0;
    while(lo<hi) {    
        int loMax = height[lo], hiMax = height[hi];      

        int candidate = (hi-lo) * (loMax<hiMax ? loMax : hiMax);
        max = candidate > max ? candidate : max;

        if(height[lo]<=height[hi]) 
            while(lo<hi && height[lo]<=loMax) ++lo; 
        else 
            while(hi>lo && height[hi]<=hiMax) --hi;
    }
    return max;
}


240	Search a 2D Matrix II
Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following
properties:

Integers in each row are sorted in ascending from left to right.
Integers in each column are sorted in ascending from top to bottom.
For example,

Consider the following matrix:

[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
Given target = 5, return true.

Given target = 20, return false.

public class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        if(matrix == null || matrix.length < 1 || matrix[0].length <1) {
            return false;
        }
        int col = matrix[0].length-1;
        int row = 0;
        while(col >= 0 && row <= matrix.length-1) {
            if(target == matrix[row][col]) {
                return true;
            } else if(target < matrix[row][col]) {
                col--;
            } else if(target > matrix[row][col]) {
                row++;
            }
        }
        return false;
    }
}

75	Sort Colors
Given an array with n objects colored red, white or blue, sort them so that objects of the same color
are adjacent, with the colors in the order red, white and blue.

Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue respectively.

public class Solution {
    public void sortColors(int[] nums) {
        int[] tmps = new int[nums.length];
        Arrays.fill(tmps,1);
        int st = 0;
        int ed = nums.length - 1;
        
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == 0) {
                tmps[st++] = 0;
            }
            if (nums[i] == 2) {
                tmps[ed--] = 2;
            }
        }
        for (int i = 0; i < nums.length; i++) {
            nums[i] = tmps[i];
        }
        System.out.println(Arrays.toString(tmps));
    }
}


277	Find the Celebrity
Suppose you are at a party with n people (labeled from 0 to n - 1) and among them, there may exist
one celebrity.
The definition of a celebrity is that all the other n - 1 people know him/her but he/she does not
know any of them.

Now you want to find out who the celebrity is or verify that there is not one. The only thing you
are allowed to do is to ask questions
like: “Hi, A. Do you know B?” to get information of whether A knows B. You need to find out the
celebrity (or verify there is not one)
by asking as few questions as possible (in the asymptotic sense).

You are given a helper function bool knows(a, b) which tells you whether A knows B. Implement a
function int findCelebrity(n), your
function should minimize the number of calls to knows.

public class Solution extends Relation {
    public int findCelebrity(int n) {
        int candidate=0;
        for(int i=1;i<n;i++)
            if(knows(candidate,i))
                candidate = i;
        for(int i=0;i<n;i++)
            if(i!=candidate&&(knows(candidate,i) || !knows(i,candidate)))
                return -1;
        return candidate;
    }
}


77	Combinations
Given two integers n and k, return all possible combinations of k numbers out of 1 ... n.

For example,
If n = 4 and k = 2, a solution is:

[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]

public class Solution {
    public List<List<Integer>> combine(int n, int k) {
        ArrayList<List<Integer>> lists = new ArrayList<List<Integer>>();
        if (k > n) {
            return lists;
        }
        
        helper(lists, new ArrayList<Integer>(), 1, n, k);
        return lists;
    }
    
    public void helper(ArrayList<List<Integer>> lists, ArrayList<Integer> list, int start, int n, int k) {
        if (k == 0) {
            lists.add(new ArrayList(list));
            return;
        }
        ArrayList<Integer> tmp = new ArrayList<Integer>(list);
        for (int i = start; i <= n; i++) {
            tmp.add(i);
            helper(lists, tmp, i+1, n, k-1);
            tmp.remove(tmp.size()-1);
        }
    } 
}


289	Game of Life
public class Solution {
    public void gameOfLife(int[][] board) {
        if (board == null) {
            return;
        }
        int row = board.length;
        int col = board[0].length;
        
        int[][] new_board = new int[row][col];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                int t = numsofNeighbour(board,i,j);
                if (board[i][j] == 1) {
                    
                    if (t < 2 || t > 3) {
                        new_board[i][j] = 0;
                    }
                    if (t == 2 || t == 3) {
                        new_board[i][j] = 1;
                    }
                }else {
                    if (t == 3) {
                        new_board[i][j] = 1;
                    }    
                }
            }
        }
        
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                board[i][j] = new_board[i][j];
            }
        }
        
    }
    
    public int numsofNeighbour(int[][] board, int i, int j) {
        int row = board.length;
        int col = board[0].length;
        int ans = 0;
        if (i-1 >= 0 && board[i-1][j] == 1) ans++;
        if (i-1 >= 0 && j-1 >= 0 && board[i-1][j-1] == 1) ans++;
        if (i-1 >= 0 && j+1 < col && board[i-1][j+1] == 1) ans++;
        if (j-1 >= 0 && board[i][j-1] == 1) ans++;
        if (j+1 < col && board[i][j+1] == 1) ans++;
        if (i+1 < row && j-1>=0 && board[i+1][j-1] == 1) ans++;
        if (i+1 < row && board[i+1][j] == 1) ans++;
        if (i+1 < row && j+1 < col && board[i+1][j+1] == 1) ans++;
        return ans;
    }
}

251	Flatten 2D Vector 
Implement an iterator to flatten a 2d vector.

For example,
Given 2d vector =

[
  [1,2],
  [3],
  [4,5,6]
]
By calling next repeatedly until hasNext returns false, the order of elements returned by next
should be: [1,2,3,4,5,6].
public class Vector2D {

    Iterator<List<Integer>> it;
    Iterator<Integer> curr;
    
    public Vector2D(List<List<Integer>> vec2d) {
        it = vec2d.iterator();
    }

    public int next() {
        hasNext();
        return curr.next();
    }

    public boolean hasNext() {
        // 当前列表的迭代器为空，或者当前迭代器中没有下一个值时，需要更新为下一个迭代器
        while((curr == null || !curr.hasNext()) && it.hasNext()){
            curr = it.next().iterator();
        }
        return curr != null && curr.hasNext();
    }
}
维护两个迭代器：一个是输入的List<List<Integer>>的迭代器，它负责遍历List<Integer>的迭代器。另一个则是
List<Integer>的迭代器，
它负责记录当前到哪一个List的迭代器了。每次next时，我们先调用一下hasNext，确保当前List的迭代器有下一个值。


"*********************************"
300	Longest Increasing Subsequence
Given an unsorted array of integers, find the length of longest increasing subsequence.

For example,
Given [10, 9, 2, 5, 3, 7, 101, 18],
The longest increasing subsequence is [2, 3, 7, 101], therefore the length is 4. Note
that there may be more than one LIS combination, it is only necessary for you to return the
length.
Your algorithm should run in O(n2) complexity.

Follow up: Could you improve it to O(n log n) time complexity?
// I got it right;
public class Solution {
    public int lengthOfLIS(int[] nums) {
        if (nums.length <= 0) {
            return nums.length;
        }
        if (nums.length == 2) {
            return nums[1]>nums[0]?2:1;
        }
        
        int[] ans = new int [nums.length];
        ans[0] = 1;
        int max = 1;
        for (int i = 0; i < nums.length-1; i++) {
            for (int j = i+1; j < nums.length; j++) {
                if (nums[j] > nums[i]) {
                    ans[j] = Math.max(ans[j],Math.max(ans[i]+1, 2)); // this line is the most important
                    max = Math.max(max, ans[j]);
                }       
            }
        }
        return max;
    }
}

public class Solution {
    public int lengthOfLIS(int[] nums) 
    {
        List<Integer> sequence = new ArrayList();
        for(int n : nums) update(sequence, n);
    
        return sequence.size();
    }
    
    private void update(List<Integer> seq, int n)
    {
        if(seq.isEmpty() || seq.get(seq.size() - 1) < n) seq.add(n);
        else
        {
            seq.set(findFirstLargeEqual(seq, n), n);
        }
    }
    
    private int findFirstLargeEqual(List<Integer> seq, int target)
    {
        int lo = 0;
        int hi = seq.size() - 1;
        while(lo < hi)
        {
            int mid = lo + (hi - lo) / 2;
            if(seq.get(mid) < target) lo = mid + 1;
            else hi = mid;
        }
    
        return lo;
    }
}

334	Increasing Triplet Subsequence
Given an unsorted array return whether an increasing subsequence of length 3 exists or not in the array.

Formally the function should:
Return true if there exists i, j, k 
such that arr[i] < arr[j] < arr[k] given 0 ≤ i < j < k ≤ n-1 else return false.
Your algorithm should run in O(n) time complexity and O(1) space complexity.

Examples:
Given [1, 2, 3, 4, 5],
return true.

Given [5, 4, 3, 2, 1],
return false.

public class Solution {
    public boolean increasingTriplet(int[] nums) {
        if (nums.length < 3) {
            return false;
        }
        
        int[] ans = new int[nums.length];
        ans[0] = 1;
        int max = 0;
        for (int i = 0; i < nums.length-1; i++) {
            for (int j = i+1; j < nums.length; j++) {
                if (nums[j] > nums[i]) {
                    ans[j] = Math.max(ans[j], Math.max(ans[i]+1,2));
                    max = Math.max(max, ans[j]);
                        if (max == 3) {
                            return true;
                        }
                }
            }
        }
        return false;
    }
}

public class Solution {
    public boolean increasingTriplet(int[] nums) {
        int min = Integer.MAX_VALUE;
        int median = Integer.MAX_VALUE;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > median) {
                return true;
            }else if (nums[i] > min) {
                median = Math.min(median, nums[i]);
            }else {
                min = nums[i];
            }
        }
        return false;
    }
}

74	Search a 2D Matrix
Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following
properties:

Integers in each row are sorted from left to right.
The first integer of each row is greater than the last integer of the previous row.
For example,

Consider the following matrix:

[
  [1,   3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]
Given target = 3, return true.
// clean but not efficient
public class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix.length == 0 || matrix[0].length == 0) {
            return false;
        }
        
        int row = matrix.length;
        int col = matrix[0].length;
        
        int i = 0;
        int j = col - 1;
        while (i < row && j > -1) {
            if (target == matrix[i][j]) {
                return true;
            }
            if (target > matrix[i][j]) {
                i++;
            }else {
                j--;
            }
        }
        return false;
    }
}

215	Kth Largest Element in an Array
Find the kth largest element in an unsorted array. Note that it is the kth largest element in the sorted
order, not the kth distinct element.

For example,
Given [3,2,1,5,6,4] and k = 2, return 5.

Note: 
You may assume k is always valid, 1 ≤ k ≤ array's length.'

public class Solution {
    public int findKthLargest(int[] nums, int k) {
        Arrays.sort(nums);
        return nums[nums.length - k];
    }
}

284	Peeking Iterator
Given an Iterator class interface with methods: next() and hasNext(), design and implement a PeekingIterator
that support the peek() operation -- it essentially peek() at the element that will be returned by the next
call to next().

Here is an example. Assume that the iterator is initialized to the beginning of the list: [1, 2, 3].
Call next() gets you 1, the first element in the list.

Now you call peek() and it returns 2, the next element. Calling next() after that still return 2.
You call next() the final time and it returns 3, the last element. Calling hasNext() after that should
return false.

class PeekingIterator implements Iterator<Integer> {  
    private Integer next = null;
    private Iterator<Integer> iter;

    public PeekingIterator(Iterator<Integer> iterator) {
        // initialize any member here.
        iter = iterator;
        if (iter.hasNext())
            next = iter.next();
    }

    // Returns the next element in the iteration without advancing the iterator. 
    public Integer peek() {
        return next; 
    }

    // hasNext() and next() should behave the same as in the Iterator interface.
    // Override them if needed.
    @Override
    public Integer next() {
        Integer res = next;
        next = iter.hasNext() ? iter.next() : null;
        return res; 
    }

    @Override
    public boolean hasNext() {
        return next != null;
    }
}

class PeekingIterator<T> implements Iterator<T> {

    T cache;
    Iterator<T> it;

    public PeekingIterator(Iterator<T> iterator) {
        // initialize any member here.
        this.cache = iterator.next();
        this.it = iterator;
    }

    // Returns the next element in the iteration without advancing the iterator.
    public T peek() {
        return cache;
    }

    // hasNext() and next() should behave the same as in the Iterator interface.
    // Override them if needed.
    @Override
    public T next() {
        T res = cache;
        cache = it.hasNext() ? it.next() : null;
        return res;
    }

    @Override
    public boolean hasNext() {
        return it.hasNext() || cache != null;
    }
}

73	Set Matrix Zeroes
Given a m x n matrix, if an element is 0, set its entire row and column to 0. Do it in place.

public class Solution {
    public void setZeroes(int[][] matrix) {
        int row = matrix.length;
        int col = matrix[0].length;
        
        Set<Integer> r = new HashSet<>();
        Set<Integer> c = new HashSet<>();
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (matrix[i][j] == 0) {
                    r.add(i);
                    c.add(j);
                }
            }
        }
        for (int rIndex: r) {
            for (int j = 0; j < col; j++) {
                matrix[rIndex][j] = 0;
            }
        }
        for (int cIndex: c) {
            for (int j = 0; j < row; j++) {
                matrix[j][cIndex] = 0;
            }
        }
    }
}


"************************"
261	Graph Valid Tree
Given n nodes labeled from 0 to n - 1 and a list of undirected edges (each edge is a pair of nodes), write
a function to check whether these edges make up a valid tree.

For example:
Given n = 5 and edges = [[0, 1], [0, 2], [0, 3], [1, 4]], return true.
Given n = 5 and edges = [[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]], return false.

public class Solution {
    public boolean validTree(int n, int[][] edges) {
        // initialize n isolated islands
        int[] nums = new int[n];
        Arrays.fill(nums, -1);

        // perform union find
        for (int i = 0; i < edges.length; i++) {
            int x = find(nums, edges[i][0]);
            int y = find(nums, edges[i][1]);

            // if two vertices happen to be in the same set
            // then there's a cycle
            if (x == y) return false;

            // union
            nums[x] = y;
        }

        return edges.length == n - 1;
    }

    int find(int nums[], int i) {
        if (nums[i] == -1) return i;
        return find(nums, nums[i]);
    }
}


162	Find Peak Element
A peak element is an element that is greater than its neighbors.
Given an input array where num[i] ≠ num[i+1], find a peak element and return its index.

The array may contain multiple peaks, in that case return the index to any one of the peaks is fine.
You may imagine that num[-1] = num[n] = -∞.

For example, in array [1, 2, 3, 1], 3 is a peak element and your function should return the index number 2.

Your slution should be in logarithmic complexity

public class Solution {
    public int findPeakElement(int[] nums) {
        int N = nums.length;
        if (N == 1) {
            return 0;
        }
    
        int left = 0, right = N - 1;
        while (right - left > 1) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < nums[mid + 1]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
    
        return (left == N - 1 || nums[left] > nums[left + 1]) ? left : right;
    }
}

We want to check mid and mid+1 elements. if(nums[mid] < nums[mid+1]), lo = mid + 1, otherwise hi = mid. 
The reason is that when there are even or odd number of elements, the mid element is always going to have
a next one mid+1. We donnot need to consider the case when there is less than 1 element as it is not valid
case for this problem. Finally we return lo as it will always be a solution since it goes to mid+1
element in the first case, which is larger.

"************************"
279	Perfect Squares
Given a positive integer n, find the least number of perfect square numbers (for example, 1, 4, 9, 16, ...)
which sum to n.
For example, given n = 12, return 3 because 12 = 4 + 4 + 4; given n = 13, return 2 because 13 = 4 + 9.

public class Solution {
    public int numSquares(int n) {
        int[] min = new int[n+1];
        min[1] = 1;
        for(int i=2;i<=n;i++){
            int tmp = Integer.MAX_VALUE;
            for(int j=1;j*j<=i;j++){
                tmp = Math.min(tmp,min[i-j*j]+1);
            }
            min[i] = tmp;
        }
        return min[n];
    }
}

四平方和定理(Lagrange's Four-Square Theorem)：所有自然数至多只要用四个数的平方和就可以表示。'

参考链接：https://leetcode.com/discuss/56982/o-sqrt-n-in-ruby-and-c
注意下面的!!a + !!b这个表达式，可能很多人不太理解这个的意思，其实很简单，感叹号!表示逻辑取反，那么一个正整数逻辑取反为0，再取反为1，
所以用两个感叹号!!的作用就是看a和b是否为正整数，都为正整数的话返回2，只有一个是正整数的话返回1
C代码：
int numSquares(int n) {
    while (n % 4 == 0)
        n /= 4;
    if (n % 8 == 7)
        return 4;
    for (int a=0; a*a<=n; ++a) {
        int b = sqrt(n - a*a);
        if (a*a + b*b == n)
            return !!a + !!b;
    }
    return 3;
}

80	Remove Duplicates from Sorted Array II
Follow up for "Remove Duplicates":
What if duplicates are allowed at most twice?

For example,
Given sorted array nums = [1,1,1,2,2,3],

Your function should return length = 5, with the first five elements of nums being 1, 1, 2, 2 and 3. 
It doesn't matter what you leave beyond the new length.'

public class Solution {
    public int removeDuplicates(int[] nums) {
        int ans = 0;
        for (int i = 0; i < nums.length; i++) {
            if (i < nums.length - 1 && nums[i] == nums[i+1]) {
                i++;
                while (i < nums.length - 1 && nums[i] == nums[i+1]) {
                    ans++;
                    i++;
                }
                if (ans != 0) {
                    nums[i-ans-1] = nums[i];
                    nums[i-ans] = nums[i];
                }
                i--;
            }else if (ans != 0) {
                nums[i-ans] = nums[i];
            }
        }
        return nums.length-ans;
    }
}

129	Sum Root to Leaf Numbers
Given a binary tree containing digits from 0-9 only, each root-to-leaf path could represent a number.
An example is the root-to-leaf path 1->2->3 which represents the number 123.

Find the total sum of all root-to-leaf numbers.
For example,

    1
   / \
  2   3
The root-to-leaf path 1->2 represents the number 12.
The root-to-leaf path 1->3 represents the number 13.

Return the sum = 12 + 13 = 25.

public class Solution {
    public int sumNumbers(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int ans = 0;
        return helper(root,ans);
    }
    
    public int helper(TreeNode root, int ans) {
        if (root == null) {
            return ans;
        }
        if (root.left == null && root.right == null) {
            return ans*10 + root.val;
        }
        if (root.left != null && root.right != null) {
            return helper(root.left, ans*10+root.val) + helper(root.right, ans*10+root.val);    
        }
        return root.left == null? helper(root.right, ans*10+root.val):helper(root.left,ans*10+root.val);
    }
}

public class Solution {
public int sumNumbers(TreeNode root) {
    int total = 0;
    LinkedList<TreeNode> q = new LinkedList<TreeNode>();
    LinkedList<Integer> sumq = new LinkedList<Integer>();
    if(root !=null){
        q.addLast(root);
        sumq.addLast(root.val);
    }
    while(!q.isEmpty()){
        TreeNode current = q.removeFirst();
        int partialSum = sumq.removeFirst();
        if(current.left == null && current.right==null){
            total+=partialSum;
        }else{
            if(current.right !=null){
                q.addLast(current.right);
                sumq.addLast(partialSum*10+current.right.val);
            }
            if(current.left !=null){
                q.addLast(current.left);
                sumq.addLast(partialSum*10+current.left.val);
            }

        }

    }
    return total;
}

274. H-Index
Given an array of citations (each citation is a non-negative integer) of a researcher, write a function to
compute the researcher's h-index.
According to the definition of h-index on Wikipedia: "A scientist has index h if h of his/her N papers have
at least h citations each, and the other N − h papers have no more than h citations each."

For example, given citations = [3, 0, 6, 1, 5], which means the researcher has 5 papers in total and each
of them had received 3, 0, 6, 1, 5 citations respectively. Since the researcher has 3 papers with at least
3 citations each and the remaining two with no more than 3 citations each, his h-index is 3.

Note: If there are several possible values for h, the maximum one is taken as the h-index.

public class Solution {
    public int hIndex(int[] citations) {
        Arrays.sort(citations);
        int ans = 0;
        for (int i = 0; i < citations.length; i++) {
            if (citations[i] >= citations.length - i) {
                ans = Math.max(ans, citations.length - i);    
            }
        }
        return ans;
    }
}




275	H-Index II
Follow up for H-Index: What if the citations array is sorted in ascending order? Could you optimize
your algorithm?

public class Solution {
    public int hIndex(int[] citations) {
        Arrays.sort(citations);
        int ans = 0;
        for (int i = 0; i < citations.length; i++) {
            if (citations[i] >= citations.length - i) {
                ans = Math.max(ans, citations.length - i);    
            }
        }
        return ans;
    }
}

public class Solution {
    public int hIndex(int[] citations) {
        if(citations == null || citations.length == 0) return 0;
        int l = 0, r = citations.length;
        int n = citations.length;
        while(l < r){
            int mid = l + (r - l) / 2;
            if(citations[mid] == n - mid) return n - mid;
            if(citations[mid] < citations.length - mid) l = mid + 1;
            else r = mid;
        }
        return n - l;
    }
}

331	Verify Preorder Serialization of a Binary Tree
One way to serialize a binary tree is to use pre-order traversal. When we encounter a non-null node, we
record the node's value. If it is a null node, we record using a sentinel value such as #.'

     _9_
    /   \
   3     2
  / \   / \
 4   1  #  6
/ \ / \   / \
# # # #   # #
For example, the above binary tree can be serialized to the string "9,3,4,#,#,1,#,#,2,#,6,#,#", where # 
represents a null node.

Given a string of comma separated values, verify whether it is a correct preorder traversal serialization
of a binary tree. Find an algorithm without reconstructing the tree.

Each comma separated value in the string must be either an integer or a character '#' representing null
pointer.

You may assume that the input format is always valid, for example it could never contain two consecutive
commas such as "1,,3".

Example 1:
"9,3,4,#,#,1,#,#,2,#,6,#,#"
Return true

Example 2:
"1,#"
Return false

Example 3:
"9,#,#,1"
Return false

Some used stack. Some used the depth of a stack. Here I use a different perspective. In a binary tree, 
if we consider null as leaves,
then all non-null node provides 2 outdegree and 1 indegree (2 children and 1 parent), except root
all null node provides 0 outdegree and 1 indegree (0 child and 1 parent).
Suppose we try to build this tree. During building, we record the difference between out degree and
in degree diff = outdegree - indegree. When the next node comes, we then decrease diff by 1, because
the node provides an in degree.
If the node is not null, we increase diff by 2, because it provides two out degrees. If a serialization
is correct, diff should never be negative and diff will be zero when finished.

public boolean isValidSerialization(String preorder) {
    String[] nodes = preorder.split(",");
    int diff = 1;
    for (String node: nodes) {
        if (--diff < 0) return false;
        if (!node.equals("#")) diff += 2;
    }
    return diff == 0;
}


351	Android Unlock Patterns
Given an Android 3x3 key lock screen and two integers m and n, where 1 ≤ m ≤ n ≤ 9, count the total number
of unlock patterns of the Android lock screen, which consist of minimum of m keys and maximum n keys.

Rules for a valid pattern:
Each pattern must connect at least m keys and at most n keys.
All the keys must be distinct.
If the line connecting two consecutive keys in the pattern passes through any other keys, the other keys
must have previously selected in the pattern. No jumps through non selected key is allowed.
The order of keys used matters.

| 1 | 2 | 3 |
| 4 | 5 | 6 |
| 7 | 8 | 9 |
Invalid move: 4 - 1 - 3 - 6 
Line 1 - 3 passes through key 2 which had not been selected in the pattern.

Invalid move: 4 - 1 - 9 - 2
Line 1 - 9 passes through key 5 which had not been selected in the pattern.

Valid move: 2 - 4 - 1 - 3 - 6
Line 1 - 3 is valid because it passes through key 2, which had been selected in the pattern

Valid move: 6 - 5 - 4 - 1 - 9 - 2
Line 1 - 9 is valid because it passes through key 5, which had been selected in the pattern.

Example:
Given m = 1, n = 1, return 9.

public class Solution {  
    private int patterns;  
    private boolean valid(boolean[] keypad, int from, int to) {  
        if (from==to) return false;  
        int i=Math.min(from, to), j=Math.max(from,to);  
        if ((i==1 && j==9) || (i==3 && j==7)) return keypad[5] && !keypad[to];  
        if ((i==1 || i==4 || i==7) && i+2==j) return keypad[i+1] && !keypad[to];  
        if (i<=3 && i+6==j) return keypad[i+3] && !keypad[to];  
        return !keypad[to];  
    }  
    private void find(boolean[] keypad, int from, int step, int m, int n) {  
        if (step == n) {  
            patterns ++;  
            return;  
        }  
        if (step >= m) patterns ++;  
        for(int i=1; i<=9; i++) {  
            if (valid(keypad, from, i)) {  
                keypad[i] = true;  
                find(keypad, i, step+1, m, n);  
                keypad[i] = false;  
            }  
        }  
    }  
    public int numberOfPatterns(int m, int n) {  
        boolean[] keypad = new boolean[10];  
        for(int i=1; i<=9; i++) {  
            keypad[i] = true;  
            find(keypad, i, 1, m, n);  
            keypad[i] = false;  
        }  
        return patterns;  
    }  
} 

78	Subsets
Given a set of distinct integers, nums, return all possible subsets.
Note: The solution set must not contain duplicate subsets.

For example,
If nums = [1,2,3], a solution is:

[
  [3],
  [1],
  [2],
  [1,2,3],
  [1,3],
  [2,3],
  [1,2],
  []
]

public class Solution {
    public List<List<Integer>> subsets(int[] nums) {
        ArrayList<List<Integer>> lists = new ArrayList<List<Integer>>();
        if (nums.length == 0) {
            return lists;
        }
        Arrays.sort(nums);
        int len = nums.length;
        int n = (int)Math.pow(2,len);
        for (int i = 0; i < n; i++) {
            ArrayList<Integer> list = new ArrayList<>();
            for (int j = 0; j < len; j++) {
                if ((i & (1<<j)) != 0) {
                    list.add(nums[j]);
                }
            }
            lists.add(list);
        }
        return lists;
    }
}

90	Subsets II
Given a collection of integers that might contain duplicates, nums, return all possible subsets.

Note: The solution set must not contain duplicate subsets.

For example,
If nums = [1,2,2], a solution is:

[
  [2],
  [1],
  [1,2,2],
  [2,2],
  [1,2],
  []
]

public class Solution {
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        ArrayList<List<Integer>> lists = new ArrayList<List<Integer>>();
        if (nums.length == 0) {
            return lists;
        }
        Arrays.sort(nums);
        for (int i = 0; i < Math.pow(2,nums.length); i++) {
            ArrayList<Integer> list = new ArrayList<>();
            for (int j = 0; j < nums.length; j++) {
                if ((i & (1<<j)) != 0) {
                    list.add(nums[j]);
                }
            }
            if (!lists.contains(list)) {
                lists.add(list);
            }
        }
        return lists;
    }
}



114	Flatten Binary Tree to Linked List
Given a binary tree, flatten it to a linked list in-place.

For example,
Given

         1
        / \
       2   5
      / \   \
     3   4   6
The flattened tree should look like:
   1
    \
     2
      \
       3
        \
         4
          \
           5
            \
             6
public class Solution {
    public void flatten(TreeNode root) {
        while (root != null) {
            TreeNode left = root.left;
            if (left == null) 
            {
                root = root.right;
                continue;
            }
            while (left.right != null) {
                left = left.right;
            }
            left.right = root.right;
            root.right = root.left;
            root.left = null;
            root = root.right;
        }
    }
}

39	Combination Sum
Given a set of candidate numbers (C) and a target number (T), find all unique combinations in C where the
candidate numbers sums to T.

The same repeated number may be chosen from C unlimited number of times.

Note:
All numbers (including target) will be positive integers.
The solution set must not contain duplicate combinations.
For example, given candidate set [2, 3, 6, 7] and target 7, 
A solution set is: 
[
  [7],
  [2, 2, 3]
]

public class Solution {
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        Arrays.sort(candidates);
        ArrayList<List<Integer>> res = new ArrayList<List<Integer>>();
        add(res, new ArrayList<Integer>(), candidates, 0, target);
        return res;
    }

    private void add(ArrayList<List<Integer>> res, ArrayList<Integer> list, int[] candidates, int start, int target){
        if(target < 0)  return;
        else if(target == 0){
            res.add(list);
            return;
        }
        for(int i=start; i<candidates.length; ++i){
            ArrayList<Integer> temp = new ArrayList<Integer>(list);
            temp.add(candidates[i]);
            add(res, temp, candidates, i, target-candidates[i]);
        }
    }
}

142	Linked List Cycle II
Given a linked list, return the node where the cycle begins. If there is no cycle, return null.
Note: Do not modify the linked list.

Follow up:
Can you solve it without using extra space?

public class Solution {
    public ListNode detectCycle(ListNode head) {
        if (head == null || head.next == null) {
            return null;
        }
        ListNode slow = head;
        ListNode fast = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) {
                ListNode slow2 = head;
                while (slow2 != slow) {
                    slow = slow.next;
                    slow2 = slow2.next;
                }
                return slow;
            }

        }
        return null;
    }
}

314	Binary Tree Vertical Order Traversal 
Given a binary tree, return the vertical order traversal of its nodes' values. (ie, from top to bottom, 
column by column).
If two nodes are in the same row and column, the order should be from left to right.

Examples:
Given binary tree [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
return its vertical order traversal as:

[
  [9],
  [3,15],
  [20],
  [7]
]
Given binary tree [3,9,20,4,5,2,7],

    _3_
   /   \
  9    20
 / \   / \
4   5 2   7
return its vertical order traversal as:

[
  [4],
  [9],
  [3,5,2],
  [20],
  [7]
]

二叉树Vertical order traversal。这道题意思很简单但例子举得不够好，假如上面第二个例子里5还有右子树的话，就会和20在一条column里。
总的来说就是假定一个node的column是 i，那么它的左子树column就是i - 1，右子树column就是i + 1。我们可以用decorator模式建立一个
TreeColumnNode，包含一个TreeNode，以及一个column value，然后用level order traversal进行计算就可以了，计算的时候用一个
HashMap保存column value以及相同value的点。也要设置一个min column value和一个max column value，方便最后按照从小到大顺序获
取hashmap里的值输出。这道题Discuss区Yavinci大神写得非常棒，放在reference里。

Time Complexity - O(n)，  Space Complexity - O(n)
public List<List<Integer>> verticalOrder(TreeNode root) {
    List<List<Integer>> res = new ArrayList<>();
    if(root == null) return res;

    Map<Integer, ArrayList<Integer>> map = new HashMap<>();
    Queue<TreeNode> q = new LinkedList<>();
    Queue<Integer> cols = new LinkedList<>();

    q.add(root); 
    cols.add(0);

    int min = 0, max = 0;
    while(!q.isEmpty()) {
        TreeNode node = q.poll();
        int col = cols.poll();
        if(!map.containsKey(col)) map.put(col, new ArrayList<Integer>());
        map.get(col).add(node.val);

        if(node.left != null) {
            q.add(node.left); 
            cols.add(col - 1);
            if(col <= min) min = col - 1;
        }
        if(node.right != null) {
            q.add(node.right);
            cols.add(col + 1);
            if(col >= max) max = col + 1;
        }
    }

    for(int i = min; i <= max; i++) {
        res.add(map.get(i));
    }

    return res;
}


201	Bitwise AND of Numbers Range
Given a range [m, n] where 0 <= m <= n <= 2147483647, return the bitwise AND of all numbers in this range,
inclusive.
For example, given the range [5, 7], you should return 4.

The idea is to use a mask to find the leftmost common digits of m and n. Example: m=1110001, n=1110111, 
and you just need to find 1110000 and it will be the answer.

public class Solution {
    public int rangeBitwiseAnd(int m, int n) {
        int c=0;
        while(m!=n){
            m>>=1;
            n>>=1;
            ++c;
        }
        return n<<c;
    }
}

109	Convert Sorted List to Binary Search Tree
Given a singly linked list where elements are sorted in ascending order, convert it to a height balanced BST.
public class Solution {
    public TreeNode sortedListToBST(ListNode head) {
        if (head == null) {
            return null;
        }
        
        ArrayList<Integer> list = new ArrayList<>();
        while (head != null) {
            list.add(head.val);
            head = head.next;
        }
        
        TreeNode root = new TreeNode(list.get(list.size()/2));
        return helper(root, list, 0, list.size()-1);
    }
    
    public TreeNode helper(TreeNode root, ArrayList<Integer> list, int lo, int hi) {
        if (lo > hi) {
            return null;
        }
        int m = lo + (hi-lo)/2;
        root = new TreeNode(list.get(m)); //************ this is the must be statement
        root.left = helper(root.left, list, lo, m-1);
        root.right = helper(root.right, list, m+1, hi);
        return root;
    }
}

public class Solution {
	public TreeNode sortedListToBST(ListNode head) {
	    if(head==null) return null;
	    return toBST(head,null);
	}
	public TreeNode toBST(ListNode head, ListNode tail){
	    ListNode slow = head;
	    ListNode fast = head;
	    if(head==tail) return null;

	    while(fast!=tail&&fast.next!=tail){
	        fast = fast.next.next;
	        slow = slow.next;
	    }
	    TreeNode thead = new TreeNode(slow.val);
	    thead.left = toBST(head,slow);
	    thead.right = toBST(slow.next,tail);
	    return thead;
	}
}


120	Triangle
Given a triangle, find the minimum path sum from top to bottom. Each step you may move to adjacent numbers
on the row below.

For example, given the following triangle
[
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
The minimum path sum from top to bottom is 11 (i.e., 2 + 3 + 5 + 1 = 11).
public class Solution {
    public int minimumTotal(List<List<Integer>> triangle) {
        int ans = 0;
        if (triangle.size() == 0) {
            return ans;
        }
        for (int i = triangle.size() - 2; i >= 0; i--) {
            List<Integer> nextList = triangle.get(i+1);
            for (int j = 0; j < triangle.get(i).size(); j++) {
                int min = Math.min(nextList.get(j), nextList.get(j+1)) + triangle.get(i).get(j);
                triangle.get(i).set(j,min);
            }
        }
        return triangle.get(0).get(0);
    }
}

//for this solution, needs to update the size;
public class Solution {
    public int minimumTotal(List<List<Integer>> triangle) {
        int min = Integer.MAX_VALUE;
        if (triangle.size() == 0) {
            return 0;
        }
        
        ArrayList<Integer> ans = new ArrayList<>();
        ans.add(triangle.get(0).get(0));
        min = Math.min(min,ans.get(0));
        for (int i = 1; i < triangle.size(); i++) {
            ArrayList<Integer> tmp = new ArrayList<>();
            int len = ans.size();
            System.out.println(len);
            for (int j = 0; j < len ; j++) {
                System.out.println(j);
                tmp.add(ans.get(j)+triangle.get(i).get(j));   // number is bigger , so use the backend method
                tmp.add(ans.get(j)+triangle.get(i).get(j+1));
                min = Math.min(min,tmp.get(j));
                min = Math.min(min,tmp.get(j+1));
            }
            ans = tmp;
        }
        return min;
    }
}

86	Partition List
Given a linked list and a value x, partition it such that all nodes less than x come before nodes greater
than or equal to x.
You should preserve the original relative order of the nodes in each of the two partitions.

For example,
Given 1->4->3->2->5->2 and x = 3,
return 1->2->2->4->3->5.

public class Solution {
    public ListNode partition(ListNode head, int x) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode ans = new ListNode(0);
        ListNode p1 = ans;
        ListNode p = new ListNode(0);
        ListNode p2 = p;
        while (head != null) {
            if (head.val < x) {
                p1.next = head;
                p1 = p1.next;
            }else {
                p2.next = head;
                p2 = p2.next;
            }
            head = head.next;
        }
        p2.next = null;
        p1.next = p.next;        
        return ans.next;
    }
}

330	Patching Array
Given a sorted positive integer array nums and an integer n, add/patch elements to the array such that any
number in range [1, n] inclusive can be formed by the sum of some elements in the array. Return the minimum
number of patches required.

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

Explanation

Let miss be the smallest sum in [0,n] that we might be missing. Meaning we already know we can build all
sums in [0,miss). Then if we have a number num <= miss in the given array, we can add it to those smaller
sums to build all sums in [0,miss+num).If we don't, then we must add such a number to the array, and it's
best to add miss itself, to maximize the reach.

Example: Let's say the input is nums = [1, 2, 4, 13, 43] and n = 100. We need to ensure that all sums in the
range [1,100] are possible. Using the given numbers 1, 2 and 4, we can already build all sums from 0 to 7,
i.e., the range [0,8). But we can't build the sum 8, and the
next given number (13) is too large. So we insert 8 into the array. Then we can build all sums in [0,16).

Do we need to insert 16 into the array? No! We can already build the sum 3, and adding the given 13 gives us
sum 16. We can also add the 13
to the other sums, extending our range to [0,29).

And so on. The given 43 is too large to help with sum 29, so we must insert 29 into our array. This extends
our range to [0,58)
But then the 43 becomes useful and expands our range to [0,101). At which point we're done.
Another implementation, though I prefer the above one.

int minPatches(vector<int>& nums, int n) {
    int count = 0, i = 0;
    for (long miss=1; miss <= n; count++)
        miss += (i < nums.size() && nums[i] <= miss) ? nums[i++] : miss;
    return count - i;
}


147	Insertion Sort List
Sort a linked list using insertion sort.
public class Solution {
    public ListNode insertionSortList(ListNode head) {
        if(head == null || head.next == null) return head;
        ListNode pre = new ListNode(-11);
        ListNode sorted = new ListNode(head.val);
        pre.next = sorted;
        head = head.next;
        while(head != null) {
            if(head.val < sorted.val) {
                ListNode tempPre = pre;
                while(pre.next != null && pre.next.val < head.val) {
                    pre = pre.next;
                }
                ListNode temp = pre.next;
                pre.next = new ListNode(head.val);
                pre.next.next = temp;
                pre = tempPre;
            } else {
                sorted.next = new ListNode(head.val);
                sorted = sorted.next;
            }
            head = head.next;
        }
        return pre.next;
    }

}


163	Missing Ranges
Given a sorted integer array where the range of elements are [0, 99] inclusive, return its missing ranges.
For example, given [0, 1, 3, 50, 75], return [“2”, “4->49”, “51->74”, “76->99”]

public class Solution {
    public List<String> findMissingRanges(int[] vals, int start, int end) {
      List<String> ranges = new ArrayList<String>();
      int prev = start - 1;
      for (int i=0; i<=vals.length; ++i) {
          int curr = (i==vals.length) ? end + 1 : vals[i];
          if ( cur-prev>=2 ) {
              ranges.add(getRange(prev+1, curr-1));
          }
          prev = curr;
      }
      return ranges;
    }
 
    private String getRange(int from, int to) {
      return (from==to) ? String.valueOf(from) : from + "->" + to;
    }
}

16	3Sum Closest
Given an array S of n integers, find three integers in S such that the sum is closest to a given number,
target. Return the sum of the three integers. You may assume that each input would have exactly one solution.

For example, given array S = {-1 2 1 -4}, and target = 1.
The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).

public class Solution {
    public int threeSumClosest(int[] num, int target) {
        if (num == null || num.length < 3) {
            return Integer.MAX_VALUE;
        }
        Arrays.sort(num);
        int closet = Integer.MAX_VALUE / 2; // otherwise it will overflow for opeartion (closet-target)'
        for (int i = 0; i < num.length - 2; i++) {
            int left = i + 1;
            int right = num.length - 1;
            while (left < right) {
                int sum = num[i] + num[left] + num[right];
                if (sum == target) {
                    return sum;
                } else if (sum < target) {
                    left++;
                } else {
                    right--;
                }
                closet = Math.abs(sum - target) < Math.abs(closet - target) ? sum : closet;
            }
        }
        return closet;
    }
}

34	Search for a Range
Given a sorted array of integers, find the starting and ending position of a given target value.
Your algorithm's runtime complexity must be in the order of O(log n).
If the target is not found in the array, return [-1, -1].

For example,
Given [5, 7, 7, 8, 8, 10] and target value 8,
return [3, 4].

public class Solution {
    public int[] searchRange(int[] nums, int target) {
      int[] ans = new int[2];
      ans[0] = ans[1] = -1;
      if (nums.length == 0) {
          return ans;
      }
      
      int lo = 0;
      int hi = nums.length-1;
      while (lo <= hi) {
          int m = lo + (hi-lo)/2;
          if (nums[m] == target) {
              int t = m;
              while (t>=0 && nums[t] == target) {
                  t--;
              }
              ans[0] = t+1;
              t = m;
              while(t<=hi && nums[t] == target) {
                  t++;
              }
              ans[1] = t-1;
          }
          if (nums[m] > target) {
              hi = m-1;
          }else {
              lo = m+1;
          }
      }
      return ans;
    }
}

106	Construct Binary Tree from Inorder and Postorder Traversal
Given inorder and postorder traversal of a tree, construct the binary tree.

Note:
You may assume that duplicates do not exist in the tree.

Inorder Traversal:
Algorithm Inorder(tree)
   1. Traverse the left subtree, i.e., call Inorder(left-subtree)
   2. Visit the root.
   3. Traverse the right subtree, i.e., call Inorder(right-subtree)

Preorder Traversal:
Algorithm Preorder(tree)
   1. Visit the root.
   2. Traverse the left subtree, i.e., call Preorder(left-subtree)
   3. Traverse the right subtree, i.e., call Preorder(right-subtree)

Postorder Traversal:
Algorithm Postorder(tree)
   1. Traverse the left subtree, i.e., call Postorder(left-subtree)
   2. Traverse the right subtree, i.e., call Postorder(right-subtree)
   3. Visit the root.

/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class Solution {
    public TreeNode buildTree(int[] inorder, int[] postorder) {
        if (inorder.length == 0) {
            return null;
        }
        int lo = 0;
        int hi = inorder.length - 1;
        TreeNode root = new TreeNode(postorder[hi]);
        if (lo == hi) {
            root.left = null;
            root.right = null;
            return root;
        }else {
            int m = findIndex(inorder, postorder[hi]);
            root.left = buildTree(Arrays.copyOfRange(inorder, lo, m), Arrays.copyOfRange(postorder,lo, m));
            root.right = buildTree(Arrays.copyOfRange(inorder, m+1, hi+1), Arrays.copyOfRange(postorder,m,hi));
        }
        return root;
    } 

    public int findIndex(int[] nums, int target) {
        for (int i = 0; i < nums.length; i++) {
            if (target == nums[i]) {
                return i;
            }
        }
        return 0;
    }
}

17	Letter Combinations of a Phone Number
Given a digit string, return all possible letter combinations that the number could represent.
A mapping of digit to letters (just like on the telephone buttons) is given below.
Input:Digit string "23"
Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].

public class Solution {
    public List<String> letterCombinations(String digits) {
        LinkedList<String> ans = new LinkedList<String>();
        if (digits.length() == 0) {
            return ans;
        }
        String[] mapping = new String[] {"0", "1", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        ans.add("");
        for(int i =0; i<digits.length();i++){
            int x = Character.getNumericValue(digits.charAt(i));
            while(ans.peek().length()==i){
                String t = ans.remove();
                for(char s : mapping[x].toCharArray())
                    ans.add(t+s);
            }
        }
        return ans;
    }
}

// peek and remove(), the head


95	Unique Binary Search Trees II
Given an integer n, generate all structurally unique BST's (binary search trees) that store values 1...n.

For example,
Given n = 3, your program should return all 5 unique BST's shown below.

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3

public class Solution {
    public List<TreeNode> generateTrees(int n) {
        if(n==0) return new LinkedList(); //here is new line
        return generateSubtrees(1, n);
    }
    
    private List<TreeNode> generateSubtrees(int s, int e) {
        List<TreeNode> res = new LinkedList<TreeNode>();
        if (s > e) {
            res.add(null); // empty tree
            return res;
        }
    
        for (int i = s; i <= e; ++i) {
            List<TreeNode> leftSubtrees = generateSubtrees(s, i - 1);
            List<TreeNode> rightSubtrees = generateSubtrees(i + 1, e);
    
            for (TreeNode left : leftSubtrees) {
                for (TreeNode right : rightSubtrees) {
                    TreeNode root = new TreeNode(i);
                    root.left = left;
                    root.right = right;
                    res.add(root);
                }
            }
        }
        return res;
    }
}

103	Binary Tree Zigzag Level Order Traversal
Given a binary tree, return the zigzag level order traversal of its nodes' values. (ie, from left to right,
then right to left for the next level and alternate between).

For example:
Given binary tree [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
return its zigzag level order traversal as:
[
  [3],
  [20,9],
  [15,7]
]

public class Solution {
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        ArrayList<List<Integer>> lists = new ArrayList<List<Integer>>();
        if (root == null) {
            return lists;
        }
        
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        int i = 0;
        while (!q.isEmpty()) {
            Queue<TreeNode> sameLevel = new LinkedList<>();
            ArrayList<Integer> list = new ArrayList<>();
            while (!q.isEmpty()) {
                TreeNode tmp = q.poll();
                if (tmp.left != null) {
                    sameLevel.offer(tmp.left);
                }
                if (tmp.right != null) {
                    sameLevel.offer(tmp.right);
                }
                list.add(tmp.val);
            }
            q = sameLevel;
            System.out.println(i);
            if (i%2 == 1) {
                Collections.reverse(list);
                lists.add(list);
            }else {
                lists.add(list);
            }
            i++;
        }
        return lists;
    }
}

105	Construct Binary Tree from Preorder and Inorder Traversal

public TreeNode buildTree(int[] preorder, int[] inorder) {
    Map<Integer, Integer> inMap = new HashMap<Integer, Integer>();

    for(int i = 0; i < inorder.length; i++) {
        inMap.put(inorder[i], i);
    }

    TreeNode root = buildTree(preorder, 0, preorder.length - 1, inorder, 0, inorder.length - 1, inMap);
    return root;
}

public TreeNode buildTree(int[] preorder, int preStart, int preEnd, int[] inorder, int inStart, int inEnd, Map<Integer, Integer> inMap) {
    if(preStart > preEnd || inStart > inEnd) return null;

    TreeNode root = new TreeNode(preorder[preStart]);
    int inRoot = inMap.get(root.val);
    int numsLeft = inRoot - inStart;

    root.left = buildTree(preorder, preStart + 1, preStart + numsLeft, inorder, inStart, inRoot - 1, inMap);
    root.right = buildTree(preorder, preStart + numsLeft + 1, preEnd, inorder, inRoot + 1, inEnd, inMap);

    return root;
}

186	Reverse Words in a String II 
Given an input string, reverse the string word by word. A word is defined as a sequence of non-space
characters.
The input string does not contain leading or trailing spaces and the words are always separated by a
single space.

For example,
Given s = "the sky is blue",
return "blue is sky the".
Could you do it in-place without allocating extra space?

public class Solution {
    public void reverseWords(char[] s) {
        if (s.length == 0) return;
        reverse(s, 0, s.length-1);
        int last = 0;
        for (int i=0; i<s.length; i++) {
            if (s[i] == ' ') {
                reverse(s, last, i-1);
                last = i + 1;
            }
        }
    }
    
    public void reverse(char[] s, int l, int r) {
        while (l <= r) {
            int temp = s[l];
            s[l] = s[r];
            s[r] = temp;
            l++;
            r--;
        }
    }
}


161	One Edit Distance 
Given two strings S and T, determine if they are both one edit distance apart.

public class Solution {
    public boolean isOneEditDistance(String s, String t) {
        int m = s.length(), n = t.length();
        if(m == n) return isOneModified(s, t);
        if(m - n == 1) return isOneDeleted(s, t);
        if(n - m == 1) return isOneDeleted(t, s);
        // 长度差距大于2直接返回false
        return false;
    }
    
    private boolean isOneModified(String s, String t){
        boolean modified = false;
        // 看是否只修改了一个字符
        for(int i = 0; i < s.length(); i++){
            if(s.charAt(i) != t.charAt(i)){
                if(modified) return false;
                modified = true;
            }
        }
        return modified;
    }
    
    public boolean isOneDeleted(String longer, String shorter){
        // 找到第一组不一样的字符，看后面是否一样
        for(int i = 0; i < shorter.length(); i++){
            if(longer.charAt(i) != shorter.charAt(i)){
                return longer.substring(i + 1).equals(shorter.substring(i));
            }
        }
        return true;
    }
}


341	Flatten Nested List Iterator
Given a nested list of integers, implement an iterator to flatten it.
Each element is either an integer, or a list -- whose elements may also be integers or other lists.

Example 1:
Given the list [[1,1],2,[1,1]],
By calling next repeatedly until hasNext returns false, the order of elements returned by next should be: 
[1,1,2,1,1].

Example 2:
Given the list [1,[4,[6]]],
By calling next repeatedly until hasNext returns false, the order of elements returned by next should be: 
[1,4,6].

public class NestedIterator implements Iterator<Integer> {
    Stack<NestedInteger> stack = new Stack<>();
    public NestedIterator(List<NestedInteger> nestedList) {
        for(int i = nestedList.size() - 1; i >= 0; i--) {
            stack.push(nestedList.get(i));
        }
    }

    @Override
    public Integer next() {
        return stack.pop().getInteger();
    }

    @Override
    public boolean hasNext() {
        while(!stack.isEmpty()) {
            NestedInteger curr = stack.peek();
            if(curr.isInteger()) {
                return true;
            }
            stack.pop();
            for(int i = curr.getList().size() - 1; i >= 0; i--) {
                stack.push(curr.getList().get(i));
            }
        }
        return false;
    }
}
// stack pop(), peek(), both return E

113	Path Sum II
Given a binary tree and a sum, find all root-to-leaf paths where each path's sum equals the given sum.'

For example:
Given the below binary tree and sum = 22,
              5
             / \
            4   8
           /   / \
          11  13  4
         /  \    / \
        7    2  5   1
return
[
   [5,4,11,2],
   [5,8,4,5]
]

public class Solution {
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        ArrayList<List<Integer>> lists = new ArrayList<List<Integer>>();
        if (root == null) {
            return lists;
        }
        helper(root, new ArrayList<Integer>(), lists, sum);
        return lists;
    }
    
    public void helper(TreeNode root, List<Integer> list, List<List<Integer>> lists, int sum) {
        if (root.left == null && root.right == null) {
            ArrayList<Integer> tmp = new ArrayList<Integer>(list);
            if (root.val == sum) {
                tmp.add(root.val);
                lists.add(tmp);
            }
            return;
        }
            ArrayList<Integer> tmp = new ArrayList<Integer>(list);
            tmp.add(root.val);
            sum -= root.val;
            if (root.left != null) {
                helper(root.left, tmp, lists, sum);
            }
            if (root.right != null) {
                helper(root.right, tmp, lists, sum);
            }
    }
}

55	Jump Game
Given an array of non-negative integers, you are initially positioned at the first index of the array.
Each element in the array represents your maximum jump length at that position.
Determine if you are able to reach the last index.

For example:
A = [2,3,1,1,4], return true.
A = [3,2,1,0,4], return false.

public class Solution {
    public boolean canJump(int[] nums) {
        int i = 0;
        int n = nums.length;
        int far = 0;
        for (;i<=far&&i<n;++i){
            far = Math.max(far,nums[i]+i);
        }
        return i==n;
    }
}

public class Solution {
    public boolean canJump(int[] nums) {
        if (nums.length == 0) {
            return true;
        }
        
        return helper(nums, 0, nums.length-1);
    }
    public boolean helper(int[] nums, int index, int len) {
        if (index <= len && nums[index] >= len) {
            return true;
        }
        if (index <= len) {
            for (int i = nums[index]; i > 0; i--) {
                if (helper(nums, index + i, len)) {
                    return true;
                }           
            }
        }
        return false;
    }
}

92	Reverse Linked List II
Reverse a linked list from position m to n. Do it in-place and in one-pass.
For example:
Given 1->2->3->4->5->NULL, m = 2 and n = 4,

return 1->4->3->2->5->NULL.

Note:
Given m, n satisfy the following condition:
1 ≤ m ≤ n ≤ length of list.

public ListNode reverseBetween(ListNode head, int m, int n) {
    ListNode newhead = new ListNode(0);
    newhead.next = head;

    // tail1 is the (m-1)th node
    ListNode tail1 = newhead;
    int i = 1;
    while (i < m) {
        head = head.next;
        tail1 = tail1.next;
        i++;
    }

    // tail2 is the mth node
    ListNode tail2 = head;
    head = head.next;
    i++;

    while (i <= n) {
        tail2.next = head.next;

        // insert head after tail1
        head.next = tail1.next;
        tail1.next = head;

        head = tail2.next;
        i++;
    }

    return newhead.next;
}


47	Permutations II
Given a collection of numbers that might contain duplicates, return all possible unique permutations.

For example,
[1,1,2] have the following unique permutations:
[
  [1,1,2],
  [1,2,1],
  [2,1,1]
]

public class Solution {
  public List<List<Integer>> permuteUnique(int[] num) {
      Arrays.sort(num);
      List<List<Integer>> result = new ArrayList<List<Integer>>();
      List<Integer> current = new ArrayList<Integer>();
      boolean[] visited = new boolean[num.length];
      permute(result, current, num, visited);
      return result;
  }

  private void permute(List<List<Integer>> result, List<Integer> current, int[] num, boolean[] visited) {
      if (current.size() == num.length) {
          result.add(new ArrayList<Integer>(current));
          return;
      }
      for (int i=0; i<visited.length; i++) {
          if (!visited[i]) {
              if (i > 0 && num[i] == num[i-1] && visited[i-1]) {
                  return;
              }
              visited[i] = true;
              current.add(num[i]);
              permute(result, current, num, visited);
              current.remove(current.size()-1);
              visited[i] = false;
          }
      }
  }
}

200	Number of Islands
Given a 2d grid map of '1's (land) and '0's (water), count the number of islands. An island is surrounded
by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four
edges of the grid are all surrounded by water.

Example 1:

11110
11010
11000
00000
Answer: 1

Example 2:

11000
11000
00100
00011
Answer: 3

public class Solution {
    public int numIslands(char[][] grid) {
        int ans = 0;
        if (grid.length == 0) {
            return ans;
        }
        int row = grid.length;
        int col = grid[0].length;

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (grid[i][j] == '1') {
                    ans++;
                    helper(grid, i , j);
                }
            }
        }
        return ans;
    }
    
    public void helper(char[][] grid, int k, int t) {
        int row = grid.length;
        int col = grid[0].length;
        grid[k][t] = '0';
        if (t+1 < col && grid[k][t+1] == '1') {
            helper(grid,k,t+1);
        }
        if (t-1 >= 0 && grid[k][t-1] == '1') {
            helper(grid,k,t-1);
        }
        if (k+1 < row && grid[k+1][t] == '1') {
            helper(grid,k+1,t);
        }
        if (k-1 >= 0 && grid[k-1][t] == '1') {
            helper(grid,k-1,t);
        }

        return;
    }
}

267	Palindrome Permutation II

Given a string s, return all the palindromic permutations (without duplicates) of it. Return an empty list
if no palindromic permutation could be form.

For example:
Given s = "aabb", return ["abba", "baab"].
Given s = "abc", return [].

If a palindromic permutation exists, we just need to generate the first half of the string.
To generate all distinct permutations of a (half of) string, use a similar approach from: Permutations II
or Next Permutation
public class Solution {
    public List<String> generatePalindromes(String s) {  
        int[] map = new int[256];  
        int min = Integer.MAX_VALUE;  
        int max = Integer.MIN_VALUE;  
        for(char c: s.toCharArray()) {  
            map[c]++;  
            min = Math.min(min, c);  
            max = Math.max(max, c);  
        }  
        int count = 0;  
        List<String> res = new ArrayList<>();  
        int oddIndex = 0;  
        for(int i=min;i<=max;i++) {  
            if(count ==0 && map[i]%2==1) {  
                oddIndex = i;  
                count++;  
            }else if(map[i]%2 == 1){  
                return res;  
            }  
        }  
        String cur = "";  
        if(count==1) {  
            cur += (char)oddIndex;  
            map[oddIndex]--;  
        }  
        dfs(map, cur, s, res);  
        return res;          
    }  
    private void dfs(int[] map, String cur, String s, List<String> res) {  
        if(cur.length()==s.length()) {  
            res.add(cur);  
            return;  
        }  
        for(int i=0;i<map.length;i++) {  
            if(map[i]>0) {  
                map[i]-=2;  
                cur = (char)i + cur + (char)i;  
                dfs(map, cur, s, res);  
                cur = cur.substring(1, cur.length()-1);  
                map[i]+=2;  
            }  
        }  
    }
} 


49	Group Anagrams
Given an array of strings, group anagrams together.
For example, given: ["eat", "tea", "tan", "ate", "nat", "bat"], 
Return:

[
  ["ate", "eat","tea"],
  ["nat","tan"],
  ["bat"]
]
Note: All inputs will be in lower-case.

public class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        ArrayList<List<String>> lists = new ArrayList<List<String>>();
        if (strs.length == 0) {
            return lists;
        }
        
        Map<String, ArrayList<String>> map = new HashMap<String, ArrayList<String>>();
        Arrays.sort(strs);
        for (int i = 0; i < strs.length; i++) {
            char[] t = strs[i].toCharArray();
            Arrays.sort(t);
            String s = String.valueOf(t);
            if (map.get(s) == null) {
                ArrayList<String> tmp = new ArrayList<>();
                tmp.add(strs[i]);
                map.put(s,tmp);
            }else {
                ArrayList<String> tmp = map.get(s);
                tmp.add(strs[i]);
                map.put(s,tmp);
            }
        }
        for (String s:map.keySet()) {
            lists.add(map.get(s));
        }
        return lists;
    }
}

40	Combination Sum II
Given a collection of candidate numbers (C) and a target number (T), find all unique combinations in C where
the candidate numbers sums to T.
Each number in C may only be used once in the combination.

Note:
All numbers (including target) will be positive integers.
The solution set must not contain duplicate combinations.
For example, given candidate set [10, 1, 2, 7, 6, 1, 5] and target 8, 
A solution set is: 
[
  [1, 7],
  [1, 2, 5],
  [2, 6],
  [1, 1, 6]
]

public class Solution {
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        ArrayList<List<Integer>> lists = new ArrayList<List<Integer>>();
        if (candidates.length == 0) {
            return lists;
        }
        
        Arrays.sort(candidates);
        helper(new ArrayList<Integer>(), lists, candidates, target, 0);
        return lists;
    }
    public void helper(ArrayList<Integer> list, List<List<Integer>> lists, int[] candidates, int target, int position) {
        ArrayList<Integer> tmp = new ArrayList<Integer>(list);
        if (tmp != null && target == 0) {
            Collections.sort(tmp);
            if (!lists.contains(tmp)) {
                lists.add(tmp);
            }
            return;
        }
        for (int i = position; i < candidates.length; i++) {
            tmp.add(candidates[i]);
            if (target-candidates[i] >= 0) {
                position++;
                helper(tmp, lists, candidates, target-candidates[i], position);
            }
            tmp.remove(tmp.size()-1);
        }
    }
}

50	Pow(x, n)
public class Solution {
    public double myPow(double x, int n) {
      double result = 1.0;
      for(int i = n; i != 0; i /= 2, x *= x) {
          if( i % 2 != 0 ) {
              result *= x;
          }
      }
      return n < 0 ? 1.0 / result : result;
    }
}


131	Palindrome Partitioning
Given a string s, partition s such that every substring of the partition is a palindrome.
Return all possible palindrome partitioning of s.

For example, given s = "aab",
Return

[
  ["aa","b"],
  ["a","a","b"]
]

public class Solution {
    public List<List<String>> partition(String s) {
       List<List<String>> res = new ArrayList<List<String>>();
       List<String> list = new ArrayList<String>();
       dfs(s,0,list,res);
       return res;
    }

    public void dfs(String s, int pos, List<String> list, List<List<String>> res){
        if(pos==s.length() && list.size()>0) res.add(new ArrayList<String>(list));
        else{
            for(int i=pos;i<s.length();i++){
                if(isPal(s,pos,i)){
                    list.add(s.substring(pos,i+1));
                    dfs(s,i+1,list,res);
                    list.remove(list.size()-1);
                }
            }
        }
    }

    public boolean isPal(String s, int low, int high){
        while(low<high) if(s.charAt(low++)!=s.charAt(high--)) return false;
        return true;
    }

}

134	Gas Station
There are N gas stations along a circular route, where the amount of gas at station i is gas[i].
You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from station i to its next
station (i+1). You begin the journey with an empty tank at one of the gas stations.
Return the starting gas station's index if you can travel around the circuit once, otherwise return -1.

public class Solution {
    public int canCompleteCircuit(int[] gas, int[] cost) {
        int totalGas = 0;
        int totalCost = 0;
        int tank = 0;
        int begin = 0;
        for (int i = 0; i < gas.length; i++)
        {
            totalGas += gas[i];
            totalCost += cost[i];
            tank += (gas[i] - cost[i]);
            if (tank < 0)
            {
                begin = i + 1;
                tank = 0;
            }
        }

        return (totalGas >= totalCost ? begin : -1);
    }
}

207	Course Schedule
There are a total of n courses you have to take, labeled from 0 to n - 1.
Some courses may have prerequisites, for example to take course 0 you have to first take course 1, 
which is expressed as a pair: [0,1]

Given the total number of courses and a list of prerequisite pairs, is it possible for you to finish all 
courses?

For example:
2, [[1,0]]
There are a total of 2 courses to take. To take course 1 you should have finished course 0. So it is 
possible.

2, [[1,0],[0,1]]
There are a total of 2 courses to take. To take course 1 you should have finished course 0, and to take 
course 0 you should also have finished course 1. So it is impossible.}

public class Solution {
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        Map<Integer, ArrayList<Integer>> map = new HashMap<Integer, ArrayList<Integer>>();
        int[] indegree = new int[numCourses];
        Queue<Integer> queue = new LinkedList<Integer>();
        int count = numCourses;
        for (int i = 0; i < numCourses; i++) {
            map.put(i, new ArrayList<Integer>());
        }
        for (int i = 0; i < prerequisites.length; i++) {
            map.get(prerequisites[i][0]).add(prerequisites[i][1]);
            indegree[prerequisites[i][1]]++;
        }
        for (int i = 0; i < numCourses; i++) {
            if (indegree[i] == 0) {
                queue.offer(i);
            }
        }
        while (!queue.isEmpty()) {
            int current = queue.poll();
            for (int i : map.get(current)) {
                if (--indegree[i] == 0) {
                    queue.offer(i);
                }
            }
            count--;
        }
        return count == 0;
    }
}

210	Course Schedule II
There are a total of n courses you have to take, labeled from 0 to n - 1.
Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is
expressed as a pair: [0,1]

Given the total number of courses and a list of prerequisite pairs, return the ordering of courses you 
should take to finish all courses.

There may be multiple correct orders, you just need to return one of them. If it is impossible to finish all 
courses, return an empty array.

For example:
2, [[1,0]]
There are a total of 2 courses to take. To take course 1 you should have finished course 0. So the correct 
course order is [0,1]

4, [[1,0],[2,0],[3,1],[3,2]]
There are a total of 4 courses to take. To take course 3 you should have finished both courses 1 and 2. Both 
courses 1 and 2 should be taken after you finished course 0. So one correct course order is [0,1,2,3]. 
Another correct ordering is[0,2,1,3].

Note:
The input prerequisites is a graph represented by a list of edges, not adjacency matrices. Read more about 
how a graph is represented.

public class Solution {
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        List<List<Integer>> adj = new ArrayList<>(numCourses);
        for (int i = 0; i < numCourses; i++) adj.add(i, new ArrayList<>());
        for (int i = 0; i < prerequisites.length; i++) adj.get(prerequisites[i][1]).add(prerequisites[i][0]);
        boolean[] visited = new boolean[numCourses];
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < numCourses; i++) {
            if (!topologicalSort(adj, i, stack, visited, new boolean[numCourses])) return new int[0];
        }
        int i = 0;
        int[] result = new int[numCourses];
        while (!stack.isEmpty()) {
            result[i++] = stack.pop();
        }
        return result;
    }

    private boolean topologicalSort(List<List<Integer>> adj, int v, Stack<Integer> stack, boolean[] visited, boolean[] isLoop) {
        if (visited[v]) return true;
        if (isLoop[v]) return false;
        isLoop[v] = true;
        for (Integer u : adj.get(v)) {
            if (!topologicalSort(adj, u, stack, visited, isLoop)) return false;
        }
        visited[v] = true;
        stack.push(v);
        return true;
    }
}



209	Minimum Size Subarray Sum
Given an array of n positive integers and a positive integer s, find the minimal length of a subarray of
 which the sum ≥ s. If there isn't one, return 0 instead.'
For example, given the array [2,3,1,2,4,3] and s = 7,
the subarray [4,3] has the minimal length under the problem constraint.

O(N) - keep a moving window expand until sum>=s, then shrink util sum<s. Each time after shrinking, 
update length. (similar to other solutions, just removed unnecessary min value assignment)

public class Solution {
    public int minSubArrayLen(int s, int[] nums) {
        int i = 0, j = 0, sum = 0, min = Integer.MAX_VALUE;
        while (j < nums.length) {
            while (sum < s && j < nums.length) sum += nums[j++];
            if(sum>=s){
                while (sum >= s && i < j) sum -= nums[i++];
                min = Math.min(min, j - i + 1);
            }
        }
        return min == Integer.MAX_VALUE ? 0 : min;
    }
}


82	Remove Duplicates from Sorted List II
Given a sorted linked list, delete all nodes that have duplicate numbers, leaving only distinct numbers 
from the original list.

For example,
Given 1->2->3->3->4->4->5, return 1->2->5.
Given 1->1->1->2->3, return 2->3.

public class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        
        ArrayList<Integer> ans = new ArrayList<>();
        ArrayList<Integer> dup = new ArrayList<>();
        while (head != null) {
            if (dup.contains(head.val)) {
                dup.add(head.val);
            }else if (ans.contains(head.val)) {
                dup.add(head.val);
                ans.remove(ans.size()-1);
            }else {
                ans.add(head.val);
            }
            head = head.next;
        }
        
        if (ans.size() == 0) {
            return null;
        }
        ListNode newhead = new ListNode(ans.get(0));
        ListNode cur = newhead;
        for (int i = 1; i < ans.size(); i++) {
            cur.next = new ListNode(ans.get(i));
            cur = cur.next;
        }
        return newhead;
    }
}


public class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null) return null;
    
        if (head.next != null && head.val == head.next.val) {
            while (head.next != null && head.val == head.next.val) {
                head = head.next;
            }
            return deleteDuplicates(head.next);
        } else {
            head.next = deleteDuplicates(head.next);
        }
        return head;
    }
}

public ListNode deleteDuplicates(ListNode head) {
    if(head==null) return null;
    ListNode FakeHead=new ListNode(0);
    FakeHead.next=head;
    ListNode pre=FakeHead;
    ListNode cur=head;
    while(cur!=null){
        while(cur.next!=null&&cur.val==cur.next.val){
            cur=cur.next;
        }
        if(pre.next==cur){
            pre=pre.next;
        }
        else{
            pre.next=cur.next;
        }
        cur=cur.next;
    }
    return FakeHead.next;
}

271	Encode and Decode Strings
The string may contain any possible characters out of 256 valid ascii characters. Your algorithm should be
generalized enough to work
on any possible characters.
Do not use class member/global/static variables to store states. Your encode and decode algorithms should 
be stateless.
Do not rely on any library method such as eval or serialize methods. You should implement your own 
encode/decode algorithm.

public class Codec {
    // Encodes a list of strings to a single string.
    public String encode(List<String> strs) {
        if(strs == null || strs.size() == 0) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        for(String s : strs) {
            int len = s.length();
            sb.append(len);
            sb.append('/');
            sb.append(s);
        }
        return sb.toString();
    }

    // Decodes a single string to a list of strings.
    public List<String> decode(String s) {
        List<String> res = new ArrayList<>();
        if(s == null ||s.length() == 0) {
            return res;
        }
        int index = 0;
        while(index < s.length()) {
            int forwardSlashIndex = s.indexOf('/', index);
            int len = Integer.parseInt(s.substring(index, forwardSlashIndex));
            res.add(s.substring(forwardSlashIndex + 1, forwardSlashIndex + 1 + len));
            index = forwardSlashIndex + 1 + len;
        }
        return res;
    }
}

310	Minimum Height Trees
For a undirected graph with tree characteristics, we can choose any node as the root. The result graph is 
then a rooted tree. Among all possible rooted trees, those with minimum height are called minimum height 
trees (MHTs). Given such a graph, write a function to find all the MHTs and return a list of their root 
labels.

Format
The graph contains n nodes which are labeled from 0 to n - 1. You will be given the number n and a list 
of undirected edges (each edge is a pair of labels).

You can assume that no duplicate edges will appear in edges. Since all edges are undirected, [0, 1] is 
the same as [1, 0] and thus will not appear together in edges.

Example 1:
Given n = 4, edges = [[1, 0], [1, 2], [1, 3]]

        0
        |
        1
       / \
      2   3
return [1]

Example 2:
Given n = 6, edges = [[0, 3], [1, 3], [2, 3], [4, 3], [5, 4]]

     0  1  2
      \ | /
        3
        |
        4
        |
        5
return [3, 4]

public List<Integer> findMinHeightTrees(int n, int[][] edges) {
    if (n == 1) return Collections.singletonList(0);

    List<Set<Integer>> adj = new ArrayList<>(n);
    for (int i = 0; i < n; ++i) adj.add(new HashSet<>());
    for (int[] edge : edges) {
        adj.get(edge[0]).add(edge[1]);
        adj.get(edge[1]).add(edge[0]);
    }

    List<Integer> leaves = new ArrayList<>();
    for (int i = 0; i < n; ++i)
        if (adj.get(i).size() == 1) leaves.add(i);

    while (n > 2) {
        n -= leaves.size();
        List<Integer> newLeaves = new ArrayList<>();
        for (int i : leaves) {
            int j = adj.get(i).iterator().next();// 为了把取出int，而不是set
            adj.get(j).remove(i);
            if (adj.get(j).size() == 1) newLeaves.add(j);
        }
        leaves = newLeaves;
    }
    return leaves;
}
First let's review some statement for tree in graph theory:

(1) A tree is an undirected graph in which any two vertices are connected by exactly one path.
(2) Any connected graph who has n nodes with n-1 edges is a tree.
(3) The degree of a vertex of a graph is the number of edges incident to the vertex.
(4) A leaf is a vertex of degree 1. An internal vertex is a vertex of degree at least 2.
(5) A path graph is a tree with two or more vertices that is not branched at all.
(6) A tree is called a rooted tree if one vertex has been designated the root.
(7) The height of a rooted tree is the number of edges on the longest downward path between root and a leaf.
OK. Let's stop here and look at our problem.

Our problem want us to find the minimum height trees and return their root labels. First we can think about
 a simple case -- a path graph.
For a path graph of n nodes, find the minimum height trees is trivial. Just designate the middle point(s) 
as roots.

Despite its triviality, let design a algorithm to find them.
Suppose we don't know n, nor do we have random access of the nodes. We have to traversal. It is very easy 
to get the idea of two pointers.
One from each end and move at the same speed. When they meet or they are one step away, (depends on the 
	parity of n), we have the roots we want.

This gives us a lot of useful ideas to crack our real problem.
For a tree we can do some thing similar. We start from every end, by end we mean vertex of degree 1 (aka 
	leaves). We let the pointers move the
same speed. When two pointers meet, we keep only one of them, until the last two pointers meet or one 
step away we then find the roots.

It is easy to see that the last two pointers are from the two ends of the longest path in the graph.
The actual implementation is similar to the BFS topological sort. Remove the leaves, update the degrees 
of inner vertexes. Then remove
the new leaves. Doing so level by level until there are 2 or 1 nodes left. What's left is our answer!

The time complexity and space complexity are both O(n).
Note that for a tree we always have V = n, E = n-1.

31	Next Permutation
Implement next permutation, which rearranges numbers into the lexicographically next greater permutation of 
numbers.
If such arrangement is not possible, it must rearrange it as the lowest possible order (ie, sorted in 
ascending order).
The replacement must be in-place, do not allocate extra memory.

Here are some examples. Inputs are in the left-hand column and its corresponding outputs are in the 
right-hand column.
1,2,3 → 1,3,2
3,2,1 → 1,2,3
1,1,5 → 1,5,1

Start from its last element, traverse backward to find the first one with index i that satisfy 
num[i-1] < num[i]. So, elements from num[i] to num[n-1] is reversely sorted.

To find the next permutation, we have to swap some numbers at different positions, to minimize the increased 
amount, we have to make the highest changed position as high as possible. Notice that index larger than or 
equal to i is not possible as num[i,n-1] is reversely sorted. So, we want to increase the number at 
index i-1, clearly, swap it with the smallest number between num[i,n-1] that is larger than num[i-1]. For 
example, original number is 121543321, we want to swap the '1' at position 2 with '2' at position 7.

The last step is to make the remaining higher position part as small as possible, we just have to reversely 
sort the num[i,n-1]

public class Solution {
    public void nextPermutation(int[] nums) {
        if (nums.length > 1) {
            int i;
            for (i = nums.length - 1; i > 0; i--) {
                if (nums[i] > nums[i-1]) {
                    break;
                }
            }

            if (i == 0) {
                Arrays.sort(nums);
            }else {//need to sort the element that index bigger than i in acsend order
                for (int j = nums.length - 1; j >= i; j--) {
                    if (nums[j] > nums[i-1]) {
                        int tmp = nums[j];
                        nums[j] = nums[i-1];
                        nums[i-1] = tmp;
                        break;
                    }
                }
                Arrays.sort(nums, i, nums.length);
            }
        }    
    }
}

"********************************"
333	Largest BST Subtree 
Given a binary tree, find the largest subtree which is a Binary Search Tree (BST), where largest means 
subtree with largest number of nodes in it.

Note:
A subtree must include all of its descendants.
Here's an example:

    10
    / \
   5  15
  / \   \
 1   8   7
The Largest BST Subtree in this case is the highlighted one.
The return value is the subtree's size, which is 3.
You can recursively use algorithm similar to 98. Validate Binary Search Tree at each node of the tree, 
which will result in O(nlogn) time complexity.
Follow up:
Can you figure out ways to solve it with O(n) time complexity?

Since this is not an overall boolean check, and each subtree can decide if itself is a BST, and update a 
global size variable, I have chosen to decide BST at each subtree, and pass a 3-element array up. If subtree 
is not BST, size will be -1, and parent tree will not be BST

time complexity is O(n), since each node is visited exactly once

private int largestBSTSubtreeSize = 0;
public int largestBSTSubtree(TreeNode root) {
    helper(root);
    return largestBSTSubtreeSize;
}

private int[] helper(TreeNode root) {
    // return 3-element array:
    // # of nodes in the subtree, leftmost value, rightmost value
    // # of nodes in the subtree will be -1 if it is not a BST
    int[] result = new int[3];
    if (root == null) {
        return result;
    }
    int[] leftResult = helper(root.left);
    int[] rightResult = helper(root.right);
    if ((leftResult[0] == 0 || leftResult[0] > 0 && leftResult[2] <= root.val) &&
        (rightResult[0] == 0 || rightResult[0] > 0 && rightResult[1] >= root.val)) {
       int size = 1 + leftResult[0] + rightResult[0];
       largestBSTSubtreeSize = Math.max(largestBSTSubtreeSize, size);
       int leftBoundary = leftResult[0] == 0 ? root.val : leftResult[1];
       int rightBoundary = rightResult[0] == 0 ? root.val : rightResult[2];
       result[0] = size;
       result[1] = leftBoundary;
       result[2] = rightBoundary;
    } else {
        result[0] = -1;
    }
    return result;
}


355	Design Twitter
Design a simplified version of Twitter where users can post tweets, follow/unfollow another user and is able
to see the 10 most recent tweets in the user's news feed. Your design should support the following methods:

postTweet(userId, tweetId): Compose a new tweet.
getNewsFeed(userId): Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed
must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent
to least recent.
follow(followerId, followeeId): Follower follows a followee.
unfollow(followerId, followeeId): Follower unfollows a followee.

I use a map to track the tweets for each user. When we need to generate a news feed, I merge the news feed for
all the followees and take the most recent 10. This is unlikely to perform, but the code passes the OJ. I'm'
sure design interviews ask for performance trade-offs and just posting this code in a design interview will 
not help you get an offer.

public class Twitter {
    Map<Integer, Set<Integer>> fans = new HashMap<>();
    Map<Integer, LinkedList<Tweet>> tweets = new HashMap<>();
    int cnt = 0;

    public void postTweet(int userId, int tweetId) {
        if (!fans.containsKey(userId)) fans.put(userId, new HashSet<>());
        fans.get(userId).add(userId);
        if (!tweets.containsKey(userId)) tweets.put(userId, new LinkedList<>());
        tweets.get(userId).addFirst(new Tweet(cnt++, tweetId));
    }

    public List<Integer> getNewsFeed(int userId) {
        if (!fans.containsKey(userId)) return new LinkedList<>();
        PriorityQueue<Tweet> feed = new PriorityQueue<>((t1, t2) -> t2.time - t1.time);
        fans.get(userId).forEach(f -> tweets.get(f).forEach(feed::add));
        List<Integer> res = new LinkedList<>();
        while (feed.size() > 0 && res.size() < 10) res.add(feed.poll().id);
        return res;
    }

    public void follow(int followerId, int followeeId) {
        if (!fans.containsKey(followerId)) fans.put(followerId, new HashSet<>());
        fans.get(followerId).add(followeeId);
    }

    public void unfollow(int followerId, int followeeId) {
        if (fans.containsKey(followerId) && followeeId != followerId) fans.get(followerId).remove(followeeId);
    }

    class Tweet {
        int time;
        int id;

        Tweet(int time, int id) {
            this.time = time;
            this.id = id;
        }
    }
}


187	Repeated DNA Sequences
All DNA is composed of a series of nucleotides abbreviated as A, C, G, and T, for example: "ACGAATTCCG". When 
studying DNA, it is sometimes useful to identify repeated sequences within the DNA.

Write a function to find all the 10-letter-long sequences (substrings) that occur more than once in a DNA 
molecule.

For example,
Given s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT",

Return:
["AAAAACCCCC", "CCCCCAAAAA"].
public class Solution {
    public List<String> findRepeatedDnaSequences(String s) {
        List<String> res = new ArrayList<String>();
        Set<String> resset = new HashSet<String>();
        if(s == null || s.length() <= 10){
            return res;
        }
        Set<String> set = new HashSet<String>();
        int len = s.length();
        for(int i = 0; i <= len - 10; i++){
            String sub = s.substring(i, i + 10);
            if(!set.add(sub)){
                resset.add(sub);
            }
        }
        res.addAll(resset);
        return res;
    }
}

229	Majority Element II
Given an integer array of size n, find all elements that appear more than ⌊ n/3 ⌋ times. 
The algorithm should run in linear time and in O(1) space.
public class Solution {
    public List<Integer> majorityElement(int[] nums) {
        ArrayList<Integer> list = new ArrayList<>();
        if (nums.length == 0) {
            return list;
        }
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.get(nums[i]) == null) {
                map.put(nums[i], 1);
            }else {
                int t = map.get(nums[i]);
                map.put(nums[i],t+1);
            }
        }
        for (int key:map.keySet()) {
            if (map.get(key) > nums.length/3) {
                list.add(key);
            }
        }
        return list;
    }
}

306	Additive Number
Additive number is a string whose digits can form additive sequence.
A valid additive sequence should contain at least three numbers. Except for the first two numbers, each 
subsequent number in the sequence must be the sum of the preceding two.

For example:
"112358" is an additive number because the digits can form an additive sequence: 1, 1, 2, 3, 5, 8.

1 + 1 = 2, 1 + 2 = 3, 2 + 3 = 5, 3 + 5 = 8
"199100199" is also an additive number, the additive sequence is: 1, 99, 100, 199.
1 + 99 = 100, 99 + 100 = 199
Note: Numbers in the additive sequence cannot have leading zeros, so sequence 1, 2, 03 or 1, 02, 3 is 
invalid.

Given a string containing only digits '0'-'9', write a function to determine if it's an additive number.'

Follow up:
How would you handle overflow for very large input integers?

其实只要前两个数固定了，后面是否能划分就是确定的了。
因为前两个数决定了第三个数，第三个数和第二个数决定了第四个。。。
所以，枚举前两个数的终点位置，进行递归判断即可。

public class Solution {
    public boolean isAdditiveNumber(String num) {
        int n = num.length();
        for (int i = 1; i <= n / 2; ++i) {
            if (num.charAt(0) == '0' && i > 1) return false;
            BigInteger x1 = new BigInteger(num.substring(0, i));
            for (int j = 1; Math.max(j, i) <= n - i - j; ++j) {
                if (num.charAt(i) == '0' && j > 1) break;
                BigInteger x2 = new BigInteger(num.substring(i, i + j));
                if (isValid(x1, x2, j + i, num)) return true;
            }
        }
        return false;
    }
    private boolean isValid(BigInteger x1, BigInteger x2, int start, String num) {
        if (start == num.length()) return true;
        x2 = x2.add(x1);
        x1 = x2.subtract(x1);
        String sum = x2.toString();
        return num.startsWith(sum, start) && isValid(x1, x2, start + sum.length(), num);
    }
}
// Runtime: 8ms
Since isValid is a tail recursion it is very easy to turn it into a loop.

Java Iterative

public class Solution {
    public boolean isAdditiveNumber(String num) {
        int n = num.length();
        for (int i = 1; i <= n / 2; ++i)
            for (int j = 1; Math.max(j, i) <= n - i - j; ++j)
                if (isValid(i, j, num)) return true;
        return false;
    }
    private boolean isValid(int i, int j, String num) {
        if (num.charAt(0) == '0' && i > 1) return false;
        if (num.charAt(i) == '0' && j > 1) return false;
        String sum;
        BigInteger x1 = new BigInteger(num.substring(0, i));
        BigInteger x2 = new BigInteger(num.substring(i, i + j));
        for (int start = i + j; start != num.length(); start += sum.length()) {
            x2 = x2.add(x1);
            x1 = x2.subtract(x1);
            sum = x2.toString();
            if (!num.startsWith(sum, start)) return false;
        }
        return true;
    }
}
// Runtime: 9ms
If no overflow, instead of BigInteger we can consider to use Long which is a lot faster.

Java Iterative Using Long

public class Solution {
    public boolean isAdditiveNumber(String num) {
        int n = num.length();
        for (int i = 1; i <= n / 2; ++i)
            for (int j = 1; Math.max(j, i) <= n - i - j; ++j)
                if (isValid(i, j, num)) return true;
        return false;
    }
    private boolean isValid(int i, int j, String num) {
        if (num.charAt(0) == '0' && i > 1) return false;
        if (num.charAt(i) == '0' && j > 1) return false;
        String sum;
        Long x1 = Long.parseLong(num.substring(0, i));
        Long x2 = Long.parseLong(num.substring(i, i + j));
        for (int start = i + j; start != num.length(); start += sum.length()) {
            x2 = x2 + x1;
            x1 = x2 - x1;
            sum = x2.toString();
            if (!num.startsWith(sum, start)) return false;
        }
        return true;
    }
}
// Runtime: 3ms


139	Word Break
Given a string s and a dictionary of words dict, determine if s can be segmented into a space-separated 
sequence of one or more dictionary words.

For example, given
s = "leetcode",
dict = ["leet", "code"].

Return true because "leetcode" can be segmented as "leet code".

public boolean wordBreak(String s, Set<String> wordDict) {
    boolean[] dp = new boolean[s.length()];
    for(int i=0; i<s.length(); i++)
        for(int j=i; j>=0; j--)
            if(wordDict.contains(s.substring(j,i+1)) && (j == 0 || dp[j-1])){
                dp[i] = true;
                break;
            }
    return dp[s.length()-1];
}

69	Sqrt(x)
Implement int sqrt(int x).
public class Solution {
    public int mySqrt(int x) {
        if(x < 4) return x == 0 ? 0 : 1;
        int res = 2 * mySqrt(x/4);
        if((res+1) * (res+1) <= x && (res+1) * (res+1) >= 0) return res+1;
        return res;
    }
}

227	Basic Calculator II
Implement a basic calculator to evaluate a simple expression string.
The expression string contains only non-negative integers, +, -, *, / operators and empty spaces . The 
integer division should truncate toward zero.
You may assume that the given expression is always valid.

Some examples:
"3+2*2" = 7
" 3/2 " = 1
" 3+5 / 2 " = 5
Note: Do not use the eval built-in library function.
"100000000/1/2/3/4/5/6/7/8/9/10"?
public class Solution {
    public int calculate(String s) {
        int len;
        if(s==null || (len = s.length())==0) return 0;
        Stack<Integer> stack = new Stack<Integer>();
        int num = 0;
        char sign = '+';
        for(int i=0;i<len;i++){
            if(Character.isDigit(s.charAt(i))){
                num = num*10+s.charAt(i)-'0';
            }
            if((!Character.isDigit(s.charAt(i)) &&' '!=s.charAt(i)) || i==len-1){
                if(sign=='-'){
                    stack.push(-num);
                }
                if(sign=='+'){
                    stack.push(num);
                }
                if(sign=='*'){
                    stack.push(stack.pop()*num);
                }
                if(sign=='/'){
                    stack.push(stack.pop()/num);
                }
                sign = s.charAt(i);
                num = 0;
            }
        }
    
        int re = 0;
        for(int i:stack){
            re += i;
        }
        return re;
    }
}

222	Count Complete Tree Nodes
Given a complete binary tree, count the number of nodes.
In a complete binary tree every level, except possibly the last, is completely filled, and all nodes in the 
last level are as far left as possible. It can have between 1 and 2h nodes at the last level h.[19]

public class Solution {
    public int countNodes(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int l = leftHeight(root.left);
        int r = leftHeight(root.right);
        if (l == r) { // left side is full
            return countNodes(root.right) + (1<<l);
        } 
        return countNodes(root.left) + (1<<r);
    }
    
    private int leftHeight(TreeNode node) {
        int h = 0;
        while (node != null) {
            h++;
            node = node.left;
        }
        return h;
    }
}

The height of a tree can be found by just going left. Let a single node tree have height 0. Find the height
 h of the whole tree. If the whole tree is empty, i.e., has height -1, there are 0 nodes.

Otherwise check whether the height of the right subtree is just one less than that of the whole tree, 
meaning left and right subtree have the same height.

If yes, then the last node on the last tree row is in the right subtree and the left subtree is a full tree 
of height h-1. So we take the 2^h-1 nodes of the left subtree plus the 1 root node plus recursively the 
number of nodes in the right subtree.
If no, then the last node on the last tree row is in the left subtree and the right subtree is a full tree 
of height h-2. So we take the 2^(h-1)-1 nodes of the right subtree plus the 1 root node plus recursively the number of nodes in the left subtree.
Since I halve the tree in every recursive step, I have O(log(n)) steps. Finding a height costs O(log(n)). 
So overall O(log(n)^2).


60	Permutation Sequence
The set [1,2,3,…,n] contains a total of n! unique permutations.
By listing and labeling all of the permutations in order,
We get the following sequence (ie, for n = 3):

"123"
"132"
"213"
"231"
"312"
"321"
Given n and k, return the kth permutation sequence.
Note: Given n will be between 1 and 9 inclusive.
The logic is as follows: for n numbers the permutations can be divided to (n-1)! groups, for n-1 numbers 
can be divided to (n-2)! groups,
and so on. Thus k/(n-1)! indicates the index of current number, and k%(n-1)! denotes remaining index for 
the remaining n-1 numbers.
We keep doing this until n reaches 0, then we get n numbers permutations that is kth.

public String getPermutation(int n, int k) {
    List<Integer> num = new LinkedList<Integer>();
    for (int i = 1; i <= n; i++) num.add(i);
    int[] fact = new int[n];  // factorial
    fact[0] = 1;
    for (int i = 1; i < n; i++) fact[i] = i*fact[i-1];
    k = k-1;
    StringBuilder sb = new StringBuilder();
    for (int i = n; i > 0; i--){
        int ind = k/fact[i-1];
        k = k%fact[i-1];
        sb.append(num.get(ind));
        num.remove(ind);
    }
    return sb.toString();
}

We know how to calculate the number of permutations of n numbers... n! So each of those with permutations 
of 3 numbers means there are 6 possible permutations. Meaning there would be a total of 24 permutations in
this particular one. So if you were to look for the (k = 14) 14th permutation, it would be in the 
3 + (permutations of 1, 2, 4) subset.

To programmatically get that, you take k = 13 (subtract 1 because of things always starting at 0) and 
divide that by the 6 we got from the factorial, which would give you the index of the number you want. 
In the array {1, 2, 3, 4}, k/(n-1)! = 13/(4-1)! = 13/3! = 13/6 = 2. The array {1, 2, 3, 4} has a value of 
3 at index 2. So the first number is a 3.

Then the problem repeats with less numbers. The permutations of {1, 2, 4} would be:
1 + (permutations of 2, 4) 
2 + (permutations of 1, 4) 
4 + (permutations of 1, 2)

But our k is no longer the 14th, because in the previous step, we've already eliminated the 12 4-number 
permutations starting with 1 and 2. So you subtract 12 from k.. which gives you 1. Programmatically that 
would be...

k = k - (index from previous) * (n-1)! = k - 2(n-1)! = 13 - 2(3)! = 1

In this second step, permutations of 2 numbers has only 2 possibilities, meaning each of the three 
permutations listed above a has two possibilities, giving a total of 6. We're looking for the first one, 
so that would be in the 1 + (permutations of 2, 4) subset.

Meaning: index to get number from is k / (n - 2)! = 1 / (4-2)! = 1 / 2! = 0.. from {1, 2, 4}, index 0 is 1
so the numbers we have so far is 3, 1... and then repeating without explanations.
{2, 4} k = k - (index from pervious) * (n-2)! = k - 0 * (n - 2)! = 1 - 0 = 1; 
third number's index = k / (n - 3)! = 1 / (4-3)! = 1/ 1! = 1... from {2, 4}, index 1 has 4 
Third number is 4 {2} 
k = k - (index from pervious) * (n - 3)! = k - 1 * (4 - 3)! = 1 - 1 = 0; 
third number's index = k / (n - 4)! = 0 / (4-4)! = 0/ 1 = 0... from {2}, index 0 has 2 Fourth number is 2
Giving us 3142. If you manually list out the permutations using DFS method, it would be 3142. Done! It 
really was all about pattern finding.


208	Implement Trie (Prefix Tree)
Implement a trie with insert, search, and startsWith methods.
public class TrieNode {
    boolean isEndOfWord;
    TrieNode[] children;

    // Initialize your data structure here.
    public TrieNode() {
        this.isEndOfWord = false;
        this.children = new TrieNode[26];
    }
}

public class Trie {
    private TrieNode root;

    public Trie() {
        root = new TrieNode();
    }

    // Inserts a word into the trie.
    public void insert(String word) {
        TrieNode runner = root;
        for(char c : word.toCharArray()){
            if(runner.children[c-'a'] == null) {
                runner.children[c-'a'] = new TrieNode();
            }
            runner = runner.children[c-'a'];
        }
        runner.isEndOfWord = true;
    }

    // Returns if the word is in the trie.
    public boolean search(String word) {
        TrieNode runner = root;
        for(char c : word.toCharArray()) {
            if(runner.children[c-'a'] == null) {
                return false;
            } else {
                runner = runner.children[c-'a'];
            }
        }
        return runner.isEndOfWord;
    }

    // Returns if there is any word in the trie
    // that starts with the given prefix.
    public boolean startsWith(String prefix) {
        TrieNode runner = root;
        for(char c : prefix.toCharArray()) {
            if(runner.children[c-'a'] == null) {
                return false;
            } else {
                runner = runner.children[c-'a'];
            }
        }
        return true;
    }
}

148	Sort List
Sort a linked list in O(n log n) time using constant space complexity.
public class Solution {
    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        
        ArrayList<Integer> list = new ArrayList<>();
        list.add(head.val);
        head = head.next;
        while (head != null) {
            int low = 0; int hi = list.size()-1;
            while (low <= hi) {
                int m = low + (hi-low)/2;
                if (head.val == list.get(m)) {
                    break;
                }else if (head.val > list.get(m)) {
                    low = m+1;
                }else {
                    hi = m-1;
                }
            }
            list.add(low+(hi-low)/2, head.val);
            head = head.next;
        }
        ListNode cur = new ListNode(0);
        ListNode new_head = cur;
        for (int i = 0; i < list.size();i++) {
            cur.next = new ListNode(list.get(i));
            cur = cur.next;
        }
        return new_head.next;
    }
}

322	Coin Change
You are given coins of different denominations and a total amount of money amount. Write a function to 
compute the fewest number of coins that you need to make up that amount. If that amount of money cannot be 
made up by any combination of the coins, return -1.

Example 1:
coins = [1, 2, 5], amount = 11
return 3 (11 = 5 + 5 + 1)

Example 2:
coins = [2], amount = 3
return -1.

Note:
You may assume that you have an infinite number of each kind of coin.

public class Solution {
	public int coinChange(int[] coins, int amount) {
	    if(amount<1) return 0;
	    return helper(coins, amount, new int[amount]);
	}

	private int helper(int[] coins, int rem, int[] count) { // rem: remaining coins after the last step; count[rem]: minimum number of coins to sum up to rem
	    if(rem<0) return -1; // not valid
	    if(rem==0) return 0; // completed
	    if(count[rem-1] != 0) return count[rem-1]; // already computed, so reuse
	    int min = Integer.MAX_VALUE;
	    for(int coin : coins) {
	        int res = helper(coins, rem-coin, count);
	        if(res>=0 && res < min)
	            min = 1+res;
	    }
	    count[rem-1] = (min==Integer.MAX_VALUE) ? -1 : min;
	    return count[rem-1];
	}
}
https://leetcode.com/discuss/76217/java-both-iterative-recursive-solutions-with-explanations
The idea is very classic dynamic programming: think of the last step we take. Suppose we have already found 
out the best way to sum up to amount a, then for the last step, we can choose any coin type which gives us 
a remainder r where r = a-coins[i] for all i's. For every remainder, go through exactly the same process as 
before until either the remainder is 0 or less than 0 (meaning not a valid solution). With this idea, the 
only remaining detail is to store the minimum number of coins needed to sum up to r so that we don't need to 
recompute it over and over again.

Code in Java:

133	Clone Graph
Clone an undirected graph. Each node in the graph contains a label and a list of its neighbors.
OJ's' undirected graph serialization:
Nodes are labeled uniquely.

We use # as a separator for each node, and , as a separator for node label and each neighbor of the node.
As an example, consider the serialized graph {0,1,2#1,2#2,2}.

The graph has a total of three nodes, and therefore contains three parts as separated by #.

First node is labeled as 0. Connect node 0 to both nodes 1 and 2.
Second node is labeled as 1. Connect node 1 to node 2.
Third node is labeled as 2. Connect node 2 to node 2 (itself), thus forming a self-cycle.
Visually, the graph looks like the following:

       1
      / \
     /   \
    0 --- 2
         / \
         \_/
/**
 * Definition for undirected graph.
 * class UndirectedGraphNode {
 *     int label;
 *     List<UndirectedGraphNode> neighbors;
 *     UndirectedGraphNode(int x) { label = x; neighbors = new ArrayList<UndirectedGraphNode>(); }
 * };
 */

Use HashMap to look up nodes and add connection to them while performing BFS.

public class Solution {
    public UndirectedGraphNode cloneGraph(UndirectedGraphNode node) {
        if (node == null) return null;

        UndirectedGraphNode newNode = new UndirectedGraphNode(node.label); //new node for return
        HashMap<Integer, UndirectedGraphNode> map = new HashMap(); //store visited nodes

        map.put(newNode.label, newNode); //add first node to HashMap

        LinkedList<UndirectedGraphNode> queue = new LinkedList(); //to store **original** nodes need to be visited
        queue.add(node); //add first **original** node to queue

        while (!queue.isEmpty()) { //if more nodes need to be visited
            UndirectedGraphNode n = queue.pop(); //search first node in the queue
            for (UndirectedGraphNode neighbor : n.neighbors) {
                if (!map.containsKey(neighbor.label)) { //add to map and queue if this node hasn't been searched before
                    map.put(neighbor.label, new UndirectedGraphNode(neighbor.label));
                    queue.add(neighbor);
                }
                map.get(n.label).neighbors.add(map.get(neighbor.label)); //add neighbor to new created nodes
            }
        }

        return newNode;
    }
}

228	Summary Ranges
Given a sorted integer array without duplicates, return the summary of its ranges.
For example, given [0,1,2,4,5,7], return ["0->2","4->5","7"].
public class Solution {
    public List<String> summaryRanges(int[] nums) {
        ArrayList<String> list = new ArrayList<>();
        if (nums.length == 0) {
            return list;
        }
        for (int i = 0; i < nums.length; i++) {
            String tmp = "";
            tmp += nums[i];
            int j = i+1;
            if (j < nums.length && nums[i] + 1 == nums[j]) {
                while (j+1 < nums.length && nums[j] + 1 == nums[j+1]) {
                    j++;
                }
                tmp = tmp + "->" + nums[j];
                list.add(tmp);
                i = j;
            }else {
                list.add(tmp);
            }
        }
        return list;
    }
}

332	Reconstruct Itinerary
Given a list of airline tickets represented by pairs of departure and arrival airports [from, to], 
reconstruct the itinerary in order. All of the tickets belong to a man who departs from JFK. Thus, 
the itinerary must begin with JFK.

Note:
If there are multiple valid itineraries, you should return the itinerary that has the smallest lexical 
order when read as a single string. For example, the itinerary ["JFK", "LGA"] has a smaller lexical order 
than ["JFK", "LGB"].
All airports are represented by three capital letters (IATA code).
You may assume all tickets form at least one valid itinerary.
Example 1:
tickets = [["MUC", "LHR"], ["JFK", "MUC"], ["SFO", "SJC"], ["LHR", "SFO"]]
Return ["JFK", "MUC", "LHR", "SFO", "SJC"].
Example 2:
tickets = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
Return ["JFK","ATL","JFK","SFO","ATL","SFO"].
Another possible reconstruction is ["JFK","SFO","ATL","JFK","ATL","SFO"]. But it is larger in lexical order.

All the airports are vertices and tickets are directed edges. Then all
these tickets form a directed graph.
The graph must be Eulerian since we know that a Eulerian path exists.

Thus, start from "JFK", we can apply the Hierholzer's' algorithm to find a Eulerian path
in the graph which is a valid reconstruction.

Since the problem asks for lexical order smallest solution, we can put the neighbors in
a min-heap. In this way, we always visit the smallest possible neighbor first in our trip.

public class Solution {

    Map<String, PriorityQueue<String>> flights;
    LinkedList<String> path;

    public List<String> findItinerary(String[][] tickets) {
        flights = new HashMap<>();
        path = new LinkedList<>();
        for (String[] ticket : tickets) {
            flights.putIfAbsent(ticket[0], new PriorityQueue<>());
            flights.get(ticket[0]).add(ticket[1]);
        }
        dfs("JFK");
        return path;
    }

    public void dfs(String departure) {
        PriorityQueue<String> arrivals = flights.get(departure);
        while (arrivals != null && !arrivals.isEmpty())
            dfs(arrivals.poll());
        path.addFirst(departure);
    }
}

18	4Sum
Given an array S of n integers, are there elements a, b, c, and d in S such that a + b + c + d = target? 
Find all unique quadruplets in the array which gives the sum of target.

Note: The solution set must not contain duplicate quadruplets.
For example, given array S = [1, 0, -1, 0, -2, 2], and target = 0.

A solution set is:
[
  [-1,  0, 0, 1],
  [-2, -1, 1, 2],
  [-2,  0, 0, 2]
]
public class Solution {
    public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> list = new ArrayList<List<Integer>>();
        Arrays.sort(nums);
        int second = 0, third = 0, nexti = 0, nextj = 0;
        for(int i=0, L=nums.length; i<L-3; i++) {
            if(nums[i]<<2 > target) return list; // return immediately
            for(int j=L-1; j>i+2; j--) {
                if(nums[j]<<2 < target) break; // break immediately
                int rem = target-nums[i]-nums[j];
                int lo = i+1, hi=j-1;
                while(lo<hi) {
                    int sum = nums[lo] + nums[hi];
                    if(sum>rem) --hi;
                    else if(sum<rem) ++lo;
                    else {
                        list.add(Arrays.asList(nums[i],nums[lo],nums[hi],nums[j]));
                        while(++lo<=hi && nums[lo-1]==nums[lo]) continue; // avoid duplicate results
                        while(--hi>=lo && nums[hi]==nums[hi+1]) continue; // avoid duplicate results
                    }
                }
                while(j>=1 && nums[j]==nums[j-1]) --j; // skip inner loop
            }
            while(i<L-1 && nums[i]==nums[i+1]) ++i; // skip outer loop
        }
        return list;
    }
}
To avoid duplicate list items, I skip unnecessary indices at two locations:

one at the end of the outer loop (i-loop)
the other at the end of the inner loop (j-loop).
To avoid useless computations, the following is kind of critical:

the function return immediately when nums[i]*4 > target
the inner loop break immediately when nums[j]*4 < target.
These two lines save quite some time due to the set up of the test cases in OJ.

221	Maximal Square
Given a 2D binary matrix filled with 0's and 1's, find the largest square containing all 1's and return its 
area.'

For example, given the following matrix:

1 0 1 0 0
1 0 1 1 1
1 1 1 1 1
1 0 0 1 0
Return 4.

public class Solution {
    public int maximalSquare(char[][] a) {
      if (a == null || a.length == 0 || a[0].length == 0)
        return 0;
    
      int max = 0, n = a.length, m = a[0].length;
    
      // dp(i, j) represents the length of the square 
      // whose lower-right corner is located at (i, j)
      // dp(i, j) = min{ dp(i-1, j-1), dp(i-1, j), dp(i, j-1) }
      int[][] dp = new int[n + 1][m + 1];
    
      for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
          if (a[i - 1][j - 1] == '1') {
            dp[i][j] = Math.min(dp[i - 1][j - 1], Math.min(dp[i - 1][j], dp[i][j - 1])) + 1;
            max = Math.max(max, dp[i][j]);
          }
        }
      }
    
      // return the area
      return max * max;
    }
}

43	Multiply Strings
Given two numbers represented as strings, return multiplication of the numbers as a string.

Note:
The numbers can be arbitrarily large and are non-negative.
Converting the input string to integer is NOT allowed.
You should NOT use internal library such as BigInteger.

public class Solution {
    public String multiply(String num1, String num2) {
        int n1 = num1.length(), n2 = num2.length();
        int[] products = new int[n1 + n2];
        for (int i = n1 - 1; i >= 0; i--) {
            for (int j = n2 - 1; j >= 0; j--) {
                int d1 = num1.charAt(i) - '0';
                int d2 = num2.charAt(j) - '0';
                products[i + j + 1] += d1 * d2;
            }
        }
        int carry = 0;
        for (int i = products.length - 1; i >= 0; i--) {
            int tmp = (products[i] + carry) % 10;
            carry = (products[i] + carry) / 10;
            products[i] = tmp;
        }
        StringBuilder sb = new StringBuilder();
        for (int num : products) sb.append(num);
        while (sb.length() != 0 && sb.charAt(0) == '0') sb.deleteCharAt(0);
        return sb.length() == 0 ? "0" : sb.toString();
    }
}

150	Evaluate Reverse Polish Notation
Evaluate the value of an arithmetic expression in Reverse Polish Notation.
Valid operators are +, -, *, /. Each operand may be an integer or another expression.

Some examples:
  ["2", "1", "+", "3", "*"] -> ((2 + 1) * 3) -> 9
  ["4", "13", "5", "/", "+"] -> (4 + (13 / 5)) -> 6
public class Solution {
    public int evalRPN(String[] tokens) {
        int a,b;
        Stack<Integer> S = new Stack<Integer>();
        for (String s : tokens) {
            if(s.equals("+")) {
                S.add(S.pop()+S.pop());
            }
            else if(s.equals("/")) {
                b = S.pop();
                a = S.pop();
                S.add(a / b);
            }
            else if(s.equals("*")) {
                S.add(S.pop() * S.pop());
            }
            else if(s.equals("-")) {
                b = S.pop();
                a = S.pop();
                S.add(a - b);
            }
            else {
                S.add(Integer.parseInt(s));
            }
        }   
        return S.pop();
    }
}

93	Restore IP Addresses
Given a string containing only digits, restore it by returning all possible valid IP address combinations.
For example:
Given "25525511135",
return ["255.255.11.135", "255.255.111.35"]. (Order does not matter)

3-loop divides the string s into 4 substring: s1, s2, s3, s4. Check if each substring is
valid. In isValid, strings whose length greater than 3 or equals to 0 is not valid; or if
the string's length is longer than 1 and the first letter is '0' then it's invalid; or the
string whose integer representation greater than 255 is invalid.
public class Solution {
    public List<String> restoreIpAddresses(String s) {
        List<String> res = new ArrayList<String>();
        int len = s.length();
        for(int i = 1; i<4 && i<len-2; i++){
            for(int j = i+1; j<i+4 && j<len-1; j++){
                for(int k = j+1; k<j+4 && k<len; k++){
                    String s1 = s.substring(0,i), s2 = s.substring(i,j), s3 = s.substring(j,k), s4 = s.substring(k,len);
                    if(isValid(s1) && isValid(s2) && isValid(s3) && isValid(s4)){
                        res.add(s1+"."+s2+"."+s3+"."+s4);
                    }
                }
            }
        }
        return res;
    }
    public boolean isValid(String s){
        if(s.length()>3 || s.length()==0 || (s.charAt(0)=='0' && s.length()>1) || Integer.parseInt(s)>255)
            return false;
        return true;
    }
}

2	Add Two Numbers
You are given two linked lists representing two non-negative numbers. The digits are stored in reverse order 
and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.
Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8

public class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        if (l1 == null || l2 == null) {
            return l1 == null?l2:l1;
        }
        
        int carry = 0;
        ListNode newHead = new ListNode(0);
        ListNode cur = newHead;
        while (l1 != null && l2 != null) {
            int tmp = l1.val + l2.val + carry;
            carry = tmp/10;
            tmp = tmp%10;
            cur.next = new ListNode(tmp);
            cur = cur.next;
            l1 = l1.next;
            l2 = l2.next;
        }
        while (l1 != null) {
            int tmp = l1.val + carry;
            carry = tmp/10;
            tmp = tmp%10;
            cur.next = new ListNode(tmp);
            cur = cur.next;
            l1 = l1.next;
        }
        while (l2 != null) {
            int tmp = l2.val + carry;
            carry = tmp/10;
            tmp = tmp%10;
            cur.next = new ListNode(tmp);
            cur = cur.next;
            l2 = l2.next;
        }
        if (carry != 0) {
            cur.next = new ListNode(carry);
        }
        return newHead.next;
    }
}

public class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode ln1 = l1, ln2 = l2, head = null, node = null;
        int carry = 0, remainder = 0, sum = 0;
        head = node = new ListNode(0);

        while(ln1 != null || ln2 != null || carry != 0) {
            sum = (ln1 != null ? ln1.val : 0) + (ln2 != null ? ln2.val : 0) + carry;
            carry = sum / 10;
            remainder = sum % 10;
            node = node.next = new ListNode(remainder);
            ln1 = (ln1 != null ? ln1.next : null);
            ln2 = (ln2 != null ? ln2.next : null);
        }
        return head.next;
    }
}

5	Longest Palindromic Substring
Given a string S, find the longest palindromic substring in S. You may assume that the maximum length of S 
is 1000, and there exists one unique longest palindromic substring.

public class Solution{
	public String longestPalindrome(String s) {
	    char[] ca = s.toCharArray();
	    int rs = 0, re = 0;
	    int max = 0;
	    for(int i = 0; i < ca.length; i++) {
	        if(isPalindrome(ca, i - max - 1, i)) {
	            rs = i - max - 1; re = i;
	            max += 2;
	        } else if(isPalindrome(ca, i - max, i)) {
	            rs = i - max; re = i;
	            max += 1;
	        }
	    }
	    return s.substring(rs, re + 1);
	}

	private boolean isPalindrome(char[] ca, int s, int e) {
	    if(s < 0) return false;

	    while(s < e) {
	        if(ca[s++] != ca[e--]) return false;
	    }
	    return true;
	}
}
Explanation for those interested. For every position i we're interested if there's a palindrome ending at 
this position (inclusive) longer than the longest palindrome found so far.
For i = 0 the only palindrome ending there is the palindrome of length 1. So the new palindrome has length 
1 and therefore the maximum increases by 1.

For every i > 0 we can have many palindromes ending there. We're only interested in those with length >= 2 
because we've already found one of length 1. Some of palindromes may have even lengths, some odd. The 
presence of a palindrome of length len >= 2 ending at i implies three things:

i - len + 1 >= 0 (obviously).
s[i - len + 1] == s[i].
s[i - len + 2 .. i - 1] is a palindrome. This is the most interesting fact because it means that if we have 
found a palindrome of length len - 2 ending at i - 1, then we may find another one of length len ending at 
i. So incrementing i by 1 can lead to increase of the palindrome length by 2. One more important corollary: 
if we find a palindrome of length len - 2 at i - 1, then we only need to check the conditions (1) and (2) 
above on the next iteration—we may skip the isPalindrome() call.


79	Word Search
Given a 2D board and a word, find if the word exists in the grid.
The word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those 
horizontally or vertically neighboring. The same letter cell may not be used more than once.

For example,
Given board =

[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]
word = "ABCCED", -> returns true,
word = "SEE", -> returns true,
word = "ABCB", -> returns false.

public class Solution {
    public boolean exist(char[][] board, String word) {
        if (board == null || board.length == 0 || board[0].length == 0 || word == null) return false;
        char[] target = word.toCharArray();
        boolean[][] visited = new boolean[board.length][board[0].length];
        for (int r = 0; r < board.length; r++) {
            for (int c = 0; c < board[0].length; c++) {
                if (board[r][c] == target[0]) {
                    if (helper(board, target, 0, r, c, visited)) return true;
                }
            }
        }
        return false;
    }
    
    public boolean helper(char[][] board, char[] target, int start, int i, int j, boolean[][] visited) {
        if (board[i][j] == target[start]) {
            visited[i][j] = true;
            if (start == target.length - 1) return true;
            if (i - 1 >= 0 && !visited[i - 1][j]) {
                if (helper(board, target, start + 1, i - 1, j, visited)) return true;
            }
            if (i + 1 < board.length && !visited[i + 1][j]) {
                if (helper(board, target, start + 1, i + 1, j, visited)) return true;
            }
            if (j - 1 >= 0 && !visited[i][j - 1]) {
                if (helper(board, target, start + 1, i, j - 1, visited)) return true;
            }
            if (j + 1 < board[0].length && !visited[i][j + 1]) {
                if (helper(board, target, start + 1, i, j + 1, visited)) return true;
            }
        }
        visited[i][j] = false;
        return false;
    }
}

61	Rotate List
Given a list, rotate the list to the right by k places, where k is non-negative.

For example:
Given 1->2->3->4->5->NULL and k = 2,
return 4->5->1->2->3->NULL.
public ListNode rotateRight(ListNode head, int n) {
    if (head==null||head.next==null) return head;
    ListNode dummy=new ListNode(0);
    dummy.next=head;
    ListNode fast=dummy,slow=dummy;

    int i;
    for (i=0;fast.next!=null;i++)//Get the total length 
        fast=fast.next;

    for (int j=i-n%i;j>0;j--) //Get the i-n%i th node
        slow=slow.next;

    fast.next=dummy.next; //Do the rotation
    dummy.next=slow.next;
    slow.next=null;

    return dummy.next;
}

143	Reorder List
Given a singly linked list L: L0→L1→…→Ln-1→Ln,
reorder it to: L0→Ln→L1→Ln-1→L2→Ln-2→…
You must do this in-place without altering the nodes' values.'

For example,
Given {1,2,3,4}, reorder it to {1,4,2,3}.

public void reorderList(ListNode head) {
    if(head==null||head.next==null) return;
      //Find the middle of the list
      ListNode p1=head;
      ListNode p2=head;
      while(p2.next!=null&&p2.next.next!=null){ 
          p1=p1.next;
          p2=p2.next.next;
        }
      //Reverse the half after middle  1->2->3->4->5->6 to 1->2->3->6->5->4
      ListNode preMiddle=p1;
      ListNode preCurrent=p1.next;
      while(preCurrent.next!=null){
          ListNode current=preCurrent.next;
          preCurrent.next=current.next;
          current.next=preMiddle.next;
          preMiddle.next=current;
        }
      //Start reorder one by one  1->2->3->6->5->4 to 1->6->2->5->3->4
      p1=head;
      p2=preMiddle.next;
      while(p1!=preMiddle){
          preMiddle.next=p2.next;
          p2.next=p1.next;
          p1.next=p2;
          p1=p2.next;
          p2=preMiddle.next;
      }
}

54	Spiral Matrix
Given a matrix of m x n elements (m rows, n columns), return all elements of the matrix in spiral order.

For example,
Given the following matrix:

[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
You should return [1,2,3,6,9,8,7,4,5].

public class Solution {
    public List<Integer> spiralOrder(int[][] matrix) {
        ArrayList<Integer> list = new ArrayList<>();
        if (matrix.length == 0) {
            return list;
        }
        int mount = matrix.length * matrix[0].length;
        int rs = 0, re = matrix.length-1;
        int cs = 0, ce = matrix[0].length-1;
        
        int i = 0;
        while (i < mount) {
            for (int j = cs; j <= ce && i < mount; j++) {
                list.add(matrix[rs][j]);
                i++;
            }
            rs++;
            for (int j = rs; j <= re && i < mount; j++) {
                list.add(matrix[j][ce]);
                i++;
            }
            ce--;
            for (int j = ce; j >= cs && i < mount; j--) {
                list.add(matrix[re][j]);
                i++;
            }
            re--;
            for (int j = re; j >= rs && i < mount; j--) {
                list.add(matrix[j][cs]);
                i++;
            }
            cs++;
        }
        return list;
    }
}
// pay attention to index, -1-1-1

152	Maximum Product Subarray
Find the contiguous subarray within an array (containing at least one number) which has the largest product.
For example, given the array [2,3,-2,4],
the contiguous subarray [2,3] has the largest product = 6.

public class Solution {
    public int maxProduct(int[] A) {
        if (A.length == 0) {
            return 0;
        }
    
        int maxherepre = A[0];
        int minherepre = A[0];
        int maxsofar = A[0];
        int maxhere, minhere;
    
        for (int i = 1; i < A.length; i++) {
            maxhere = Math.max(Math.max(maxherepre * A[i], minherepre * A[i]), A[i]);
            minhere = Math.min(Math.min(maxherepre * A[i], minherepre * A[i]), A[i]);
            maxsofar = Math.max(maxhere, maxsofar);
            maxherepre = maxhere;
            minherepre = minhere;
        }
        return maxsofar;
    }

}

3	Longest Substring Without Repeating Characters
Given a string, find the length of the longest substring without repeating characters.

Examples:
Given "abcabcbb", the answer is "abc", which the length is 3.
Given "bbbbb", the answer is "b", with the length of 1.
Given "pwwkew", the answer is "wke", with the length of 3. Note that the answer must be a substring, "pwke" 
is a subsequence and not a substring.

public class Solution {
    public int lengthOfLongestSubstring(String s) {
        int[] map = new int[128];
	    int max = 0, j = 0;
	    char[] str = s.toCharArray();
	    int length = s.length();

	    for(int i = 0; i < length; i++) {
	        if(map[str[i]] > 0)
	            j =  Math.max(j, map[str[i]]);
	        map[str[i]] = i + 1;
	        max = Math.max(max, i - j + 1);
	    }
	    return max;
    }
}

The idea is almost the same with a Hashmap solution, which costs about 20ms. To speed up,1) replace Hashmap 
with an array, and the index is the int value of its asicii code. 2) use char array instead of string.

304	Range Sum Query 2D - Immutable
Given a 2D matrix matrix, find the sum of the elements inside the rectangle defined by its upper left corner
(row1, col1) and lower right corner (row2, col2).

Range Sum Query 2D
The above rectangle (with the red border) is defined by (row1, col1) = (2, 1) and (row2, col2) = (4, 3), 
which contains sum = 8.

Example:
Given matrix = [
  [3, 0, 1, 4, 2],
  [5, 6, 3, 2, 1],
  [1, 2, 0, 1, 5],
  [4, 1, 0, 1, 7],
  [1, 0, 3, 0, 5]
]

sumRegion(2, 1, 4, 3) -> 8
sumRegion(1, 1, 2, 2) -> 11
sumRegion(1, 2, 2, 4) -> 12
Note:
You may assume that the matrix does not change.
There are many calls to sumRegion function.
You may assume that row1 ≤ row2 and col1 ≤ col2.

public class NumMatrix {
  private int[][] dp;

  public NumMatrix(int[][] matrix) {
      if(   matrix           == null
         || matrix.length    == 0
         || matrix[0].length == 0   ){
          return;   
      }

      int m = matrix.length;
      int n = matrix[0].length;

      dp = new int[m + 1][n + 1];
      for(int i = 1; i <= m; i++){
          for(int j = 1; j <= n; j++){
              dp[i][j] = dp[i - 1][j] + dp[i][j - 1] -dp[i - 1][j - 1] + matrix[i - 1][j - 1] ;
          }
      }
  }

  public int sumRegion(int row1, int col1, int row2, int col2) {
      int iMin = Math.min(row1, row2);
      int iMax = Math.max(row1, row2);

      int jMin = Math.min(col1, col2);
      int jMax = Math.max(col1, col2);

      return dp[iMax + 1][jMax + 1] - dp[iMax + 1][jMin] - dp[iMin][jMax + 1] + dp[iMin][jMin];    
  }
}

71	Simplify Path
Given an absolute path for a file (Unix-style), simplify it.

For example,
path = "/home/", => "/home"
path = "/a/./b/../../c/", => "/c"

public String simplifyPath(String path) {
    Deque<String> stack = new LinkedList<>();
    Set<String> skip = new HashSet<>(Arrays.asList("..",".",""));
    for (String dir : path.split("/")) {
        if (dir.equals("..") && !stack.isEmpty()) stack.pop();
        else if (!skip.contains(dir)) stack.push(dir);
    }
    String res = "";
    for (String dir : stack) res = "/" + dir + res;
    return res.isEmpty() ? "/" : res;
}

98	Validate Binary Search Tree
Given a binary tree, determine if it is a valid binary search tree (BST).
Assume a BST is defined as follows:

The left subtree of a node contains only nodes with keys less than the node's key.
The right subtree of a node contains only nodes with keys greater than the node's key.
Both the left and right subtrees must also be binary search trees.
Example 1:
    2
   / \
  1   3
Binary tree [2,1,3], return true.
Example 2:
    1
   / \
  2   3
Binary tree [1,2,3], return false.

public class Solution {
    public boolean isValidBST(TreeNode root) {
        if (root == null) {
            return true;
        }
        if (root.left == null && root.right == null) {
            return true;
        }
        return check(root, Integer.MAX_VALUE, Integer.MIN_VALUE);
    }

    public boolean check(TreeNode node, int max, int min) {
        if (node == null) {
            return true;
        }
        if (node.val > max || node.val < min) {
            return false;
        }

        // if node's value is INT_MIN, it should not have left child any more
        if (node.val == Integer.MIN_VALUE && node.left != null) {
            return false;
        }

        // if node's value is INT_MAX, it should not have right child any more
        if (node.val == Integer.MAX_VALUE && node.right != null) {
            return false;
        }

        return check(node.left, node.val - 1, min) && check(node.right, max, node.val + 1);
    }
}

353	Design Snake Game

211	Add and Search Word - Data structure design
Design a data structure that supports the following two operations:

void addWord(word)
bool search(word)
search(word) can search a literal word or a regular expression string containing only letters a-z or .. A . 
means it can represent any one letter.

For example:

addWord("bad")
addWord("dad")
addWord("mad")
search("pad") -> false
search("bad") -> true
search(".ad") -> true
search("b..") -> true
public class WordDictionary {
    class TrieNode {
        TrieNode[] child = new TrieNode[26];
        boolean isWord = false;
    }
    TrieNode root = new TrieNode();
    public void addWord(String word) {
        TrieNode p = root;
        for (char c : word.toCharArray()) {
            if (p.child[c - 'a'] == null) p.child[c - 'a'] = new TrieNode();
            p = p.child[c - 'a'];
        }
        p.isWord = true;
    }

    public boolean search(String word) {
        return helper(word, 0, root);
    }

    private boolean helper(String s, int index, TrieNode p) {
        if (index >= s.length()) return p.isWord;
        char c = s.charAt(index);
        if (c == '.') {
            for (int i = 0; i < p.child.length; i++)
                if (p.child[i] != null && helper(s, index + 1, p.child[i]))
                    return true;
            return false;
        } else return (p.child[c - 'a'] != null && helper(s, index + 1, p.child[c - 'a']));
    }
}


127	Word Ladder
Given two words (beginWord and endWord), and a dictionary's word list', find the length of shortest 
transformation sequence from beginWord to endWord, such that:

Only one letter can be changed at a time
Each intermediate word must exist in the word list
For example,

Given:
beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log"]
As one shortest transformation is "hit" -> "hot" -> "dot" -> "dog" -> "cog",
return its length 5.

Note:
Return 0 if there is no such transformation sequence.
All words have the same length.
All words contain only lowercase alphabetic characters.

public class Solution {
    public int ladderLength(String beginWord, String endWord, Set<String> wordDict) {
        int len = 1;
        Set<String> beginSet = new HashSet<>();
        Set<String> endSet = new HashSet<>();
        Set<String> visited = new HashSet<>();
        beginSet.add(beginWord);
        endSet.add(endWord);
        visited.add(beginWord);
        visited.add(endWord);

        while (!beginSet.isEmpty() && !endSet.isEmpty()) {
            // add new words to smaller set to achieve better performance
            boolean isBeginSetSmall = beginSet.size() < endSet.size();
            Set<String> small = isBeginSetSmall ? beginSet : endSet;
            Set<String> big = isBeginSetSmall ? endSet : beginSet;
            Set<String> next = new HashSet<>();
            len++;
            for (String str : small) {
                // construct all possible words
                for (int i = 0; i < str.length(); i++) {
                    for (char ch = 'a'; ch <= 'z'; ch++) {
                        StringBuilder sb = new StringBuilder(str);
                        sb.setCharAt(i, ch);
                        String word = sb.toString();
                        if (big.contains(word)) {
                            return len;
                        }
                        if (wordDict.contains(word) && !visited.contains(word)) {
                            visited.add(word);
                            next.add(word);
                        }
                    }
                }
            }
            if (isBeginSetSmall) {
                beginSet = next;
            } else {
                endSet = next;
            }
        }
        return 0;
    }

}

179	Largest Number
Given a list of non negative integers, arrange them such that they form the largest number.
For example, given [3, 30, 34, 5, 9], the largest formed number is 9534330.
Note: The result may be very large, so you need to return a string instead of an integer.

public class Solution {
    public  String largestNumber(int[] num) {
        if(num==null || num.length==0)
            return "";
        String[] Snum = new String[num.length];
        for(int i=0;i<num.length;i++)
            Snum[i] = num[i]+"";
    
        Comparator<String> comp = new Comparator<String>(){
            @Override
            public int compare(String str1, String str2){
                String s1 = str1+str2;
                String s2 = str2+str1;
                return s1.compareTo(s2);
            }
        };
    
        Arrays.sort(Snum,comp);
        if(Snum[Snum.length-1].charAt(0)=='0')
            return "0";
    
        StringBuilder sb = new StringBuilder();
    
        for(String s: Snum)
            sb.insert(0, s);
    
        return sb.toString();
    }
}

The logic is pretty straightforward. Just compare number by convert it to string.
Thanks for Java 8, it makes code beautiful.
Java:
public class Solution {
    public String largestNumber(int[] num) {
        String[] array = Arrays.stream(num).mapToObj(String::valueOf).toArray(String[]::new);
        Arrays.sort(array, (String s1, String s2) -> (s2 + s1).compareTo(s1 + s2));
        return Arrays.stream(array).reduce((x, y) -> x.equals("0") ? y : x + y).get();
    }
}

15	3Sum
Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? Find all unique 
triplets in the array which gives the sum of zero.

Note: The solution set must not contain duplicate triplets.
For example, given array S = [-1, 0, 1, 2, -1, -4],
A solution set is:
[
  [-1, 0, 1],
  [-1, -1, 2]
]

public class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new LinkedList<>();
        for(int i = 0;i< nums.length-2;i++){
            if(i==0 || (i>0 && nums[i] !=nums[i-1])){
            int lo = i+1;
            int hi = nums.length-1;
            int sum = 0 - nums[i];
            while(lo<hi){
                if(nums[lo] + nums[hi]==sum){
                    res.add(Arrays.asList(nums[i],nums[lo],nums[hi]));
                while(lo<hi && nums[lo]== nums[lo+1]) lo++;
                while(lo<hi && nums[hi] == nums[hi-1]) hi--;
                lo++; hi--; }
                else if(nums[lo]+nums[hi]<sum) {
                     while (lo < hi && nums[lo] == nums[lo+1]) lo++;
                    lo++;
                }
                else{
                    while (lo < hi && nums[hi] == nums[hi-1]) hi--;
                    hi--;
                }
            }
            }
        }
        return res;
    }
}

220	Contains Duplicate III
Given an array of integers, find out whether there are two distinct indices i and j in the array such that 
the difference between nums[i] and nums[j] is at most t and the difference between i and j is at most k.

public class Solution {
    public static boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        if (k < 1 || t < 0) return false;
        Map<Long, Long> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            long j = t == 0 ? (long) nums[i] - Integer.MIN_VALUE : (((long) nums[i] - Integer.MIN_VALUE) / t);
            if (map.containsKey(j) || (map.containsKey(j - 1) && Math.abs(map.get(j - 1) - nums[i]) <= t)
                    || (map.containsKey(j + 1) && Math.abs(map.get(j + 1) - nums[i]) <= t)) return true;
            if (map.keySet().size() == k) map.remove(map.keySet().iterator().next());
            map.put(j, (long) nums[i]);
        }
        return false;
    }
}

307	Range Sum Query - Mutable
Given an integer array nums, find the sum of the elements between indices i and j (i ≤ j), inclusive.
The update(i, val) function modifies nums by updating the element at index i to val.
Example:
Given nums = [1, 3, 5]

sumRange(0, 2) -> 9
update(1, 2)
sumRange(0, 2) -> 8
Note:
The array is only modifiable by the update function.
You may assume the number of calls to update and sumRange function is distributed evenly.

public class NumArray {

    class SegmentTreeNode {
        int start, end;
        SegmentTreeNode left, right;
        int sum;

        public SegmentTreeNode(int start, int end) {
            this.start = start;
            this.end = end;
            this.left = null;
            this.right = null;
            this.sum = 0;
        }
    }

    SegmentTreeNode root = null;

    public NumArray(int[] nums) {
        root = buildTree(nums, 0, nums.length-1);
    }

    private SegmentTreeNode buildTree(int[] nums, int start, int end) {
        if (start > end) {
            return null;
        } else {
            SegmentTreeNode ret = new SegmentTreeNode(start, end);
            if (start == end) {
                ret.sum = nums[start];
            } else {
                int mid = start  + (end - start) / 2;             
                ret.left = buildTree(nums, start, mid);
                ret.right = buildTree(nums, mid + 1, end);
                ret.sum = ret.left.sum + ret.right.sum;
            }         
            return ret;
        }
    }

    void update(int i, int val) {
        update(root, i, val);
    }

    void update(SegmentTreeNode root, int pos, int val) {
        if (root.start == root.end) {
           root.sum = val;
        } else {
            int mid = root.start + (root.end - root.start) / 2;
            if (pos <= mid) {
                 update(root.left, pos, val);
            } else {
                 update(root.right, pos, val);
            }
            root.sum = root.left.sum + root.right.sum;
        }
    }

    public int sumRange(int i, int j) {
        return sumRange(root, i, j);
    }

    public int sumRange(SegmentTreeNode root, int start, int end) {
        if (root.end == end && root.start == start) {
            return root.sum;
        } else {
            int mid = root.start + (root.end - root.start) / 2;
            if (end <= mid) {
                return sumRange(root.left, start, end);
            } else if (start >= mid+1) {
                return sumRange(root.right, start, end);
            }  else {    
                return sumRange(root.right, mid+1, end) + sumRange(root.left, start, mid);
            }
        }
    }
}

91	Decode Ways
A message containing letters from A-Z is being encoded to numbers using the following mapping:
'A' -> 1
'B' -> 2
...
'Z' -> 26
Given an encoded message containing digits, determine the total number of ways to decode it.
For example,
Given encoded message "12", it could be decoded as "AB" (1 2) or "L" (12).
The number of ways decoding "12" is 2.

public class Solution {
    public int numDecodings(String s) {
        int n = s.length();
        if (n == 0) return 0;

        int[] memo = new int[n+1];
        memo[n]  = 1;
        memo[n-1] = s.charAt(n-1) != '0' ? 1 : 0;

        for (int i = n - 2; i >= 0; i--)
            if (s.charAt(i) == '0') continue;
            else memo[i] = (Integer.parseInt(s.substring(i,i+2))<=26) ? memo[i+1]+memo[i+2] : memo[i+1];

        return memo[0];
    }
}

130	Surrounded Regions
Given a 2D board containing 'X' and 'O', capture all regions surrounded by 'X'.

A region is captured by flipping all 'O's into 'X's in that surrounded region.

For example,
X X X X
X O O X
X X O X
X O X X
After running your function, the board should be:

X X X X
X X X X
X X X X
X O X X

public class Solution {
    public void solve(char[][] board) {
        if(board==null||board.length==0||board[0].length==0) return;
        for(int i=0;i<board.length;i++) if(board[i][0]=='O') linkedUnit(board,i,0);
        for(int i=1;i<board[0].length;i++) if(board[0][i]=='O') linkedUnit(board,0,i);
        for(int i=1;i<board[0].length;i++) if(board[board.length-1][i]=='O') 
        	linkedUnit(board,board.length-1,i);
        for(int i=1;i<board.length-1;i++) if(board[i][board[0].length-1]=='O') 
        	linkedUnit(board,i,board[0].length-1);
        for(int i=0;i<board.length;i++){
            for(int j=0;j<board[0].length;j++){
                if(board[i][j]=='1') board[i][j] = 'O';
                else if(board[i][j]=='O') board[i][j] = 'X';
                else continue;
            }
        }
    }
    private void linkedUnit(char[][] board, int x, int y){
        board[x][y] = '1';
        if(x-1>0&&board[x-1][y]=='O') linkedUnit(board, x-1, y);
        if(x+1<board.length&&board[x+1][y]=='O') linkedUnit(board, x+1, y);
        if(y-1>0&&board[x][y-1]=='O') linkedUnit(board, x, y-1);
        if(y+1<board[x].length&&board[x][y+1]=='O') linkedUnit(board, x, y+1);
    }
}

29	Divide Two Integers
Divide two integers without using multiplication, division and mod operator.
If it is overflow, return MAX_INT.
public static int divide(int dividend, int divisor) {

    if(dividend==0)
        return 0;
    int signal;
    if((dividend<0&&divisor>0)||(dividend>0&&divisor<0))
        signal=-1;
    else
        signal=1;
    long absDividend=Math.abs((long)dividend);//Math.abs(最小负数) 结果还是其本身. 在进行该运算前，要将其转化为long类型。
    long absDivisor=Math.abs((long)divisor);//
    long result=0;
    while(absDividend>=absDivisor){
        long tmp=absDivisor,count=1;;
        while(tmp<=absDividend){
            tmp=tmp<<1;//这里可能溢出！！超出int表示的范围
            count=count<<1;//这里可能溢出！！超出int表示的范围
        }
        tmp=tmp>>1;
        count=count>>1;
        result+=count;
        absDividend-=tmp;
    }
      if(signal==-1)                 
        return (int)(signal*result);               
      else{
        if(result>Integer.MAX_VALUE)//溢出
           return Integer.MAX_VALUE;
        else
           return (int)result;
    }
}


151	Reverse Words in a String
Given an input string, reverse the string word by word.

For example,
Given s = "the sky is blue",
return "blue is sky the".
public class Solution {
    public String reverseWords(String s) {
        String[] parts = s.trim().split("\\s+");
        String out = "";
        if (parts.length > 0) {
            for (int i = parts.length - 1; i > 0; i--) {
                out += parts[i] + " ";
            }
            out += parts[0];
        }
        return out;
    }
}

166	Fraction to Recurring Decimal
Given two integers representing the numerator and denominator of a fraction, 
return the fraction in string format.
If the fractional part is repeating, enclose the repeating part in parentheses.

For example,
Given numerator = 1, denominator = 2, return "0.5".
Given numerator = 2, denominator = 1, return "2".
Given numerator = 2, denominator = 3, return "0.(6)"
public class Solution {
    public String fractionToDecimal(int numerator, int denominator) {
        if (denominator == 0)
            return "NaN";
        if (numerator == 0)
            return "0";
        StringBuilder result = new StringBuilder();
        Long n = new Long(numerator);
        Long d = new Long(denominator);
        // negative or positive
        if (n*d < 0)
            result.append("-");
        n = Math.abs(n);
        d = Math.abs(d);
        result.append(Long.toString(n / d));
        // result is integer or float
        if (n % d == 0)
            return result.toString();
        else
            result.append(".");
        // deal with the float part
        // key is the remainder, value is the start positon of possible repeat numbers
        HashMap<Long, Integer> map = new HashMap<Long, Integer>();
        Long r = n % d; 
        while (r > 0) {
            if (map.containsKey(r)) {
                result.insert(map.get(r), "(");
                result.append(")");
                break;
            }
            map.put(r, result.length());
            r *= 10;
            result.append(Long.toString(r / d));
            r %= d;
        }
        return result.toString();
    }
}

