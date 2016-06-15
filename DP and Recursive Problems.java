DP and Recursive Problems

String[] array = s.trim().split("\\s+");

return Integer.toBinaryString(n).replace("0","").length();   

Deque<Integer> dequeue = new LinkedList<>();
dequeue.removeFirst();
dequeue.addLast(val);

StringBuilder sb = new StringBuilder(s);
return sb.reverse().toString();

------recursion----------
339	Nested List Weight Sum
Given a nested list of integers, return the sum of all integers in the list weighted by their depth.
Each element is either an integer, or a list -- whose elements may also be integers or other lists.

Example 1:
Given the list [[1,1],2,[1,1]], return 10. (four 1's at depth 2, one 2 at depth 1)'

Example 2:
Given the list [1,[4,[6]]], return 27. (one 1 at depth 1, one 4 at depth 2, and one 6 at depth 3;
 1 + 42 + 63 = 27)

public class {
	public int depthSum(List<NestedInteger> nestedList) {
		return helper(nestedList, 1);
	}

	private int helper(List<NestedInteger) list, int depth) {
		int res = 0;
		for (NestedInteger e:list) {
			res += e.isInteger()?:e.getInteger() * depth: helper(e.getList(), depth+1);
		}
		return res;
	}
}


293	Flip Game
You are playing the following Flip Game with your friend: Given a string that contains only these two
characters: + and -, you and your friend take turns to flip twoconsecutive "++" into "--". The game
ends when a person can no longer make a move and therefore the other person will be the winner.

Write a function to compute all possible states of the string after one valid move
For example, given s = "++++", after one move, it may become one of the following states:

[
  "--++",
  "+--+",
  "++--"
]
If there is no valid move, return an empty list [].

public class Solution {
	public List<String> generatePossibleNextMoves(String s) {
		ArrayList<String> list = new ArrayList<>();
		if (s.length() <= 1) {
			return list;
		}

		char[] str = s.toCharArray();
		for (int i = 0; i < str.length-1; i++) {
			if (str[i] == '+' && str[i+1] == '+') {
				str[i] = '-';
				str[i+1] = '-';
				list.add(new String(str));
				str[i] = '+';
				str[i+1] = '+';
			}
		}

		return list;
	}
}

public class Solution {
    public List<String> generatePossibleNextMoves(String s) {
        List<String> res = new ArrayList<>();
        if(s.length() < 2) return res;
        for(int i=0; i<s.length()-1;i++){
            if(s.charAt(i) != '+' || s.charAt(i+1) != '+') continue;
            String t = s.substring(0, i) + "--" + s.substring(i+2);
            res.add(t);
        }

        return res;
    }
}

104	Maximum Depth of Binary Tree
Given a binary tree, find its maximum depth.
The maximum depth is the number of nodes along the longest path from the root node down to the
farthest leaf node.

public class Solution {
    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return Math.max(maxDepth(root.left),maxDepth(root.right)) + 1;
    }
}

226	Invert Binary Tree
Invert a binary tree.

     4
   /   \
  2     7
 / \   / \
1   3 6   9
to
     4
   /   \
  7     2
 / \   / \
9   6 3   1

public class Solution {
    public TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return root;
        }
        
        TreeNode left = invertTree(root.left);
        TreeNode right = invertTree(root.right);
        
        root.left = right;
        root.right = left;
        return root;
    }
}

100	Same Tree
Given two binary trees, write a function to check if they are equal or not.
Two binary trees are considered equal if they are structurally identical and the nodes have the same value.

public class Solution {
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null || q == null) {
            return p == null && q == null;
        }
        
        if (p.val == q.val) {
            return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
        }
        return false;
    }
}


Arrays.sort(intervals, new Comparator<Interval>() {
            public int compare(Interval a, Interval b) {
                return a.start-b.start;
            }
        });


206	Reverse Linked List
Reverse a singly linked list.
public class Solution {
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        
        ListNode newHead = new ListNode(0);
        ListNode current = head;
        while (current != null) {
            ListNode tmp = newHead.next;
            newHead.next = current;
            current = current.next;
            newHead.next.next = tmp;
        }
        
        return newHead.next;
    }
}

//pay attention to order
public class Solution {
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode p1 = head.next;
        head.next = null;
        ListNode p2 = reverseList(p1);
        p1.next = head;
        return p2;
    }
}


'***************************************'
325	Maximum Size Subarray Sum Equals k
Given an array nums and a target value k, find the maximum length of a subarray that sums to k. If there
isn't' one, return 0 instead.
Example 1:
Given nums = [1, -1, 5, -2, 3], k = 3,
return 4. (because the subarray [1, -1, 5, -2] sums to 3 and is the longest)
Example 2:
Given nums = [-2, -1, 2, 1], k = 1,
return 2. (because the subarray [-1, 2] sums to 1 and is the longest)
Follow Up:
Can you do it in O(n) time?

public int maxSubArrayLen(int[] nums, int k) {
    int sum = 0, max = 0;
    HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
    for (int i = 0; i < nums.length; i++) {
        sum = sum + nums[i];
        if (sum == k) max = i + 1;
        else if (map.containsKey(sum - k)) max = Math.max(max, i - map.get(sum - k));
        if (!map.containsKey(sum)) map.put(sum, i);
    }
    return max;
}

235	Lowest Common Ancestor of a Binary Search Tree
// BST
public class Solution {
	public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
	    if(root.val<Math.min(p.val,q.val)) return lowestCommonAncestor(root.right,p,q);
	    if(root.val>Math.max(p.val,q.val)) return lowestCommonAncestor(root.left,p,q);
	    return root;
	}
}


231	Power of Two
Given an integer, write a function to determine if it is a power of two.
public class Solution {
    public boolean isPowerOfTwo(int n) {
      if(n<=0) {
          return false;
      }
      if(Integer.toBinaryString(n).replace("0","").length() == 1) {
        return true;
      }
      return false;
    }
}

326	Power of Three
Given an integer, write a function to determine if it is a power of three.

Follow up:
Could you do it without using any loop / recursion?

public class Solution {
    public boolean isPowerOfThree(int n) {
        if (n == 1) {
            return true;
        }    
        
        while (n % 3 == 0 && n > 1) {
            n = n / 3;
        }
        if (n == 1) {
            return true;
        }
        return false;
    }
}


263	Ugly Number
Write a program to check whether a given number is an ugly number.
Ugly numbers are positive numbers whose prime factors only include 2, 3, 5. For example, 6, 8 are ugly
while 14 is not ugly since it includes another prime factor 7.
Note that 1 is typically treated as an ugly number.

public class Solution {
    public boolean isUgly(int num) {
        if(num == 0) {
            return false;
        }
        while(num % 5 == 0) {
            num /= 5;
        }
        while(num % 3 == 0) {
            num /= 3;
        }
        while(num % 2 == 0) {
            num /= 2;
        }
        if(num == 1) {
            return true;
        }
        return false;
    }
}

264 Ugly Number II
Write a program to find the n-th ugly number.
Ugly numbers are positive numbers whose prime factors only include 2, 3, 5. For example,
1, 2, 3, 4, 5, 6, 8, 9, 10, 12 is the sequence
of the first 10 ugly numbers.
Note that 1 is typically treated as an ugly number.

public class Solution {
    public int nthUglyNumber(int n) {
        if (n == 1) {
            return 1;
        }
        ArrayList<Integer> list = new ArrayList<>();
        list.add(1);
        int i, j , k;
        i = j = k = 0;
        while (list.size() != n) {
            int tmp_i = list.get(i) * 2;
            int tmp_j = list.get(j) * 3;
            int tmp_k = list.get(k) * 5;
            int min = Math.min(tmp_i,Math.min(tmp_j,tmp_k));
            if (!list.contains(min)) {
                list.add(min);
            }
            if (min == tmp_i) {
                i++;
            }else if (min == tmp_j) {
                j++;
            }else {
                k++;
            }
        }
        return list.get(n-1);
    }
}

83	Remove Duplicates from Sorted List
Given a sorted linked list, delete all duplicates such that each element appear only once.

For example,
Given 1->1->2, return 1->2.
Given 1->1->2->3->3, return 1->2->3.

public class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        
        ListNode current = head;
        while (current.next != null && current.next.next != null) {
            if (current.val == current.next.val) {
                current.next = current.next.next;
                continue;
            }
            current = current.next;
        }
        if (current.val == current.next.val) {
            current.next = null;
        }
        return head;
    }
}

public class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        if(head==null||head.next==null) return head;
        ListNode dummy=head;
        while(dummy.next!=null){
            if(dummy.next.val==dummy.val){
                dummy.next=dummy.next.next;
            }else dummy=dummy.next;
        }
        return head;
    }
}

246	Strobogrammatic Number
A strobogrammatic number is a number that looks the same when rotated 180 degrees (looked at upside down).
Write a function to determine if a number is strobogrammatic. The number is represented as a string.
For example, the numbers "69", "88", and "818" are all strobogrammatic.

public class Solution {
    public boolean isStrobogrammatic(String num) {
        HashMap<Character, Character> map = new HashMap<Character, Character>();
        map.put('0','0');
        map.put('1','1');
        map.put('6','9');
        map.put('9','6');
        map.put('8','8');
        for(int i = 0, j = num.length()-1; i <= j; i++, j--) {
            if(!map.containsKey(num.charAt(i)))
                return false;
            if(map.get(num.charAt(i)) != num.charAt(j)) 
                return false;
        }
        return true;
    }
}
// remember to check whether number

141	Linked List Cycle
Given a linked list, determine if it has a cycle in it.

Follow up:
Can you solve it without using extra space?

public class Solution {
    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) {
            return false;
        }
        
        ListNode fast = head;
        ListNode slow = head;
        
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) {
                return true;
            }
        }
        return false;
    }
}

121	Best Time to Buy and Sell Stock
Say you have an array for which the ith element is the price of a given stock on day i.
If you were only permitted to complete at most one transaction (ie, buy one and sell one share of the
stock), design an algorithm to find the maximum profit.

public class Solution {
    public int maxProfit(int[] prices) {
        if (prices.length == 0) {
            return 0;
        }
        
        int minPrice = prices[0];
        int max = 0;
        for (int i = 0; i < prices.length; i++) {
            minPrice = Math.min(minPrice, prices[i]);
            max = Math.max(max, prices[i] - minPrice);
        }
        return max;
    }
}

21	Merge Two Sorted Lists
Merge two sorted linked lists and return it as a new list. The new list should be made by splicing together
the nodes of the first two lists.

public class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null || l2 == null) {
            return l1 == null?l2:l1;
        }
        
        ListNode head = new ListNode(0);
        ListNode current = head;
        ListNode p1 = l1;
        ListNode p2 = l2;
        
        while (p1 != null && p2 != null) {
            if (p1.val <= p2.val) {
                current.next = p1;
                p1 = p1.next;
            }else {
                current.next = p2;
                p2 = p2.next;
            }
            current = current.next;
        }
        
        if (p1 != null) {
            current.next = p1;
        }else {
            current.next = p2;
        }
        return head.next;
    }
}

24	Swap Nodes in Pairs
Given a linked list, swap every two adjacent nodes and return its head.

For example,
Given 1->2->3->4, you should return the list as 2->1->4->3.

Your algorithm should use only constant space. You may not modify the values in the list, only nodes
itself can be changed.

public class Solution {
    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        
        ListNode p = head.next.next;
        ListNode tmp = head.next;
        tmp.next = head;
        head.next = null;
        if (p != null) {
            ListNode p2 = swapPairs(p);
            head.next = p2;
        }else {
            head.next = null;
        }
        return tmp;
    }
}

345	Reverse Vowels of a String
Write a function that takes a string as input and reverse only the vowels of a string.

Example 1:
Given s = "hello", return "holle".
Example 2:
Given s = "leetcode", return "leotcede".

public class Solution {
    public String reverseVowels(String s) {
        StringBuilder sb = new StringBuilder();
        int j = s.length() - 1;
        for (int i = 0; i < s.length(); i++)
        {
            if ("AEIOUaeiou".indexOf(s.charAt(i)) != -1)
            {
                while (j >= 0 && "AEIOUaeiou".indexOf(s.charAt(j)) == -1)
                {
                    j--;
                }
                sb.append(s.charAt(j));
                j--;
            }
            else
                sb.append(s.charAt(i));
        }
        return sb.toString();
    }
}

198	House Robber
You are a professional robber planning to rob houses along a street. Each house has a certain amount of
money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have
security system connected and it will automatically contact the police if two adjacent houses were broken
into on the same night.

Given a list of non-negative integers representing the amount of money of each house, determine the maximum
amount of money you can rob tonight without alerting the police.

public class Solution {
    public int rob(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        if (nums.length == 1) {
            return nums[0];
        }
        int[] sum = new int[nums.length];
        sum[0] = nums[0];
        sum[1] = Math.max(nums[0], nums[1]);
        for (int i = 2; i < nums.length; i++) {
            sum[i] = Math.max(sum[i-1], sum[i-2] + nums[i]);
        }
        return sum[nums.length-1];
    }
}

270	Closest Binary Search Tree Value
Given a non-empty binary search tree and a target value, find the value in the BST that is closest to the
target.
Note: Given target value is a floating point. You are guaranteed to have only one unique value in the BST
that is closest to the target.

public class Solution {
    public int closestValue(TreeNode root, double target) {
        int closest = root.val;
        while(root != null){
            // 如果该节点的离目标更近，则更新到目前为止的最近值
            closest = Math.abs(closest - target) < Math.abs(root.val - target) ? closest : root.val;
            // 二叉搜索
            root = target < root.val ? root.left : root.right;
        }
        return closest;
    }
}

110	Balanced Binary Tree
Given a binary tree, determine if it is height-balanced.
For this problem, a height-balanced binary tree is defined as a binary tree in which the depth of the two
subtrees of every node never differ by more than 1.

public class Solution {
    public boolean isBalanced(TreeNode root) {
      if (root == null) {
        return true;
      }
      if(root.left == null && root.right == null) {
        return true;
      }
      if(isBalanced(root.left) && isBalanced(root.right) && Math.abs(height(root.left)-height(root.right)) <= 1) {
          return true;
      }
      return false;
    }
    
    public int height(TreeNode root) {
        if(root == null) {
            return 0;
        }
        if (root.left == null && root.right == null) {
            return 1;
        }
        return Math.max(height(root.left), height(root.right)) + 1;
    }
}

// why remove height if (root.left == null && root.right == null) wrong?

'***************************'
101	Symmetric Tree
Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).
For example, this binary tree is symmetric:

    1
   / \
  2   2
 / \ / \
3  4 4  3
But the following is not:
    1
   / \
  2   2
   \   \
   3    3
Note:
Bonus points if you could solve it both recursively and iteratively.

public class Solution {
    public boolean isSymmetric(TreeNode root) {
        if (root == null || root.left == null && root.right == null) {
            return true;
        }
        
        if (root.left != null && root.right != null) {
            return mirror(root.left, root.right);
        }
        
        return false;
    }
    
    public boolean mirror(TreeNode left, TreeNode right) {
        if (left == null && right == null) {
            return true;
        }
        
        if (left != null && right != null && left.val == right.val) {
            return mirror(left.left,right.right) && mirror(left.right, right.left);
        }

        return false;        
    }
}

107	Binary Tree Level Order Traversal II
Given a binary tree, return the bottom-up level order traversal of its nodes' values.' 
(ie, from left to right, level by level from leaf to root).

For example:
Given binary tree {3,9,20,#,#,15,7},
    3
   / \
  9  20
    /  \
   15   7
return its bottom-up level order traversal as:
[
  [15,7],
  [9,20],
  [3]
]

public class Solution {
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        ArrayList<List<Integer>> lists = new ArrayList<List<Integer>>();
        if (root != null) {
            Queue<TreeNode> sameLevel = new LinkedList<>();
            sameLevel.add(root);
            
            while (!sameLevel.isEmpty()) {
                Queue<TreeNode> tmp = new LinkedList<>();
                ArrayList<Integer> list = new ArrayList<>();
                while (!sameLevel.isEmpty()) {
                    TreeNode node = sameLevel.remove();
                    list.add(node.val);
                    if (node.left != null) {
                        tmp.add(node.left);
                    }
                    if (node.right != null) {
                        tmp.add(node.right);
                    }
                    
                }
                lists.add(0,list);
                sameLevel = tmp;
            }
        }
        return lists;
    }
}


66	Plus One
Given a non-negative number represented as an array of digits, plus one to the number.
The digits are stored such that the most significant digit is at the head of the list.

public class Solution {
    public int[] plusOne(int[] digits) {
        if (digits.length == 0) {
            int[] a = new int[1];
            a[0] = 1;
            return a;
        }
        
        for (int i = digits.length - 1; i >= 0; i--) {
            if (digits[i]+1 <= 9) {
                digits[i] += 1;
                return digits;
            }else{
                digits[i] = 0;
                if(i-1>=0) {
                    continue;
                }else {
                    int[] ans = new int[digits.length + 1];
                    ans[0] = 1;
                    return ans;
                }
            }
        }
        return null;
    }
}

342	Power of Four
Given an integer (signed 32 bits), write a function to check whether it is a power of 4.
Example:
Given num = 16, return true. Given num = 5, return false.
Follow up: Could you solve it without loops/recursion?

public class Solution {
    public boolean isPowerOfFour(int num) {
        if (num <= 0) {
            return false;
        }
        
        while (num > 0 && num % 4 == 0) {
            num = num / 4;
        }
        if (num == 1) {
            return true;
        }
        return false;
    }
}


172	Factorial Trailing Zeroes
Given an integer n, return the number of trailing zeroes in n!.
Note: Your solution should be in logarithmic time complexity.
Because from 1 to n, the number of 2 factors is always bigger than the number of 5 factors. 
So we only need to find the number of 5 factors among 1...n.

1st loop: 5, 10, 15, 20, 25, 30, ....
2nd loop: 25 50 ...... .....

public class Solution {
    public int trailingZeroes(int n) {
        return n == 0 ? 0 : n / 5 + trailingZeroes(n / 5);
    }
}

We can easily observe that the number of 2s in prime factors is always more than or equal to the number 
of 5s. So if we count 5s in prime factors, we are done. How to count total number of 5s in prime factors
of n!? A simple way is to calculate floor(n/5). For example, 7! has one 5, 10! has two 5s. It is done yet,
there is one more thing to consider. Numbers like 25, 125, etc have more than one 5. For example if we 
consider 28!, we get one extra 5 and number of 0s become 6. Handling this is simple, first divide n by 5 
and remove all single 5s, then divide by 25 to remove extra 5s and so on. Following is the summarized 
formula for counting trailing 0s.

Trailing 0s in n! = Count of 5s in prime factors of n!
                  = floor(n/5) + floor(n/25) + floor(n/125) + ....
int findTrailingZeros(int  n)
{
    // Initialize result
    int count = 0;
 
    // Keep dividing n by powers of 5 and update count
    for (int i=5; n/i>=1; i *= 5)
          count += n/i;
 
    return count;
}

102	Binary Tree Level Order Traversal
Given a binary tree, return the level order traversal of its nodes' values.'
 (ie, from left to right, level by level).

For example:
Given binary tree {3,9,20,#,#,15,7},
    3
   / \
  9  20
    /  \
   15   7
return its level order traversal as:
[
  [3],
  [9,20],
  [15,7]
]

public class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        ArrayList<List<Integer>> lists = new ArrayList<List<Integer>>();
        
        if (root == null) {
            return lists;
        }
        Queue<TreeNode> curLevel = new LinkedList<TreeNode>();
        curLevel.add(root);
        
        while (!curLevel.isEmpty()) {
            ArrayList<Integer> list = new ArrayList<Integer>();
            Queue<TreeNode> tmp = new LinkedList<TreeNode>();
            
            while (!curLevel.isEmpty()) {
                TreeNode t = curLevel.remove();
                if (t.left != null) {
                    tmp.add(t.left);
                }
                if (t.right != null) {
                    tmp.add(t.right);
                }
                list.add(t.val);
            }
            curLevel = tmp;
            lists.add(list);
        }
        
        return lists;
    }
}

112	Path Sum
Given a binary tree and a sum, determine if the tree has a root-to-leaf path such that adding up all the
values along the path equals the given sum.

For example:
Given the below binary tree and sum = 22,
              5
             / \
            4   8
           /   / \
          11  13  4
         /  \      \
        7    2      1
return true, as there exist a root-to-leaf path 5->4->11->2 which sum is 22.

public class Solution {
    public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null) {
            return false;
        }
        if (root.left == null && root.right == null && sum == root.val) {
            return true;
        }
        sum = sum - root.val;
        return hasPathSum(root.right, sum) || hasPathSum(root.left, sum);
    }
}
// do not add root.val > sum, for case [-2,null,-3], -5 
// Remember java use root.left  root.right. root.value!!!!!!

'*****************************'
276	Paint Fence
There is a fence with n posts, each post can be painted with one of the k colors.
You have to paint all the posts such that no more than two adjacent fence posts have the same color. 
Return the total number of ways you can paint the fence. 

We know for each post, it could differ or same as its previous post's color.
Assume: 
differ_count: represents the current post with different color with its previous post(the painting ways)
same_count: represents the current post share the same color with its previous post(the painiting ways)

We could have following trasitinao function
differ_count(i) = differ_count(i-1) * (k-1) + same_count(i-1) * (k-1)
same_count(i) = differ_count(i-1) //cause the current post must have the same color with post i-1, thus we could only use the way that differ_count(i-1)

Base case:
2 is a perfect base case for use to start, since it has simple same_count and differ_count;
'
public class Solution {
    public int numWays(int n, int k) {
        if (n == 0 || k == 0)
            return 0;
        if (n == 1)
            return k;
        int same_count = k;
        int differ_count = k * (k - 1);
        for (int i = 3; i <= n; i++) {
            int temp = differ_count;
            differ_count = differ_count * (k - 1) + same_count * (k - 1);
            same_count = temp;
        }
        return same_count + differ_count;
    }
}


111	Minimum Depth of Binary Tree
Given a binary tree, find its minimum depth.
The minimum depth is the number of nodes along the shortest path from the root node down to the nearest
leaf node.

public class Solution {
    public int minDepth(TreeNode root) {
        if (root == null)
            return 0;
        if (root.left != null && root.right != null)
            return Math.min(minDepth(root.left), minDepth(root.right))+1;
        else
            return Math.max(minDepth(root.left), minDepth(root.right))+1;
    }
}

160	Intersection of Two Linked Lists
Write a program to find the node at which the intersection of two singly linked lists begins.
For example, the following two linked lists:

A:          a1 → a2
                   ↘
                     c1 → c2 → c3
                   ↗            
B:     b1 → b2 → b3
begin to intersect at node c1

public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) {
            return null;
        }
        
        ListNode p1 = headA;
        ListNode p2 = headB;
        while (p1 != p2) {
            p1 = p1 == null? headB: p1.next;
            p2 = p2 == null? headA: p2.next;
        }
        return p1;
    }
}

219	Contains Duplicate II
Given an array of integers and an integer k, find out whether there are two distinct indices i and j in the
array such that nums[i] = nums[j] and the difference between i and j is at most k.

public class Solution {
    public boolean containsNearbyDuplicate(int[] nums, int k) {
        if (nums.length == 0 || k == 0) {
            return false;
        }
        Map<Integer, Integer> map = new HashMap<>();
        
        for (int i = 0; i < nums.length; i++) {
            if (map.get(nums[i]) == null) {
                map.put(nums[i], i);
            }else {
                if (i-map.get(nums[i]) <= k) {
                    return true;
                }else {
                    map.put(nums[i],i);
                }
            }
        }
        return false;
    }
}

public boolean containsNearbyDuplicate(int[] nums, int k) {
    HashSet<Integer> hs=new HashSet<>();
    for(int i=0;i<nums.length;i++)
    {
        if(hs.add(nums[i])==false) return true;
        if(hs.size()==k+1) hs.remove(nums[i-k]);
    }
    return false;
}


'***************************'
205	Isomorphic Strings
Given two strings s and t, determine if they are isomorphic.
Two strings are isomorphic if the characters in s can be replaced to get t.
All occurrences of a character must be replaced with another character while preserving the order of
characters. No two characters may map to the same character but a character may map to itself.

For example,
Given "egg", "add", return true.
Given "foo", "bar", return false.
Given "paper", "title", return true.

Note:
You may assume both s and t have the same length.

public class Solution {
    public boolean isIsomorphic(String s, String t) {
        if (s.length() == 0 && t.length() == 0) {
            return true;
        }
        
        Map<Character, Character> map = new HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            char cs = s.charAt(i);
            char ct = t.charAt(i);
            
            if (map.get(cs) == null) {
                if (map.values().contains(ct)) {
                    return false;
                }
                map.put(cs, ct);
            }else {
                if (map.get(cs) != ct) {
                    return false;
                }
            }
        }
        return true;
    }
}
// pay attention to "abb" "aaa"

public class Solution {
    public boolean isIsomorphic(String s1, String s2) {
        int[] m = new int[512];
        for (int i = 0; i < s1.length(); i++) {
            if (m[s1.charAt(i)] != m[s2.charAt(i)+256]) return false;
            m[s1.charAt(i)] = m[s2.charAt(i)+256] = i+1;
        }
        return true;
    }
}

'***************************'
19	Remove Nth Node From End of List
Given a linked list, remove the nth node from the end of list and return its head.
For example,

   Given linked list: 1->2->3->4->5, and n = 2.

   After removing the second node from the end, the linked list becomes 1->2->3->5.
Note:
Given n will always be valid.
Try to do this in one pass

public ListNode removeNthFromEnd(ListNode head, int n) {
    ListNode slow = head;
    ListNode fast = head;
    for (int i = 0; i < n; i++) { // pay attention to length
        fast = fast.next;
    }
    if (fast == null) {
        return slow.next;
    }
    while (fast.next != null) {
        slow = slow.next;
        fast = fast.next;
    }
    slow.next = slow.next.next;
    return head;
}

20	Valid Parentheses
Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input
string is valid.

The brackets must close in the correct order, "()" and "()[]{}" are all valid but "(]" and "([)]"
are not.

public class Solution {
  public boolean isValid(String s) {
    while (s.contains("()") || s.contains("[]") || s.contains("{}")) {
      s = s.replace("()", "");
      s = s.replace("[]", "");
      s = s.replace("{}", "");
    }   
    if (s.length() == 0) 
      return true;
    else 
      return false;
  }
}

"***************************"
38	Count and Say
The count-and-say sequence is the sequence of integers beginning as follows:
1, 11, 21, 1211, 111221, ...

1 is read off as "one 1" or 11.
11 is read off as "two 1s" or 21.
21 is read off as "one 2, then one 1" or 1211.
Given an integer n, generate the nth sequence.

Note: The sequence of integers will be represented as a string.

public class Solution {
    public String countAndSay(int n) {
        String s = "1";
        for (int i = 1; i < n; i++) {
            StringBuilder sb = new StringBuilder();
            for (int j = 1, count = 1; j <= s.length(); j++) {
                if (j == s.length() || s.charAt(j - 1) != s.charAt(j)) {
                    sb.append(count);
                    sb.append(s.charAt(j - 1));
                    count = 1;
                } else count++;
            }
            s = sb.toString();
        }
        return s;
    }
}

"**********************************"
157	Read N Characters Given Read4
The API: int read4(char *buf) reads 4 characters at a time from a file.
The return value is the actual number of characters read. For example, it returns 3 if there is only
3 characters left in the file.
By using the read4 API, implement the function int read(char *buf, int n) that reads n characters from
the file.
public class Solution extends Reader4 {
    /**
     * @param buf Destination buffer
     * @param n   Maximum number of characters to read
     * @return    The number of characters read
     */
    public int read(char[] buf, int n) {

        char[] buffer = new char[4];
        boolean endOfFile = false;
        int readBytes = 0;

        while (readBytes < n && !endOfFile) {
            int currReadBytes = read4(buffer);
            if (currReadBytes !=4) {
                endOfFile = true;
            }
            int length = Math.min(n - readBytes, currReadBytes);
            for (int i=0; i<length; i++) {
                buf[readBytes + i] = buffer[i];
            }
            readBytes += length;
        }
        return readBytes;
    }
}


"***********************"
190	Reverse Bits
Reverse bits of a given 32 bits unsigned integer.
For example, given input 43261596 (represented in binary as 00000010100101000001111010011100),
return 964176192 (represented in binary as 00111001011110000010100101000000).

Follow up:
If this function is called many times, how would you optimize it?

    public int reverseBits(int n) {
        int result = 0;
        for (int i = 0; i < 32; i++) {
            result += n & 1;
            n >>>= 1;
            if (i < 31) {
                result <<= 1;
            }
        }
        return result;
    }

public class Solution {
    // you need treat n as an unsigned value
    public int reverseBits(int n) {
      StringBuffer sb = new StringBuffer(Integer.toBinaryString(n));
      return Integer.parseInt(sb.reverse(), 2);
    }
}

(this solution is not working because most of the time the length of the string will less than 32 bit)

How to optimize if this function is called multiple times? We can divide an int into 4 bytes, and reverse
each byte then combine into an int. For each byte, we can use cache to improve performance.

// cache
private final Map<Byte, Integer> cache = new HashMap<Byte, Integer>();
public int reverseBits(int n) {
    byte[] bytes = new byte[4];
    for (int i = 0; i < 4; i++) // convert int into 4 bytes
        bytes[i] = (byte)((n >>> 8*i) & 0xFF);
    int result = 0;
    for (int i = 0; i < 4; i++) {
        result += reverseByte(bytes[i]); // reverse per byte
        if (i < 3)
            result <<= 8;
    }
    return result;
}

private int reverseByte(byte b) {
    Integer value = cache.get(b); // first look up from cache
    if (value != null)
        return value;
    value = 0;
    // reverse by bit
    for (int i = 0; i < 8; i++) {
        value += ((b >>> i) & 1);
        if (i < 7)
            value <<= 1;
    }
    cache.put(b, value);
    return value;
}

"***********************"
257	Binary Tree Paths
Given a binary tree, return all root-to-leaf paths.
For example, given the following binary tree:

   1
 /   \
2     3
 \
  5
All root-to-leaf paths are:

["1->2->5", "1->3"]

public class Solution {
    public List<String> binaryTreePaths(TreeNode root) {
        ArrayList<String> ans = new ArrayList<>();
        
        if (root == null) {
            return ans;
        }
        helper(ans, root, "");
        return ans;
    }
    
    public void helper(List<String> list, TreeNode root, String s) {
        String str = s;
        if (root == null) {
            return;    
        }
        if (root.left == null && root.right == null) {
            str = str + "->" + root.val;
            list.add(str.substring(2));
            return;
        }
        helper(list, root.left, s + "->" + root.val);
        helper(list, root.right, s + "->" + root.val);
    }
}

public class Solution {
    public List<String> binaryTreePaths(TreeNode root) {
      List<String> paths = new ArrayList<>();
      if(root != null) {
        allPaths(root, "", paths);
      }
      return paths;
    }
    
    public void allPaths(TreeNode root, String str, List<String> paths){
      if (root.left == null && root.right == null) {
        paths.add(str + root.val);
      }
      if (root.left != null) allPaths(root.left, str + root.val + "->", paths);
      if (root.right != null) allPaths(root.right, str + root.val + "->", paths);
    }
}



"**************************"
234	Palindrome Linked List
Given a singly linked list, determine if it is a palindrome.

public class Solution 
{
    public boolean isPalindrome(ListNode head) 
    {
        Stack<Integer> s = new Stack<Integer>();
        ListNode temp = head;
        ListNode cur = head;
        while(temp != null)
        {
            s.push(temp.val);
            temp = temp.next;
        }
        while(cur != null)
        {
            if(cur.val != s.peek())
                return false;
            else
            {
                cur = cur.next;
                s.pop();
            }
        }
        return true;
    }
}


28	Implement strStr()
Implement strStr().
Returns the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.

public class Solution {
    public int strStr(String haystack, String needle) {
        return haystack.indexOf(needle);    
    }
}

Here is a pretty concise implementation of a Knuth-Morris-Pratt algorithm in Java. Instead of commenting
and explaining the approach I want to give a really-really useful link to TopCoder tutorial on the topic.
The code is just a slightly modified version of the code from the tutorial and an explanation there is
perfect.

public class Solution {

    private int[] failureFunction(char[] str) {
        int[] f = new int[str.length+1];
        for (int i = 2; i < f.length; i++) {
            int j = f[i-1];
            while (j > 0 && str[j] != str[i-1]) j = f[j];
            if (j > 0 || str[j] == str[i-1]) f[i] = j+1;
        }
        return f;
    }

    public int strStr(String haystack, String needle) {
        if (needle.length() == 0) return 0;
        if (needle.length() <= haystack.length()) {
            int[] f = failureFunction(needle.toCharArray());
            int i = 0, j = 0;
            while (i < haystack.length()) {
                if (haystack.charAt(i) == needle.charAt(j)) {
                    i++; j++;
                    if (j == needle.length()) return i-j;
                } else if (j > 0) j = f[j];
                else i++;
            }
        }
        return -1;
    }
}

"*******************"
204	Count Primes
Description:
Count the number of prime numbers less than a non-negative number, n.

2 , 3, 5, 7 , 11, 13,
public class Solution {
    public int countPrimes(int n) {
        boolean[] isPrimes = new boolean[n];
        int count = 0;
        for (int i = 2; i < n; i++) {
            if (isPrimes[i] == false) {
                count++;
                for (int j = 2; i*j < n; j++) {
                    isPrimes[i*j] = true;
                }
            }
        }
        return count;
    }
}

6	ZigZag Conversion
The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you
may want to display this pattern in a fixed font for better legibility)

P   A   H   N
A P L S I I G
Y   I   R
And then read line by line: "PAHNAPLSIIGYIR"
Write the code that will take a string and make this conversion given a number of rows:

string convert(string text, int nRows);
convert("PAYPALISHIRING", 3) should return "PAHNAPLSIIGYIR".

public class Solution {
    public String convert(String s, int numRows) {
        if (s.length() == 0 || numRows == 1) {
            return s;
        }
        
        StringBuilder[] sbs = new StringBuilder[numRows];
        for (int i = 0; i < sbs.length; i++) sbs[i] = new StringBuilder();
        int index = 0;
        while (index < s.length()) {
            for (int i = 0; i < numRows && index < s.length(); i++) {
                sbs[i].append(s.charAt(index++));
            }
            
            for (int i = numRows - 2; i > 0 && index < s.length(); i--) {
                sbs[i].append(s.charAt(index++));
            }
        }
        
        String ans = "";
        for (int i = 0; i < numRows; i++) {
            ans += sbs[i].toString();
        }
        
        return ans;
    }
}

"**********************"
7	Reverse Integer
Reverse digits of an integer.
Example1: x = 123, return 321
Example2: x = -123, return -321

public class Solution {
    public int reverse(int x)
    {
        int result = 0;
    
        while (x != 0)
        {
            int tail = x % 10;
            int newResult = result * 10 + tail;
            if ((newResult - tail) / 10 != result)
            { return 0; }
            result = newResult;
            x = x / 10;
        }
        
        return result;
    }
}

189	Rotate Array
Rotate an array of n elements to the right by k steps.
For example, with n = 7 and k = 3, the array [1,2,3,4,5,6,7] is rotated to [5,6,7,1,2,3,4].
public class Solution {
    public void rotate(int[] nums, int k) {
        if (nums.length == 0) {
            return;
        }
        
        int len = nums.length;
        if (k > len) {
            k = k % len;
        }
        
        helper(nums, 0, len-k-1);  // x-1, next is x
        helper(nums, len-k, len-1);
        helper(nums, 0, len-1);
    }
    
    public void helper(int[] nums, int i, int j) {
        if (nums.length == 0) {
            return;
        }
        int lo = i;
        int hi = j;
        
        while (lo < hi) {
            int t = nums[hi];
            nums[hi] = nums[lo];
            nums[lo] = t;
            lo++;
            hi--;
        }
        return;
    }
}


165	Compare Version Numbers
Compare two version numbers version1 and version2.
If version1 > version2 return 1, if version1 < version2 return -1, otherwise return 0.

You may assume that the version strings are non-empty and contain only digits and the . character.
The . character does not represent a decimal point and is used to separate number sequences.
For instance, 2.5 is not "two and a half" or "half way to version three", it is the fifth second-level
revision of the second first-level revision.

Here is an example of version numbers ordering:
0.1 < 1.1 < 1.2 < 13.37

public class Solution {
public int compareVersion(String version1, String version2) {
    if (version1 == null || version2 == null) return 0;
    String[] vr1 = version1.split("\\.");
    String[] vr2 = version2.split("\\.");
    int l1 = vr1.length;
    int l2 = vr2.length;
    int len = l1 >= l2 ? l1 : l2;
    int v1, v2;
    for (int i = 0; i < len; i++) {
        v1 = (i >= l1 ? 0 : Integer.parseInt(vr1[i]));
        v2 = (i >= l2 ? 0 : Integer.parseInt(vr2[i]));
        if (v1 > v2) return 1;
        else if (v1 < v2) return -1;
    }
    return 0;
}

288	Unique Word Abbreviation
An abbreviation of a word follows the form <first letter><number><last letter>. Below are some examples of
word abbreviations:
a) it                      --> it    (no abbreviation)

     1
b) d|o|g                   --> d1g

              1    1  1
     1---5----0----5--8
c) i|nternationalizatio|n  --> i18n

              1
     1---5----0
d) l|ocalizatio|n          --> l10n
Assume you have a dictionary and given a word, find whether its abbreviation is unique in the dictionary. 
A word's abbreviation is unique if no other word from the dictionary has the same abbreviation.'
Example: 
Given dictionary = [ "deer", "door", "cake", "card" ]

isUnique("dear") -> false
isUnique("cart") -> true
isUnique("cane") -> false
isUnique("make") -> true

public class ValidWordAbbr {   
    HashMap<String,String> map;
    public ValidWordAbbr(String[] dictionary) {
        map = new HashMap<String,String>();
        for(String str:dictionary){
            String key = getKey(str);
            if(map.containsKey(key) && !map.get(key).equals(str))
                map.put(key,"");
            else
                map.put(key,str);
        }
    }

    public boolean isUnique(String word) {
        String key = getKey(word);
        return !map.containsKey(key)||map.get(key).equals(word);
    }
    
    private String getKey(String str){
        if(str.length()<=2) return str;
        return str.charAt(0)+Integer.toString(str.length()-2)+str.charAt(str.length()-1);
    }
}


8	String to Integer (atoi)
Implement atoi to convert a string to an integer.
Hint: Carefully consider all possible input cases. If you want a challenge, please do not see below and
ask yourself what are the possible input cases.
Notes: It is intended for this problem to be specified vaguely (ie, no given input specs). You are
responsible to gather all the input requirements up front.

public class Solution {
    public int myAtoi(String str) {
        if (str.isEmpty())
            return 0;
        str = str.trim();
        int i = 0, ans = 0, sign = 1, len = str.length();
        if (str.charAt(i) == '-' || str.charAt(i) == '+')
            sign = str.charAt(i++) == '+' ? 1 : -1;
        for (; i < len; ++i) {
            int tmp = str.charAt(i) - '0';
            if (tmp < 0 || tmp > 9)
                break;
            if (ans > Integer.MAX_VALUE / 10
                    || (ans == Integer.MAX_VALUE / 10 && Integer.MAX_VALUE % 10 < tmp))
                return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            else
                ans = ans * 10 + tmp;
        }
        return sign * ans;
    }
}
