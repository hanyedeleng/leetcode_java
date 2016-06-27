Dec  = Decimal Value
Char = Character

'5' has the int value 53
if we write '5'-'0' it evaluates to 53-48, or the int 5
if we write char c = 'B'+32; then c stores 'b'


Dec  Char                           Dec  Char     Dec  Char     Dec  Char
---------                           ---------     ---------     ----------
  0  NUL (null)                      32  SPACE     64  @         96  `
  1  SOH (start of heading)          33  !         65  A         97  a
  2  STX (start of text)             34  "         66  B         98  b
  3  ETX (end of text)               35  #         67  C         99  c
  4  EOT (end of transmission)       36  $         68  D        100  d
  5  ENQ (enquiry)                   37  %         69  E        101  e
  6  ACK (acknowledge)               38  &         70  F        102  f
  7  BEL (bell)                      39  '         71  G        103  g
  8  BS  (backspace)                 40  (         72  H        104  h
  9  TAB (horizontal tab)            41  )         73  I        105  i
 10  LF  (NL line feed, new line)    42  *         74  J        106  j
 11  VT  (vertical tab)              43  +         75  K        107  k
 12  FF  (NP form feed, new page)    44  ,         76  L        108  l
 13  CR  (carriage return)           45  -         77  M        109  m
 14  SO  (shift out)                 46  .         78  N        110  n
 15  SI  (shift in)                  47  /         79  O        111  o
 16  DLE (data link escape)          48  0         80  P        112  p
 17  DC1 (device control 1)          49  1         81  Q        113  q
 18  DC2 (device control 2)          50  2         82  R        114  r
 19  DC3 (device control 3)          51  3         83  S        115  s
 20  DC4 (device control 4)          52  4         84  T        116  t
 21  NAK (negative acknowledge)      53  5         85  U        117  u
 22  SYN (synchronous idle)          54  6         86  V        118  v
 23  ETB (end of trans. block)       55  7         87  W        119  w
 24  CAN (cancel)                    56  8         88  X        120  x
 25  EM  (end of medium)             57  9         89  Y        121  y
 26  SUB (substitute)                58  :         90  Z        122  z
 27  ESC (escape)                    59  ;         91  [        123  {
 28  FS  (file separator)            60  <         92  \        124  |
 29  GS  (group separator)           61  =         93  ]        125  }
 30  RS  (record separator)          62  >         94  ^        126  ~
 31  US  (unit separator)            63  ?         95  _        127  DEL
 ------------------------------------------------------------------------"
346	Moving Average from Data Stream
Given a stream of integers and a window size, calculate the moving average of all integers in the
sliding window.

For example,
MovingAverage m = new MovingAverage(3);
m.next(1) = 1
m.next(10) = (1 + 10) / 2
m.next(3) = (1 + 10 + 3) / 3
m.next(5) = (10 + 3 + 5) / 3

public class MovingAverage {
  private Deque<Integer> dequeue = new LinkedList<>();
  private int size;
  private long sum;

  public MovingAverage(int size) {
  	this.size = size;
  }

  public double next(int val) {
    if (dequeue.size() == size) {
      sum -= dequeue.removeFirst();    	
    }

    dequeue.addLast(val);
    sum+=val;
    return (double)sum/dequeue.size();
  }
}

//.add() .addFirst() .addLast() .getFirst() .getLast() .remove() .removeFirst(). .removeLast();
add(element): Adds an element to the tail.
addFirst(element): Adds an element to the head.
addLast(element): Adds an element to the tail.
offer(element): Adds an element to the tail and returns a boolean to explain if the insertion was successful.
offerFirst(element): Adds an element to the head and returns a boolean to explain if the insertion was successful.
offerLast(element): Adds an element to the tail and returns a boolean to explain if the insertion was successful.
iterator(): Returna an iterator for this deque.
descendingIterator(): Returns an iterator that has the reverse order for this deque.
push(element): Adds an element to the head.
pop(element): Removes an element from the head and returns it.
removeFirst(): Removes the element at the head.
removeLast(): Removes the element at the tail.


344	Reverse String
Write a function that takes a string as input and returns the string reversed.

Example:
Given s = "hello", return "olleh".

public class Solution {
	public String reverseString(String s) {
	    if (s.length() == 0) {
	        return s;
	    }
	    
	    StringBuilder sb = new StringBuilder(s);
	    return sb.reverse().toString();
	}
}

//method 2: use swap method
public String reverseString(String s){
    if(s == null || s.length() == 0)
        return "";
    char[] cs = s.toCharArray();
    int begin = 0, end = s.length() - 1;
    while(begin <= end){
        char c = cs[begin];
        cs[begin] = cs[end];
        cs[end] = c;
        begin++;
        end--;
    }

    return new String(cs);
}

// StringBuilder   .append() .delete(int start, int end) .indexOf()--return type int 
// .insert() .lastIndexOf() .replace() .reverse() .toString();


292	Nim Game
266	Palindrome Permutation
Given a string, determine if a permutation of the string could form a palindrome.
For example,
"code" -> False, "aab" -> True, "carerac" -> True.

public class Solution {
	public boolean canPermutePalindrome(String s) {
		if (s.length() == 0) {
			return true;
		}

		Set<Character> set = new HashSet<>();
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (!set.add(c)) {
                set.remove(c);
            }
		}
		return set.size() <= 1;
	}
}

// .add(), .clear(), .contains(Object o), containsAll(Collection<?> c), isEmpty(), .remove()
// .removeAll(Collection<?> c), size(), toArray()

258	Add Digits
Given a non-negative integer num, repeatedly add all its digits until the result has only one digit.

For example:
Given num = 38, the process is like: 3 + 8 = 11, 1 + 1 = 2. Since 2 has only one digit, return it.
Follow up:
Could you do it without any loop/recursion in O(1) runtime?

public class Solution {
    public int addDigits(int num) {
        if (num == 0) {
            return 0;
        }
        if (num%9 == 0) {
            return 9;
        }
        return num%9;
    }
}
// remember case 0, 9 , 18 ...


243	Shortest Word Distance
Given a list of words and two words word1 and word2, return the shortest distance between these two
words in the list.
For example,
Assume that words = ["practice", "makes", "perfect", "coding", "makes"].
Given word1 = “coding”, word2 = “practice”, return 3.
Given word1 = "makes", word2 = "coding", return 1.
Note:
You may assume that word1 does not equal to word2, and word1 and word2 are both in the list.

public class Solution {
	public shortestWordDistance(String w1, String w1, String[] words) {
		int index_i = -1, index_j = -1;
		int min = 1;
		for (int i = 0; i < words.length; i++) {
			if (words[i] == w1) {
				index_i = i;
			}else if (words[i] == w2) {
				index_j = i;
			}

			if (index_i > 0 && index_j > 0) {
				min = Math.min(Math.abs(i-j), min);
			}
		}
		return min;	
	}
}



283	Move Zeroes
Given an array nums, write a function to move all 0's' to the end of it while maintaining the relative
order of the non-zero elements.

For example, given nums = [0, 1, 0, 3, 12], after calling your function, nums should be [1, 3, 12, 0, 0].
Note:
You must do this in-place without making a copy of the array.
Minimize the total number of operations.

public class Solution {
    public void moveZeroes(int[] nums) {
        if (nums.length <= 1) {
            return;
        }
        
        int nums_zero = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == 0) {
                nums_zero++;
            }else {
                if (nums_zero > 0) {
                    nums[i-nums_zero] = nums[i];
                    nums[i] = 0;
                }
            }    
        }
        return;
    }
}


349	Intersection of Two Arrays
Given two arrays, write a function to compute their intersection.

Example:
Given nums1 = [1, 2, 2, 1], nums2 = [2, 2], return [2].

Note:
Each element in the result must be unique.
The result can be in any order.

public class Solution {
    public int[] intersection(int[] nums1, int[] nums2) {
        Set<Integer> s = new HashSet<>();
        if (nums1.length == 0 || nums2.length == 0) {
            return nums1.length == 0?nums1:nums2;
        }
        
        Set<Integer> t = new HashSet<>();
        for (int i = 0; i < nums1.length; i++) {
            t.add(nums1[i]);
        }

        for (int i = 0; i < nums2.length; i++) {
            if (t.contains(nums2[i])) {
                s.add(nums2[i]);
            }
        }
        
        int[] ans = new int[s.size()];

        int index = 0;
        
        for( Integer i : s ) {
          ans[index++] = i;
        }
        
        return ans;
    }
}
// use contains instead of .add() 
[4,7,9,7,6,7]
[5,0,0,6,1,6,2,2,4]
//

237	Delete Node in a Linked List
Write a function to delete a node (except the tail) in a singly linked list, given only access to that node.

Supposed the linked list is 1 -> 2 -> 3 -> 4 and you are given the third node with value 3, the linked list
should become 1 -> 2 -> 4 after calling your function.

[0,1]
node at index 0 (node.val = 0)

public class Solution {
    public void deleteNode(ListNode node) {
        // this means delete the current node.
        if (node != null && node.next != null) {
            node.val = node.next.val;
            node.next = node.next.next;
        }
    }
}
// pay attention to not the tail, and java use . operation instead of ->
// listNode must use value and next both to remove node;


242	Valid Anagram
Given two strings s and t, write a function to determine if t is an anagram of s.
For example,
s = "anagram", t = "nagaram", return true.
s = "rat", t = "car", return false.

Note:
You may assume the string contains only lowercase alphabets.

public class Solution {
    public boolean isAnagram(String s, String t) {
        if (s.length() == 0 || t.length() == 0) {
            return s.length() == 0 && t.length() == 0;
        }
        
        char[] ss = s.toCharArray();
        char[] tt = t.toCharArray();
        
        Arrays.sort(ss);
        Arrays.sort(tt);
        
        String s1 = String.valueOf(ss);
        String s2 = String.valueOf(tt);

        if (s1.equals(s2)) {
            return true;
        }
        return false;
    }
}
// .toCharArray()   String.valueOf(ss);

public boolean isAnagram(String s, String t) {

    if(s.length() != t.length()) {
        return false;
    }

    int[] count = new int[26];

    for(int i = 0; i < s.length(); i++) {
        count[s.charAt(i) - 'a']++;
        count[t.charAt(i) - 'a']--;
    }

    for(int x : count) {
        if(x != 0) return false;
    }

    return true;
}


171	Excel Sheet Column Number
Given a column title as appear in an Excel sheet, return its corresponding column number.
For example:
    A -> 1
    B -> 2
    C -> 3
    ...
    Z -> 26
    AA -> 27
    AB -> 28 

public class Solution {
    public int titleToNumber(String s) {
        int ans = 0;
        if (s.length() == 0) {
            return ans;
        }
        
        for (int i = 0; i < s.length(); i++) {
            ans = ans * 26 + s.charAt(i) - 'A' + 1;
        }
        return ans;
    }
}

168. Excel Sheet Column Title
Given a positive integer, return its corresponding column title as appear in an Excel sheet.
For example:

    1 -> A
    2 -> B
    3 -> C
    ...
    26 -> Z
    27 -> AA
    28 -> AB 
public class Solution {
    public String convertToTitle(int n) {
        String result = "";
        if (n == 0) {
            return result;
        }
        
        while (n > 0) {
            result = (char)('A' + (n-1) % 26) + result;
            n = (n-1) / 26;
        }
        return result;
    }
}


252	Meeting Rooms
Given an array of meeting time intervals consisting of start and end times 
[[s1,e1],[s2,e2],...] (si < ei), determine if a person could attend all meetings.
For example,
Given [[0, 30],[5, 10],[15, 20]],
return false.
/**
 * Definition for an interval.
 * public class Interval {
 *     int start;
 *     int end;
 *     Interval() { start = 0; end = 0; }
 *     Interval(int s, int e) { start = s; end = e; }
 * }
 */
public class Solution {
    public boolean canAttendMeetings(Interval[] intervals) {
        Arrays.sort(intervals, new Comparator<Interval>() {
            public int compare(Interval a, Interval b) {
                return a.start-b.start;
            }
        });
        for(int i =0;i< intervals.length-1;i++){
            if(intervals[i].end > intervals[i+1].start)
                return false;
        }
        return true;
    }
}

169	Majority Element
Given an array of size n, find the majority element. The majority element is the element that
appears more than ⌊ n/2 ⌋ times.
You may assume that the array is non-empty and the majority element always exist in the array.

public class Solution {
    public int majorityElement(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        Arrays.sort(nums);
        return nums[nums.length/2];
    }
}

217	Contains Duplicate
Given an array of integers, find if the array contains any duplicates. Your function should return true
if any value appears at least twice in the array, and it should return false if every element is distinct.

public class Solution {
    public boolean containsDuplicate(int[] nums) {
        Set<Integer> s = new HashSet<>();
        for (int i = 0; i < nums.length; i++) {
            if (!s.add(nums[i])) {
                return true;
            }
        }
        return false;
    }
}

350	Intersection of Two Arrays II
Given two arrays, write a function to compute their intersection.

Example:
Given nums1 = [1, 2, 2, 1], nums2 = [2, 2], return [2, 2].

Each element in the result should appear as many times as it shows in both arrays.
The result can be in any order.

Follow up:
What if the given array is already sorted? How would you optimize your algorithm?
What if nums1's size is small compared to num2's size? Which algorithm is better?
What if elements of nums2 are stored on disk, and the memory is limited such that you cannot load all elements into the memory at once?

public class Solution {
    public int[] intersect(int[] nums1, int[] nums2) {
        if (nums1.length == 0 || nums2.length == 0) {
            return nums1.length == 0? nums1: nums2;
        }
        
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums1.length; i++) {
            if (map.get(nums1[i]) != null) {
                map.put(nums1[i], map.get(nums1[i])+1);
            }else {
                map.put(nums1[i], 1);
            }
        }
        ArrayList<Integer> list = new ArrayList<>();
        for (int i = 0; i < nums2.length; i++) {
            if (map.get(nums2[i]) != null && map.get(nums2[i]) > 0) {
                list.add(nums2[i]);
                map.put(nums2[i], map.get(nums2[i]) - 1);
            }
        }
        
        int[] ans = new int[list.size()];
        int index = 0;
        for (int i = 0; i < list.size(); i++) {
            ans[index++] = list.get(i);
        }
        return ans;
    }
}

If only nums2 cannot fit in memory, put all elements of nums1 into a HashMap, read chunks of array that fit
 into the memory, and record the intersections.

If both nums1 and nums2 are so huge that neither fit into the memory, sort them individually (external sort),
 then read 2 elements from each array at a time in memory, record intersections.

// pay attention. map.get(key) == -1 !!

public class Solution {
	public int[] intersect(int[] nums1, int[] nums2) {
	    Arrays.sort(nums2);
	    Arrays.sort(nums1);
	    List<Integer> m = new ArrayList<Integer>();
	    int y = 0;
	    int x = 0;

	    while (x < nums2.length && y < nums1.length) {
	        if (nums1[y] == nums2[x]) {
	            m.add(nums1[y]);
	            y++;
	            x++;
	        }else if(nums1[y]<nums2[x]){
	            y++;
	        }else{
	            x++;
	        }
	    }
	    int[] sum = new int[m.size()];
	    for(int i=0;i<m.size();i++){
	        sum[i] = m.get(i).intValue();
	    }
	    return sum;
	}
}


13	Roman to Integer
Given a roman numeral, convert it to an integer.
Input is guaranteed to be within the range from 1 to 3999.

public class Solution {
    public int romanToInt(String s) {
        // I V X L C D M
        int ans = 0;
        if (s.length() == 0) {
            return ans;
        }
        
        int pre = charToInt(s.charAt(0));
        ans = pre;
        
        for (int i = 1; i < s.length(); i++) {
            int cur = charToInt(s.charAt(i));
            if (cur > pre) {
                ans = ans - 2 * pre;
            }
            ans = ans + cur;
            pre = cur;
        }
        return ans;
    }
    
    public int charToInt(char c) {
        switch(c) {
            case 'I': return 1;
            case 'V': return 5;
            case 'X': return 10;
            case 'L': return 50;
            case 'C': return 100;
            case 'D': return 500;
            case 'M': return 1000;
        }
        return 0;
    }
}



191	Number of 1 Bits
public class Solution {
    // you need to treat n as an unsigned value
    public int hammingWeight(int n) {
        return Integer.toBinaryString(n).replace("0","").length();    
    }
}

//An ugly number must be multiplied by either 2, 3, or 5 from a smaller ugly number.
//The key is how to maintain the order of the ugly numbers. Try a similar approach of merging from three 
//sorted lists: L1, L2, and L3.
//Assume you have Uk, the kth ugly number. Then Uk+1 must be Min(L1 * 2, L2 * 3, L3 * 5).



70	Climbing Stairs
You are climbing a stair case. It takes n steps to reach to the top.
Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

public class Solution {
    public int climbStairs(int n) {
        if (n <= 1) {
            return 1;
        }
        int[] cost= new int[n+1];
        cost[0] = cost[1] = 1;
        
        for (int i = 2; i < n+1; i++) {
            cost[i] = cost[i-1] + cost[i-2];
        }
        
        return cost[n];
    }
}

202	Happy Number
Write an algorithm to determine if a number is "happy".

A happy number is a number defined by the following process: Starting with any positive integer, replace
the number by the sum of the squares of its digits, and repeat the process until the number equals 1 (where
it will stay), or it loops endlessly in a cycle which does not include 1. Those numbers for which this
process ends in 1 are happy numbers.

Example: 19 is a happy number

public class Solution {
    public boolean isHappy(int n) {
        HashSet<Integer> set = new HashSet<Integer>();
        set.add(n);
        while (n != 1) {
            int result = 0;
            while (n != 0) {
                result += Math.pow(n % 10, 2);
                n /= 10;
            }
            if (set.contains(result)) {
                return false;
            }
            set.add(result);
            n = result;
        }
        return true;
    }
}
// use set to check duplicate


27	Remove Element
Given an array and a value, remove all instances of that value in place and return the new length.
Do not allocate extra space for another array, you must do this in place with constant memory.
The order of elements can be changed. It doesn't matter what you leave beyond the new length.
Example:
Given input array nums = [3,2,2,3], val = 3
Your function should return length = 2, with the first two elements of nums being 2

public class Solution {
    public int removeElement(int[] nums, int val) {
        if (nums.length == 0) {
            return 0;
        }
        
        int result = nums.length;
        int count = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == val) {
                count++;
            }else{
                nums[i-count] = nums[i];
            }    
        }
        return result - count;
    }
}


232	Implement Queue using Stacks	
Implement the following operations of a queue using stacks.

push(x) -- Push element x to the back of queue.
pop() -- Removes the element from in front of queue.
peek() -- Get the front element.
empty() -- Return whether the queue is empty.
Notes:
You must use only standard operations of a stack -- which means only push to top, peek/pop from top, size,
and is empty operations are valid.
Depending on your language, stack may not be supported natively. You may simulate a stack by using a list
or deque (double-ended queue), as long as you use only standard operations of a stack.
You may assume that all operations are valid (for example, no pop or peek operations will be called on an
empty queue).

class MyQueue {
    Stack<Integer> s = new Stack<>();
    Stack<Integer> t = new Stack<>();
    // Push element x to the back of queue.
    public void push(int x) {
        s.push(x);
    }

    // Removes the element from in front of queue.
    public void pop() {
        while (!s.isEmpty()) {
            t.push((int) s.peek());
            s.pop();
        }
        t.pop();
        while (!t.isEmpty()) {
            s.push((int) t.peek());
            t.pop();
        }
    }

    // Get the front element.
    public int peek() {
        while (!s.isEmpty()) {
            t.push((int) s.peek());
            s.pop();
        }
        int result = (int) t.peek();
        while (!t.isEmpty()) {
            s.push((int) t.peek());
            t.pop();
        } 
        return result;
    }

    // Return whether the queue is empty.
    public boolean empty() {
        if (s.isEmpty()) {
            return true;
        }
        return false;
    }
}


26	Remove Duplicates from Sorted Array
Given a sorted array, remove the duplicates in place such that each element appear only once and return the
new length.
Do not allocate extra space for another array, you must do this in place with constant memory.

For example,
Given input array nums = [1,1,2],

Your function should return length = 2, with the first two elements of nums being 1 and 2 respectively. It
doesn't matter what you leave beyond the new length.'

public class Solution {
    public int removeDuplicates(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        int result = nums.length;
        int count = 0;
        for (int i = 0; i < nums.length; i++) {
            if (i+1 < nums.length && nums[i] == nums[i+1]) {
                count++;
            }else{
                nums[i-count] = nums[i];
            }
        }
        return result - count;
    }
}


118	Pascal's Triangle'
Given numRows, generate the first numRows of Pascal's triangle.'

For example, given numRows = 5,
Return

[
     [1],
    [1,1],
   [1,2,1],
  [1,3,3,1],
 [1,4,6,4,1]
]

public class Solution {
    public List<List<Integer>> generate(int numRows) {
        ArrayList<List<Integer>> lists = new ArrayList<List<Integer>>();
        if (numRows == 0) {
            return lists;
        }
        
        ArrayList<Integer> list = new ArrayList<>();
        list.add(1);
        lists.add(list);
        if (numRows == 1) {
            return lists;
        }
        for (int i = 2; i <= numRows; i++) {
            list = new ArrayList<>();
            list.add(1);
            List<Integer> tmp = lists.get(i-2); // pay attention to use List<Integer> here
            while (list.size() < i-1) {
                list.add(tmp.get(list.size()) + tmp.get(list.size() - 1));
            }
            list.add(1);
            lists.add(list);
        }
        return lists;
    }
}

public class Solution {
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> triangle = new ArrayList<List<Integer>>();
        if (numRows <=0){
            return triangle;
        }
        for (int i=0; i<numRows; i++){
            List<Integer> row =  new ArrayList<Integer>();
            for (int j=0; j<i+1; j++){
                if (j==0 || j==i){
                    row.add(1);
                } else {
                    row.add(triangle.get(i-1).get(j-1)+triangle.get(i-1).get(j));
                }
            }
            triangle.add(row);
        }
        return triangle;
    }
}

119	Pascal's Triangle II'
Given an index k, return the kth row of the Pascal's triangle.'
For example, given k = 3,
Return [1,3,3,1].

public class Solution {
    public List<Integer> getRow(int rowIndex) {
        ArrayList<Integer> list = new ArrayList<>();
        list.add(1);
        if (rowIndex == 0) {
            return list;
        }
        while (list.size() < rowIndex + 1) {
            ArrayList<Integer> tmp = new ArrayList<>();
            tmp.add(1);
            while (tmp.size() < list.size()) {
                tmp.add(list.get(tmp.size()) + list.get(tmp.size()-1));
            }
            tmp.add(1);
            list = tmp;
        }
        return list;
    }
}

public class Solution {
    public static List<Integer> getRow(int rowIndex) {
	    List<Integer> ret = new ArrayList<Integer>();
	    ret.add(1);
	    for (int i = 1; i <= rowIndex; i++) {
	        for (int j = i - 1; j >= 1; j--) {
	            int tmp = ret.get(j - 1) + ret.get(j);
	            ret.set(j, tmp);
	        }
	        ret.add(1);
	    }
	    return ret;
	}
}



9	Palindrome Number
Determine whether an integer is a palindrome. Do this without extra space.
public class Solution {
    public boolean isPalindrome(int x) {
        int sum = 0;
        int ori = x;
        while (x > 0) {
            sum = sum *10 + x%10;
            x /= 10;
        }
        
        if (sum == ori) {
            return true;
        }
        return false;
    }
}

249	Group Shifted Strings
Given a string, we can "shift" each of its letter to its successive letter, for example: "abc" -> "bcd".
We can keep "shifting" which forms the sequence:
"abc" -> "bcd" -> ... -> "xyz"
Given a list of strings which contains only lowercase alphabets, group all strings that belong to the same
shifting sequence.
For example, given: ["abc", "bcd", "acef", "xyz", "az", "ba", "a", "z"],
Return:
[
  ["abc","bcd","xyz"],
  ["az","ba"],
  ["acef"],
  ["a","z"]
]
Note: For the return value, each inner list's elements must follow the lexicographic order.'

public class Solution {
    public List<List<String>> groupStrings(String[] strings) {
        List<List<String>> result = new ArrayList<>();
        Map<String, List<String>> map = new HashMap<>();
        for(String s: strings){
            String key = getKey(s);
            if(!map.containsKey(key))
                map.put(key, new ArrayList<String>());
            map.get(key).add(s);
        }
        for(String key: map.keySet()){
            List<String> list = map.get(key);
            Collections.sort(list);
            result.add(list);
        }
        return result;
    }
    
    private String getKey(String s){
        StringBuffer key = new StringBuffer();
        for (int i = 1; i < s.length(); i++) {
            int asc = s.charAt(i) - s.charAt(i-1);
            char c = (char) (asc>=0?asc:asc+26);
            key.append(c);
        }
        return key.toString();
    }
}


36	Valid Sudoku
The Sudoku board could be partially filled, where empty cells are filled with the character '.'.
public class Solution {
    public boolean isValidSudoku(char[][] board) {
        int row = board.length;
        int col = board[0].length;
        
        for (int i = 0; i < row; i++) {
            Set<Character> s = new HashSet<>();
            Set<Character> t = new HashSet<>();
            for (int j = 0; j < col; j++) {
                if (board[i][j] != '.') {
                    if (!s.add(board[i][j])) {
                        return false;
                    }
                }
                if (board[j][i] != '.') {
                    if (!t.add(board[j][i])) {
                        return false;
                    }
                }
                if (i % 3 == 0 && j % 3 == 0) {
                    if (!check(board, i, j)) {
                        return false;
                    }
                }
            }
        }
        return true;
    }
    
    public boolean check(char[][] board, int r, int c) {
        Set<Character> set = new HashSet<>();
        for (int i = r; i < r + 3; i++) {
            for (int j = c; j < c + 3; j++) {
                if (board[i][j] != '.') {
                    if (!set.add(board[i][j])) {
                        return false;
                    }
                }
            }
        }
        return true;
    }
}

public class Solution {
    public boolean isValidSudoku(char[][] board) {
        for (int i=0; i<9; i++) {
            if (!isParticallyValid(board,i,0,i,8)) return false;
            if (!isParticallyValid(board,0,i,8,i)) return false;
        }
        for (int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                if (!isParticallyValid(board,i*3,j*3,i*3+2,j*3+2)) return false;
            }
        }
        return true;
    }
    private boolean isParticallyValid(char[][] board, int x1, int y1,int x2,int y2){
        Set singleSet = new HashSet();
        for (int i= x1; i<=x2; i++){
            for (int j=y1;j<=y2; j++){
                if (board[i][j]!='.') if(!singleSet.add(board[i][j])) return false;
            }
        }
        return true;
    }
}


225	Implement Stack using Queues
Implement the following operations of a stack using queues.

push(x) -- Push element x onto stack.
pop() -- Removes the element on top of the stack.
top() -- Get the top element.
empty() -- Return whether the stack is empty.

class MyStack {
    // Push element x onto stack.
    Queue<Integer> q = new LinkedList<>();
    public void push(int x) {
        q.add(x);
        for (int i = 1; i < q.size(); i++) {
            q.add(q.remove());
        }
    }

    // Removes the element on top of the stack.
    public void pop() {
        q.remove();   
    }

    // Get the top element.
    public int top() {
        return q.peek();    
    }

    // Return whether the stack is empty.
    public boolean empty() {
        return q.isEmpty();
    }
}


88	Merge Sorted Array
Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.
Note:
You may assume that nums1 has enough space (size that is greater or equal to m + n) to hold additional
elements from nums2. The number of elements initialized in nums1 and nums2 are m and n respectively.

public class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        if (n == 0) {
            return;
        }
        int index = m+n-1;
        int i = m-1;
        int j = n-1;
        while (i >= 0 && j >= 0) {
            if (nums1[i] >= nums2[j]) {
                nums1[index--] = nums1[i];
                i--;
            }else {
                nums1[index--] = nums2[j];
                j--;
            }
        }
        while ( j >= 0) {
            nums1[index--] = nums2[j];
            j--;
        }
        return;
    }
}

public void merge(int[] nums1, int m, int[] nums2, int n) {
    while(n>0){
        if(m>0&&nums1[m-1]>nums2[n-1]){
            nums1[m+n-1] = nums1[m-1];
            m--;
        }
        else{
            nums1[m+n-1] = nums2[n-1];
            n--;
        }
    }
}


223	Rectangle Area
Find the total area covered by two rectilinear rectangles in a 2D plane.
Each rectangle is defined by its bottom left corner and top right corner as shown in the figure.

Rectangle Area
Assume that the total area is never beyond the maximum possible value of int.

public class Solution {
    public int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {
        return (C-A) * (D-B) + (G-E) * (H-F) - computerAreaJoin(A, B, C, D, E, F, G, H);
    }
    
    public int computerAreaJoin(int A, int B, int C, int D, int E, int F, int G, int H) {
        int hTop = Math.min(D,H);
        int hLow = Math.max(B,F);
        
        int wTop = Math.min(C,G);
        int wLow = Math.max(A,E);
        if (hTop < hLow || wTop < wLow) {
            return 0;
        }
        return (hTop - hLow) * (wTop - wLow);
    }
}


299	Bulls and Cows
You are playing the following Bulls and Cows game with your friend: You write down a number and ask your
friend to guess what the number is. Each time your friend makes a guess, you provide a hint that indicates
how many digits in said guess match your secret number exactly in both digit and position (called "bulls")
and how many digits match the secret number but locate in the wrong position (called "cows"). Your friend
will use successive guesses and hints to eventually derive the secret number.

For example:

Secret number:  "1807"
Friend's guess: "7810"
Hint: 1 bull and 3 cows. (The bull is 8, the cows are 0, 1 and 7.)
Write a function to return a hint according to the secret number and friend's guess, use A to indicate the
bulls and B to indicate the cows. In the above example, your function should return "1A3B".

Please note that both secret number and friend's guess may contain duplicate digits, for example:

Secret number:  "1123"
Friend's guess: "0111"
In this case, the 1st 1 in friend's guess is a bull, the 2nd or 3rd 1 is a cow, and your function should
return "1A1B".
You may assume that the secret number and your friend's guess only contain digits, and their lengths are
always equal.

public class Solution {
    public String getHint(String secret, String guess) {
        if (secret.length() == 0) {
            return "0A0B";
        }
        Map<Character, Integer> s = new HashMap<>();
        Map<Character, Integer> g = new HashMap<>();
        int bull = 0;
        int cow = 0;
        for (int i = 0; i < secret.length(); i++) {
            if (secret.charAt(i) == guess.charAt(i)) {
                bull++;
            }else {
                if (s.get(secret.charAt(i)) != null) {
                    s.put(secret.charAt(i),s.get(secret.charAt(i))+1);
                }else {
                    s.put(secret.charAt(i),1);
                }
                if (g.get(guess.charAt(i)) != null) {
                    g.put(guess.charAt(i),g.get(guess.charAt(i))+1);
                }else {
                    g.put(guess.charAt(i),1);
                }
            }    
        }
        
        for (char c : s.keySet()) {
            if (g.get(c) != null) {
                cow += Math.min(s.get(c), g.get(c));
            }
        }
        
        String result = bull + "A" + cow + "B";
        return result;
    }
}



290	Word Pattern
Given a pattern and a string str, find if str follows the same pattern.
Here follow means a full match, such that there is a bijection between a letter in pattern and a non-empty
word in str.

Examples:
pattern = "abba", str = "dog cat cat dog" should return true.
pattern = "abba", str = "dog cat cat fish" should return false.
pattern = "aaaa", str = "dog cat cat dog" should return false.
pattern = "abba", str = "dog dog dog dog" should return false.
Notes:
You may assume pattern contains only lowercase letters, and str contains lowercase letters separated by
a single space.

public class Solution {
    public boolean wordPattern(String pattern, String str) {
        String[] array = str.trim().split(" ");
        if (pattern.length() != array.length) {
            return false;
        }
        
        Map<Character, String> map = new HashMap<>();
        for (int i = 0; i < pattern.length(); i++) {
            if (map.get(pattern.charAt(i)) == null) {
                if (map.values().contains(array[i])) {
                    return false;
                }else {
                    map.put(pattern.charAt(i), array[i]);
                }
            }else {
                if (!map.get(pattern.charAt(i)).equals(array[i])) {
                    return false;
                }
            }
        }
        return true;
    }
}


58	Length of Last Word
Given a string s consists of upper/lower-case alphabets and empty space characters ' ', return the length
of last word in the string.
If the last word does not exist, return 0.
Note: A word is defined as a character sequence consists of non-space characters only.
For example, 
Given s = "Hello World",
return 5.

public class Solution {
    public int lengthOfLastWord(String s) {
        if (s.length() == 0) {
            return 0;
        }
        
        String[] array = s.trim().split("\\s+");
        return array[array.length-1].length();
    }
}

public int lengthOfLastWord(String s) {
    return s.trim().length()-s.trim().lastIndexOf(" ")-1;
}
// use ""



203	Remove Linked List Elements
Remove all elements from a linked list of integers that have value val.

Example
Given: 1 --> 2 --> 6 --> 3 --> 4 --> 5 --> 6, val = 6
Return: 1 --> 2 --> 3 --> 4 --> 5

public class Solution {
    public ListNode removeElements(ListNode head, int val) {
        while (head != null && head.val == val) {
            head = head.next;
        }
        if (head == null) {
            return head;
        }
        
        ListNode tmp = head;
        ListNode upper = tmp;
        
        while (tmp != null) {
            if (tmp.val == val) {
                if (tmp.next != null) {
                    upper.next = tmp.next;
                    tmp = upper.next;
                }else {
                    upper.next = null;
                    return head;
                }
            }else {
                upper = tmp;
                tmp = tmp.next;
            }    
        }
        return head;
    }
}


14	Longest Common Prefix
Write a function to find the longest common prefix string amongst an array of strings.

public class Solution {
    public String longestCommonPrefix(String[] strs) {
        String ans = "";
        if (strs.length == 0) {
            return ans;
        }
        
        ans = strs[0];
        
        for (int i = 1; i < strs.length; i++) {
            while (strs[i].indexOf(ans) != 0) {
                ans = ans.substring(0,ans.length()-1);
            }
        }
        return ans;
    }
}

67	Add Binary
Given two binary strings, return their sum (also a binary string).

For example,
a = "11"
b = "1"
Return "100".

class Solution {
    public String addBinary(String a, String b) {
        int c = 0;
        StringBuilder sb = new StringBuilder();
        for(int i = a.length() - 1, j = b.length() - 1; i >= 0 || j >= 0;){
            if(i >= 0) c += a.charAt(i--) - '0';
            if(j >= 0) c += b.charAt(j--) - '0';
            sb.insert(0, (char)((c % 2) + '0'));
            c /= 2;
        }
        if(c == 1) sb.insert(0, "1");
        return sb.toString();
    }
}


303	Range Sum Query - Immutable
Given an integer array nums, find the sum of the elements between indices i and j (i ≤ j), inclusive.

Example:
Given nums = [-2, 0, 3, -5, 2, -1]
sumRange(0, 2) -> 1
sumRange(2, 5) -> -1
sumRange(0, 5) -> -3
Note:
You may assume that the array does not change.
There are many calls to sumRange function.

public class NumArray {
    int nums[];
    int[] sums;
    int s;
    public NumArray(int[] nums) {
        this.nums = nums;
        sums = new int[nums.length];
        for (int i = 0; i < nums.length; i++) {
            s = s + nums[i];
            sums[i] = s;
        }
        
    }

    public int sumRange(int i, int j) {
        if (i > j) {
            return 0;
        }
        return sums[j]-sums[i] + nums[i];
    }
}


// remember to initialize sbs!!!
125	Valid Palindrome
Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring
cases.
For example,
"A man, a plan, a canal: Panama" is a palindrome.
"race a car" is not a palindrome.
Note:
Have you consider that the string might be empty? This is a good question to ask during an interview.
For the purpose of this problem, we define empty string as valid palindrome.

public class Solution {
  public boolean isPalindrome(String s) {
    if (s.length() == 0) {
      return true;
    }
    String ss = s.toLowerCase().replaceAll("[^a-z0-9]", "");
    int i = 0;
    int j = ss.length() - 1;
    while (i <= j) {
      if (ss.charAt(i) != ss.charAt(j)) {
        return false;
      }
      i++;
      j--;
    }
    return true;
  }

}

170	Two Sum III - Data structure design
Design and implement a TwoSum class. It should support the following operations: add and find.
add - Add the number to an internal data structure.
find - Find if there exists any pair of numbers which sum is equal to the value.

For example,
add(1); add(3); add(5);
find(4) -> true
find(7) -> false

public class TwoSum {
  private HashMap<Integer, Integer> elements = new HashMap<Integer, Integer>();
 
  public void add(int number) {
    if (elements.containsKey(number)) {
      elements.put(number, elements.get(number) + 1);
    } else {
      elements.put(number, 1);
    }
  }
 
  public boolean find(int value) {
    for (Integer i : elements.keySet()) {
      int target = value - i;
      if (elements.containsKey(target)) {
        if (i == target && elements.get(target) < 2) {// check one pair, not the same 1 sum=2, only one 1
          continue;
        }
        return true;
      }
    }
    return false;
  }
}

1	Two Sum
Given an array of integers, return indices of the two numbers such that they add up to a specific target.
You may assume that each input would have exactly one solution.

Example:
Given nums = [2, 7, 11, 15], target = 9,
Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].

public class Solution {
    public int[] twoSum(int[] nums, int target) {
        int[] ans = new int[2];
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.get(target-nums[i]) == null) {
                map.put(nums[i], i);
            }else {
                if (map.get(target-nums[i]) != null) {
                    ans[0] = map.get(target-nums[i]);
                    ans[1] = i;
                    break;
                }
            }
        }
        return ans;
    }
}



278	First Bad Version
You are a product manager and currently leading a team to develop a new product. Unfortunately, the latest
version of your product fails the quality check. Since each version is developed based on the previous
version, all the versions after a bad version are also bad.

Suppose you have n versions [1, 2, ..., n] and you want to find out the first bad one, which causes all
the following ones to be bad.

You are given an API bool isBadVersion(version) which will return whether version is bad. Implement a
function to find the first bad version. You should minimize the number of calls to the API.

public class Solution extends VersionControl {
    public int firstBadVersion(int n) {
        if (n == 1) {
            return n;
        }
        int lo = 1;
        int hi = n;
        
        while (lo <= hi) {
            int m = lo + (hi-lo)/2;
            if (isBadVersion(m)) {
                hi = m - 1;
            }else {
                lo = m + 1;
            }
        }
        return lo;
    }
}
// use [1,2,3] check

155	Min Stack
Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

push(x) -- Push element x onto stack.
pop() -- Removes the element on top of the stack.
top() -- Get the top element.
getMin() -- Retrieve the minimum element in the stack.
Example:
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin();   --> Returns -3.
minStack.pop();
minStack.top();      --> Returns 0.
minStack.getMin();   --> Returns -2.

public class MinStack {
    long min;
    Stack<Long> stack;

    public MinStack(){
        stack=new Stack<>();
    }

    public void push(int x) {
        if (stack.isEmpty()){
            stack.push(0L);
            min=x;
        }else{
            stack.push(x-min);//Could be negative if min value needs to change
            if (x<min) min=x;
        }
    }

    public void pop() {
        if (stack.isEmpty()) return;
        long pop=stack.pop();
        if (pop<0)  min=min-pop;//If negative, increase the min value

    }

    public int top() {
        long top=stack.peek();
        if (top>0){
            return (int)(top+min);
        }else{
           return (int)(min);
        }
    }

    public int getMin() {
        return (int)min;
    }
}
// remember to use Long
// 




