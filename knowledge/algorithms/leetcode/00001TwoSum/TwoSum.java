import java.util.HashMap;
import java.util.Map;

public class TwoSum{
    public static int[] TwoSum(int[] nums, int target){
        Map<Integer, Integer> numToIndex =new HashMap<>();
        for (int i=0;;++i){
            int complement = target - nums[i]; 
            if (numToIndex.containsKey(complement)){
                return new int[] {numToIndex.get(complement), i};
            }
            numToIndex.put(nums[i], i);
        }
    }
    public static void main(String[] args){
        int[] result = TwoSum(new int[]{2,7,11,15}, 9);
        System.out.println("[" + result[0] + "," + result[1] + "]");
    }

}


