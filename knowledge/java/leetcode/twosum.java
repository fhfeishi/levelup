package my_java;

public class twosum {
    public twosum(){}
    public int add(int a, int b){
        return a + b;
    }
    public static void main(String[] args){
        twosum twosum_a = new twosum();
        int result = twosum_a.add(5, 7);
        System.out.println("5+7=" + result);

    }
}
