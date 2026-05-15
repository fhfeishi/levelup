

function TwoSum(num_lst, target, num=2){
    let numTwoIndex = new Map()   
    for (let i =0; i <num_lst.length;i++){
        let complement = target - num_lst[i];
        if (numTwoIndex.has(complement)){
            return [numTwoIndex.get(complement), i]
        }
        numTwoIndex.set(num_lst[i],i)
    }
}
console.log(TwoSum([2,7,11,15], 9))