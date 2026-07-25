
const twoSum = function(nums, target) {
    const m = new Map();
    for (let i = 0; i < nums.length; ++i) {
        const x = nums[i];
        const y = target - x;
        if (m.has(y)) {
            return [m.get(y), i];
        }
        m.set(x, i);
    }
}




