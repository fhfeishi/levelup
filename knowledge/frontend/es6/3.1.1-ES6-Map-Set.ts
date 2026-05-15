// 3.1.2 Es6 : Map Set


var myMap = new Map();
var keyString = 'a string';

myMap.set(keyString, "value associated with 'a string'");

myMap.get(keyString); // "value associated with 'a string'"
myMap.get("a string"); // "value associated with 'a string'" 

console.log(myMap.get(keyString));  // ❌ undefined   不要用 [] 或 . 运算符，那是对象的写法。
// Map 不是普通的对象（{}），所以不能用 obj[key] 这种方式来访问。
// myMap[keyString] 实际上相当于去 Map 对象自身的属性表里查找 "a string" 这个键，
// 而 Map 内部存储的键值对不会放到这里。


console.log(myMap[keyString]);  // ❌ undefined    不要用 [] 或 . 运算符，那是对象的写法。
// 这其实是在访问 myMap 对象的属性 keyString。
// 但 Map 对象并没有这个属性（它的值存放在内部数据结构里）


// Map 迭代
// Map.entries() returns an iterator of key/value pairs for each element in the Map object
// for (var [key, value] of myMap.entries()) {...}

// myMap[key1] = value1
// myMap.key1 = value1

// Map.keys() returns an iterator of keys in the Map object
// for (var key of myMap.keys()) {...}

// Map.values() returns an iterator of values in the Map object
// for (var value of myMap.values()) {...} 

// Map.forEach() calls a function for each key/value pair in the Map object
// Map.forEach((value, key) => {
//   console.log(key + ' = ' + value);
// });


// 在 JavaScript 里：
// 函数是对象，不同的函数，即便长得一模一样（代码相同），它们在内存中的引用也是不一样的。
// Map 的键比较用的是 引用相等（SameValueZero），即必须是 同一个对象引用 才算同一个键。


// 元组  数组
// ES6/JavaScript：没有元组，只有数组；数组可以用来模拟元组，但没有类型约束。
// TypeScript：专门有元组（Tuple）类型，可以严格规定长度和每个位置的类型。



// Set 
// Set 是一个值的集合，类似于数组，但成员的值都是唯一的，没有重复的值。
// +0 与 -0 在存储判断唯一性的时候是恒等的，所以不重复；
// undefined 与 undefined 是恒等的，所以不重复；
// NaN 与 NaN 是不恒等的，但是在 Set 中只能存一个，不重复。

let mySet = new Set();
mySet.add(1); // Set { 1 }
mySet.add(+0); // Set { 1, 0 }
mySet.add(-0); // Set { 1, 0 } 
mySet.add(NaN); // Set { 1, 0, NaN }
mySet.add(undefined); // Set { 1, 0, NaN, undefined }


mySet.add(NaN); // Set { 1, 0, NaN, undefined }


// Set.has()   

// Array.filter(  ... )


// 3.1.2   Reflect  Proxy 

// Proxy 与 Reflect 是 ES6 为了操作对象引入的 API 。
//     Proxy 可以对目标对象的读取、函数调用等操作进行拦截，然后进行操作处理。
// 它不直接操作对象，而是像代理模式，通过对象的代理对象进行操作，
// 在进行这些操作时，可以添加一些需要的额外操作。
//     Reflect 可以用于获取目标对象的行为，它与 Object 类似，
// 但是更易读，为操作对象提供了一种更优雅的方式。它的方法与 Proxy 是对应的。



// 点语法 obj.key → 固定属性名 "key"
// 方括号 obj[key] → 动态属性名，取变量的值



// 3.2.1 ES6 字符串 string  

// includes()：返回布尔值，判断是否找到参数字符串。
// startsWith()：返回布尔值，判断参数字符串是否在原字符串的头部。
// endsWith()：返回布尔值，判断参数字符串是否在原字符串的尾部。






// 3.2.4 ES6 数组 array  


// Array.of()  将参数中所有值作为元素形成数组。
// Array.from()  将类数组对象或可迭代对象转化为数组。

// 视图  DataView 


