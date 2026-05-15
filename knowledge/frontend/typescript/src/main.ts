import './style.css'
import typescriptLogo from './typescript.svg'
import viteLogo from '/vite.svg'
import { setupCounter } from './counter.ts'

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <div>
    <a href="https://vite.dev" target="_blank">
      <img src="${viteLogo}" class="logo" alt="Vite logo" />
    </a>
    <a href="https://www.typescriptlang.org/" target="_blank">
      <img src="${typescriptLogo}" class="logo vanilla" alt="TypeScript logo" />
    </a>
    <h1>Vite + TypeScript</h1>
    <div class="card">
      <button id="counter" type="button"></button>
    </div>
    <p class="read-the-docs">
      Click on the Vite and TypeScript logos to learn more
    </p>
  </div>
`
setupCounter(document.querySelector<HTMLButtonElement>('#counter')!)

const hello: string = 'Hello, TypeScript!';
console.log(hello);

enum Role {
  Admin,
  User,
  Guest,
}


interface User {
  id: number,
  username: string,
  role: Role,
  hobbies: string[],
  contactInfo: [area: string, phoneNumber: number]  //
}


var str = '1' 
var str2:number = <number> <any> str   //str、str2 是 string 类型
console.log(str2)
console.log(typeof str2)   // string 


var num1:number = 1
var res1 = num1 >0 ? "zheng" : "fu"
console.log(res1)  // zheng

/*
for (init; condition; increment) {
  // 循环体
}
*/


var num2: number=3;
var i:number;
var fact2 = 1;
for(i=num2;i>=1;i--){
  fact2 *= i;
}
console.log(fact2);  // 6



/*
for...in 语句用于一组值的集合或列表进行迭代输出。
for (var val in list) {
  // 循环体
}

*/




/**
 * while  先测试条件再循环
 * while (condition){ statement(s); }
 * 
 */


/**
 * do ... while
 * 
 * do{ statement(s); } while (condition); 先循环再测试条件
 * 
 */

/**  break; */
/**  continue */


/** for(;;) {console.log("这段代码会不停的执行");}  */
/** while(true) {console.log("这段代码会不停的执行");} */


/**  递归函数即在函数内调用函数本身。  */

/** lambda 
 * ( [param1, param2,…param n] )=>statement;
 */


/** Map  python: OrderedDict   Object不一定有序 */
