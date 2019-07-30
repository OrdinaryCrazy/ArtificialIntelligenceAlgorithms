// ======================================================================
// OOP 面向对象编程
// 设计与问题本质相对应的数据格式——类，Bottom-Up: 类是低级组织，程序是高级组织
// 示例程序：
// 编译： g++ -std=c++11 learn1.cpp -o learn1 
// ======================================================================
#include <iostream> /* 类库 */

#include <climits>
/* climits 头文件定义了诸如 INT_MAX, SHRT_MAX 之类与系统有关的符号常量
 * 来表示对于类型的限制
 */
#include <cfloat>
/* cfloat 头文件定义了系统对于浮点数的限制 */
void backward(void)
{
    using namespace std;
    cout << "Enter your agent code:_____\b\b\b\b\b";
    long code;
    cin >> code;
    cout << "\aYou Entered: " << code << endl;
}

int main(void)
{
    /* 名称空间：代码的封装单元 */
    using namespace std;
    /* 流与插入运算符(运算符重载) */
    cout << "Learning C++" << endl;

    cout << "Size Of INT: " << sizeof(int) << endl;
    //----------------------------------------------
    /* 修改输出流基数 */
    // cout << hex;
    // cout << 0x87 << endl;
    // cout << dec;
    // cout << 0x87 << endl;
    //----------------------------------------------
    /* 转义字符 */
    // backward();
    //----------------------------------------------
    /* 类型转换 */
    // int sta = 5;
    // cout << typeid(sta).name() << endl;
    // cout << typeid( static_cast<long> (sta) ).name() << endl;
    //----------------------------------------------
    /* String 类 */
    string str = "hello";
    cout << str << endl;
    cout << str.size() << endl;

    return 0;
}
/* 
 * 类：1，表示什么信息；2，执行什么操作
 * 类之于对象就像类型之于变量
 */