#define DEPTH 3
#include <omp.h>
#include <queue>
// #define DEBUG
//----------------------------------------------------------------
struct step
{
    int x;
    int y;
};
typedef struct step step;
typedef struct order
{
    int x;
    int y;
    int priority;
    //-------------------------------------------------------
    friend bool operator <(const order &a, const order &b)
    {
        return a.priority < b.priority;
    }
}order;
int startup[15][15] = 
{
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
    0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0,
    0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 0,
    0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 3, 2, 1, 0,
    0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 4, 3, 2, 1, 0,
    0, 1, 2, 3, 4, 5, 6, 6, 6, 5, 4, 3, 2, 1, 0,
    0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0,
    0, 1, 2, 3, 4, 5, 6, 6, 6, 5, 4, 3, 2, 1, 0,
    0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 4, 3, 2, 1, 0,
    0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 3, 2, 1, 0,
    0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 0,
    0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
//----------------------------------------------------------------
#define MAX 1   // HUMAN
#define MIN 0   // AGENT
void printBoard(int**board);
int gameover(int** board)
{
    // 棋盘上 -1 表示电脑的黑子， 1 表示人的白子， 0 表示空白
    // 返回 -1 表示电脑获胜， 1 表示人获胜， 0 表示未分胜负
    // UP-DOWN----------------------------------------------------
    for(int i = 0; i < 11; i++)
    {
        for(int j = 0; j < 15; j++)
        {
            if( board[i][j] != 0 &&
                board[i][j] == board[i + 1][j]  &&  board[i][j] == board[i + 2][j]  &&
                board[i][j] == board[i + 3][j]  &&  board[i][j] == board[i + 4][j]  )
            {
                return board[i][j];
            }
        }
    }
    // RIGHT-LEFT----------------------------------------------------
    for(int i = 0; i < 15; i++)
    {
        for(int j = 0; j < 11; j++)
        {
            if( board[i][j] != 0 &&
                board[i][j] == board[i][j + 1]  &&  board[i][j] == board[i][j + 2]  &&
                board[i][j] == board[i][j + 3]  &&  board[i][j] == board[i][j + 4]  )
            {
                return board[i][j];
            }
        }
    }
    // UL-DR----------------------------------------------------
    for(int i = 0; i < 11; i++)
    {
        for(int j = 0; j < 11; j++)
        {
            if( board[i][j] != 0 &&
                board[i][j] == board[i + 1][j + 1]  &&  board[i][j] == board[i + 2][j + 2]  &&
                board[i][j] == board[i + 3][j + 3]  &&  board[i][j] == board[i + 4][j + 4]  )
            {
                return board[i][j];
            }
        }
    }
    // UR-DL----------------------------------------------------
    for(int i = 4; i < 15; i++)
    {
        for(int j = 0; j < 11; j++)
        {
            if( board[i][j] != 0 &&
                board[i][j] == board[i - 1][j + 1]  &&  board[i][j] == board[i - 2][j + 2]  &&
                board[i][j] == board[i - 3][j + 3]  &&  board[i][j] == board[i - 4][j + 4]  )
            {
                return board[i][j];
            }
        }
    }
    //----------------------------------------------------
    return 0;
}
#include "evaluate.h"
int evaluate(int** board, int player)
{
    //return 0;
    int max_benefit = 0;    // HUMAN 当前的优势
    int min_benefit = 0;    // AGENT 当前的优势

    // 棋型：长连 11111
    switch( gameover(board) )
    {
        case -1: min_benefit += 100000000; return max_benefit - min_benefit;
        case  1: max_benefit += 100000000; return max_benefit - min_benefit;
    }

    typeCount tempTC;

    omp_set_num_threads(6); // 设置线程数量
    int penalty[6] = {40000000, 1000000, 500000, 10000, 500, 10};
    //=============================================================================================
    #pragma omp parallel private(tempTC)
    {
        int id = omp_get_thread_num();
        switch( id )
        {
            case 0: tempTC = PerfectFour(board);    break;
            case 1: tempTC = ThreatFour(board);     break;
            case 2: tempTC = ThreatThree(board);    break;
            case 3: tempTC = TryThree(board);       break;
            case 4: tempTC = GoodTwo(board);        break;
            case 5: tempTC = LimitedTwo(board);     break;
        }
        #pragma omp critical
        {
            min_benefit += tempTC.minC * tempTC.minC * penalty[id];
            max_benefit += tempTC.maxC * tempTC.maxC * penalty[id];
        }
    }
    //=============================================================================================
    for(int i = 0; i < 15; i++)
    {
        for(int j = 0; j < 15; j++)
        {
            if(board[i][j] ==  1)   max_benefit += startup[i][j];
            if(board[i][j] == -1)   min_benefit += startup[i][j];
        }
    }
    player == MAX ? max_benefit *= 2 : min_benefit *= 2;
    return max_benefit - min_benefit;
}
int AlphaBetaMINIMAX(int** board, int depth, int player, int alpha, int beta, step& next)
{
    if( depth == 0 || gameover(board) != 0 )
    {
        return evaluate(board, player);
    }
//====================================================================================================
    order pri[15][15];
    std::priority_queue<order> expandCandidate;
    for(int i = 0; i < 15; i++)
    {
        for(int j = 0; j < 15; j++)
        {
            if(board[i][j] == 0)
            {
                pri[i][j].x = i;
                pri[i][j].y = j;
                pri[i][j].priority =  startup[i][j];
        i > 1 ? pri[i][j].priority += board[i - 1][j    ] == MAX : 0;
        i <14 ? pri[i][j].priority += board[i + 1][j    ] == MAX : 0;
        j > 1 ? pri[i][j].priority += board[i    ][j - 1] == MAX : 0;
        j <14 ? pri[i][j].priority += board[i    ][j + 1] == MAX : 0;
 i > 1&&j > 1 ? pri[i][j].priority += board[i - 1][j - 1] == MAX : 0;
 i > 1&&j <14 ? pri[i][j].priority += board[i - 1][j + 1] == MAX : 0;
 i <14&&j > 1 ? pri[i][j].priority += board[i + 1][j - 1] == MAX : 0;
 i <14&&j <14 ? pri[i][j].priority += board[i + 1][j + 1] == MAX : 0;
                expandCandidate.push( pri[i][j] );
            }
        }
    }
//===================================== MAX ==========================================================
    if(player == MAX)
    {
        while(!expandCandidate.empty())
        {
            order searchingGrid = expandCandidate.top();
            expandCandidate.pop();
            board[searchingGrid.x][searchingGrid.y] = 1;
            int score = AlphaBetaMINIMAX(board, depth - 1, player^1, alpha, beta, next);
            board[searchingGrid.x][searchingGrid.y] = 0;
            if(score > alpha)
            {
                alpha = score;
                if(depth == DEPTH)
                {
                    next.x = searchingGrid.x;
                    next.y = searchingGrid.y;
                }
            }
            if(alpha >= beta)    // 剪枝，其父必不选
            {
                return alpha;
            }
        }
        return alpha;
    }
//===================================== MIN ==========================================================
    else
    {
        while(!expandCandidate.empty())
        {
            order searchingGrid = expandCandidate.top();
            expandCandidate.pop();
            board[searchingGrid.x][searchingGrid.y] = -1;
            int score = AlphaBetaMINIMAX(board, depth - 1, player^1, alpha, beta, next);
            board[searchingGrid.x][searchingGrid.y] = 0;
            if(score < beta)
            {
                beta = score;
                if(depth == DEPTH)
                {
                    next.x = searchingGrid.x;
                    next.y = searchingGrid.y;
                }
            }
            if(alpha >= beta)    // 剪枝，其父必不选
            {
                return beta;
            }
        }
        return beta;
    }
//======================================================================================================
}
void printBoard(int**board)
{   
    printf("-----------------------------------------------------------------------\n");
    printf("   ");
    for(int j = 0; j < 15; j++)
    {
        printf("%2d ", j);
    }
    putchar('\n');
    for(int i = 0; i < 15; i++)
    {
        printf("%2d ", i);
        for(int j = 0; j < 15; j++)
        {
            printf("%2d ", board[i][j]);
        }
        putchar('\n');
    }
    printf("-----------------------------------------------------------------------\n");
}