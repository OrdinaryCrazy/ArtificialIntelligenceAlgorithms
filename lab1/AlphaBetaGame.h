#include <cstring>
//----------------------------------------------------------------
struct step
{
    int x;
    int y;
};
typedef struct step step;
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
int gameover(int** board)
{
    // 棋盘上 -1 表示电脑的黑子， 1 表示人的白子， 0 表示空白
    // 返回 -1 表示电脑获胜， 1 表示人获胜， 0 表示未分胜负
    // UP-DOWN----------------------------------------------------
    for(int i = 0; i < 11; i++)
    {
        for(int j = 0; j < 15; j++)
        {
            if( board[i][j] == board[i + 1][j]  &&  board[i][j] == board[i + 2][j]  &&
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
            if( board[i][j] == board[i][j + 1]  &&  board[i][j] == board[i][j + 2]  &&
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
            if( board[i][j] == board[i + 1][j + 1]  &&  board[i][j] == board[i + 2][j + 2]  &&
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
            if( board[i][j] == board[i - 1][j + 1]  &&  board[i][j] == board[i - 2][j + 2]  &&
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
int evaluate(int** board)
{
    int max_benefit = 0;    // HUMAN 当前的优势
    int min_benefit = 0;    // AGENT 当前的优势

    // 棋型：长连 11111
    switch( gameover(board) )
    {
        case -1: min_benefit += 1e6; return max_benefit - min_benefit;
        case  1: max_benefit += 1e6; return max_benefit - min_benefit;
    }

    typeCount tempTC;

    // 棋型：活四 011110
    tempTC = PerfectFour(board);
    min_benefit += tempTC.minC * 1e6;
    max_benefit += tempTC.maxC * 1e6;

    // 棋型：冲四 011112 or 10111 or 11011 or 11101 or 2111110
    tempTC = ThreatFour(board);
    min_benefit += tempTC.minC * tempTC.minC * 1e5;
    max_benefit += tempTC.maxC * tempTC.maxC * 1e5;

    // 棋型：活三 01110 or 010110 or 011010
    tempTC = ThreatThree(board);
    min_benefit += tempTC.minC * tempTC.minC * 1e5;
    max_benefit += tempTC.maxC * tempTC.maxC * 1e5;

    // 棋型：眠三 001112 or 211100 or 010112 or 011012 or 10011 or 11001 or 10101 or 2011102
    tempTC = TryThree(board);
    min_benefit += tempTC.minC * tempTC.minC * 1e4;
    max_benefit += tempTC.maxC * tempTC.maxC * 1e4;

    // 棋型：活二 00110 or 01100 or 01010 or 010010
    tempTC = GoodTwo(board);
    min_benefit += tempTC.minC * tempTC.minC * 1e4;
    max_benefit += tempTC.maxC * tempTC.maxC * 1e4;

    // 棋型：眠二 000112 or 211000 or 010012 or 10001 or 2010102 or 2011002
    tempTC = LimitedTwo(board);
    min_benefit += tempTC.minC * tempTC.minC * 1e3;
    max_benefit += tempTC.maxC * tempTC.maxC * 1e3;

    return max_benefit - min_benefit;
}
int AlphaBetaMINIMAX(int** board, int depth, int player, int alpha, int beta, step& next)
{
    if( depth == 0 || gameover(board) != 0 )
    {
        return evaluate(board);
    }
//===================================== MAX ==========================================================
    if(player == MAX)
    {
        for(int i = 0; i < 15; i++)
        {
            for(int j = 0; j < 15; j++)
            {
                if(board[i][j] == 0)
                {
                    board[i][j] = 1;
                    int score = AlphaBetaMINIMAX(board, depth - 1, player^1, alpha, beta) + startup[i][j];
                    board[i][j] = 0;
                    if(score > alpha)
                    {
                        alpha = score;
                        next.x = i;
                        next.y = j;
                    }
                    if(alpha >= beta)    // 剪枝，其父必不选
                    {
                        return alpha;
                    }
                }
            }
        }
        return alpha;
    }
//===================================== MIN ==========================================================
    else
    {
        for(int i = 0; i < 15; i++)
        {
            for(int j = 0; j < 15; j++)
            {
                if(board[i][j] == 0)
                {
                    board[i][j] = -1;
                    int score = AlphaBetaMINIMAX(board, depth - 1, player^1, alpha, beta) + startup[i][j];
                    board[i][j] = 0;
                    if(score < beta)
                    {
                        beta = score;
                        next.x = i;
                        next.y = j;
                    }
                    if(alpha >= beta)    // 剪枝，其父必不选
                    {
                        return beta;
                    }
                }
            }
        }
        return beta;
    }
//======================================================================================================
}
void printBoard(int**board)
{
    for(int i = 0; i < 15; i++)
    {
        for(int j = 0; j < 15; j++)
        {
            printf("%2.d ", board[i][j]);
        }
        putchar('\n');
    }
}