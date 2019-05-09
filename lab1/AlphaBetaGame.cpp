#include <cstdio>
#include <cstdlib>

#include "AlphaBetaGame.h"

#define DEPTH   2
// 电脑-1 人1
int main(int argc, char* argv[])
{
    FILE* outputFile = fopen("output.txt","w");
    fprintf(outputFile,"AI\tME\n");
    int board[15][15] = {0};
    while (true)
    {
        step next;
        //============================ AGENT ======================================
        AlphaBetaMINIMAX(board, DEPTH, MIN, (-1)*__INT_MAX__, __INT_MAX__, next);
        fprintf(outputFile,"[%2.d, %2.d]\t", next.x, next.y);
        board[next.x][next.y] = -1;
        printBoard(board);
        if(gameover(board) == -1)
        {
            fprintf(outputFile,"\nAI WIN!");
            break;
        }
        //============================ HUMAN ======================================
        scanf("%d,%d", &next.x, &next.y);
        fprintf(outputFile,"[%2.d, %2.d]\n", next.x, next.y);
        board[next.x][next.y] = 1;
        printBoard(board);
        if(gameover(board) == 1)
        {
            fprintf(outputFile,"\nHUMAN WIN!");
            break;
        }
        //=========================================================================
    }
    fclose(outputFile);
    return 0;
}