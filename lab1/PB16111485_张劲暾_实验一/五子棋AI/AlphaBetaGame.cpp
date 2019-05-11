#include <cstdio>
#include <cstdlib>

#include "AlphaBetaGame.h"

#define GRAPH
// 电脑-1 人1
int main(int argc, char* argv[])
{
    FILE* outputFile = fopen("output.txt","w");
    fprintf(outputFile,"AI\t\t\tME\n");
    int** board = (int**)malloc(sizeof(int*) * 15);
    for(int i = 0; i < 15; i++)
    {
        board[i] = (int*)malloc(sizeof(int) * 15);
    }
    for(int i = 0; i < 15; i++)
    {
        for(int j = 0; j < 15; j++)
        {
            board[i][j] = 0;
        }
    }
    while (true)
    {
        step next;
        //============================ AGENT ======================================
        #ifdef DEBUG
            printf("%d\n",AlphaBetaMINIMAX(board, DEPTH, MIN, (-1)*__INT_MAX__, __INT_MAX__, next));
        #else
            AlphaBetaMINIMAX(board, DEPTH, MIN, (-1)*__INT_MAX__, __INT_MAX__, next);
        #endif
        fprintf(outputFile,"[%2.d, %2.d]\t", next.x, next.y);

        printf("%d %d\n", next.x, next.y);
        fflush(stdout);
        board[next.x][next.y] = -1;
        #ifndef GRAPH
            printBoard(board);
        #endif
        if(gameover(board) == -1)
        {
            printf("AI Win\n");
            fflush(stdout);
            fprintf(outputFile,"\n\nAI WIN!");
            break;
        }
        next.x = next.y = 0;
        //============================ HUMAN ======================================
        scanf("%d %d", &next.x, &next.y);
        fprintf(outputFile,"[%2.d, %2.d]\n", next.x, next.y);
        board[next.x][next.y] = 1;
        #ifndef GRAPH
            printBoard(board);
        #endif
        if(gameover(board) == 1)
        {
            printf("AI Lose\n");
            fflush(stdout);
            fprintf(outputFile,"\nHUMAN WIN!");
            break;
        }
        next.x = next.y = 0;
        //=========================================================================
    }
    for(int i = 0; i < 15; i++)
    {
        free(board[i]);
    }
    free(board);
    fclose(outputFile);
    return 0;
}