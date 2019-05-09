#include <cstdlib>
// #define DEBUG
struct grid
{
    int obstacle;
    int h;
    int g;
    int enterDirection;
    int inPath;
};
typedef struct grid grid;
//==================================================================================================
grid** buildGraph(int height, int width, char* inputFile)
{   
    grid** graph = (grid**)malloc(sizeof(grid*) * (height + 2));
    for(int i = 0; i < height + 2; i++)
    {
        graph[i] = (grid*)malloc(sizeof(grid) * (width + 2));
    }
    for(int i = 0; i < width + 2; i++)
    {
        graph[0][i].obstacle = graph[height + 1][i].obstacle = 1;
    }
    for(int i = 0; i < height + 2; i++)
    {
        graph[i][0].obstacle = graph[i][width + 1].obstacle = 1;
    }
    FILE* inputf = fopen(inputFile, "r");
    for(int i = 1; i <= height; i++)
    {
        for(int j = 1; j <= width; j++)
        {
            fscanf(inputf,"%d",&graph[i][j].obstacle);
            fgetc(inputf);
            graph[i][j].inPath = 0;
            graph[i][j].g = __INT_MAX__;
        }
    }
    fclose(inputf);
    //-------------------------------------------------
    #ifdef DEBUG 
        for(int i = 0; i < height + 2; i++)
        {
            for(int j = 0; j < width + 2; j++)
            {
                printf("%d ",graph[i][j].obstacle);
            }
            printf("\n");
        }
    #endif
    //-------------------------------------------------
    return graph;
}
//==================================================================================================
#include <queue>
#include <cmath>
#define GET     -1
#define UP      0
#define DOWN    1
#define LEFT    2
#define RIGHT   3
char direction[4] = {'U','D','L','R'};
int iterativeDeepeningSearch(grid** graph, int curX, int curY, int g, int bound, int endX, int endY)
{
    #ifdef DEBUG
        for(int i = 0; i < 30 + 2; i++)
        {
            for(int j = 0; j < 60 + 2; j++)
            {
                if(graph[i][j].obstacle){printf(".. ");}
                else printf("%2.d ",graph[i][j].g % 200);
            }
            printf("\n");
        }
        getchar();
    #endif
    int f = g + graph[curX][curY].h;
    if(f > bound)
    {
        #ifdef DEBUG
            printf("EXCEEDING %d\n",f);
        #endif
        return f;
    }
    if(curX == endX && curY == endY)
    {
        return GET;
    }
    int min = __INT_MAX__;

    if( graph[curX][curY + 1].obstacle == 0 && 
        graph[curX][curY + 1].inPath == 0   &&
        graph[curX][curY + 1].g > g + 1
        )
    {
        graph[curX][curY + 1].g = g + 1;
        graph[curX][curY + 1].inPath = 1;
        graph[curX][curY + 1].enterDirection = RIGHT;
        f = iterativeDeepeningSearch(graph, curX, curY + 1, g + 1, bound, endX, endY);
        if(f == GET)
        {
            return GET;
        }
        min > f ? min = f : 0;
        graph[curX][curY + 1].inPath = 0;
    }
    if( graph[curX + 1][curY].obstacle == 0 && 
        graph[curX + 1][curY].inPath == 0   &&
        graph[curX + 1][curY].g > g + 1
        )
    {
        graph[curX + 1][curY].g = g + 1;
        graph[curX + 1][curY].inPath = 1;
        graph[curX + 1][curY].enterDirection = DOWN;
        f = iterativeDeepeningSearch(graph, curX + 1, curY, g + 1, bound, endX, endY);
        if(f == GET)
        {
            return GET;
        }
        min > f ? min = f : 0;
        graph[curX + 1][curY].inPath = 0;
    }
    if( graph[curX - 1][curY].obstacle == 0 && 
        graph[curX - 1][curY].inPath == 0   &&
        graph[curX - 1][curY].g > g + 1
        )
    {
        graph[curX - 1][curY].g = g + 1;
        graph[curX - 1][curY].inPath = 1;
        graph[curX - 1][curY].enterDirection = UP;
        f = iterativeDeepeningSearch(graph, curX - 1, curY, g + 1, bound, endX, endY);
        if(f == GET)
        {
            return GET;
        }
        min > f ? min = f : 0;
        graph[curX - 1][curY].inPath = 0;
    }
    if( graph[curX][curY - 1].obstacle == 0 && 
        graph[curX][curY - 1].inPath == 0   &&
        graph[curX][curY - 1].g > g + 1
        )
    {
        graph[curX][curY - 1].g = g + 1;
        graph[curX][curY - 1].inPath = 1;
        graph[curX][curY - 1].enterDirection = LEFT;
        f = iterativeDeepeningSearch(graph, curX, curY - 1, g + 1, bound, endX, endY);
        if(f == GET)
        {
            return GET;
        }
        min > f ? min = f : 0;
        graph[curX][curY - 1].inPath = 0;
    }
    return min;
}
std::vector<char> IDAStarSearch(grid** graph, int height, int width, int beginX, int beginY, int endX, int endY)
{
    for(int i = 1; i <= height; i++)
    {
        for(int j = 1; j <= width; j++)
        {
            graph[i][j].h = abs(i - endX) + abs(j - endY);
        }
    }
//--------------------------------------------------------------------------------------------------------------
    int bound = graph[beginX][beginY].h;
    graph[beginX][beginY].inPath = 1;
    graph[beginX][beginY].g = 0;
    while(true){
        int temp = iterativeDeepeningSearch(graph, beginX, beginY, 0, bound, endX, endY);
        if(temp == GET)
        {
            break;
        }
        //-------------------------------------------------
        #ifdef DEBUG 
            printf("old bound %d, new bound %d\n",bound, temp);
        #endif
        //-------------------------------------------------
        bound = temp;
        for(int i = 1; i <= height; i++)
        {
            for(int j = 1; j <= width; j++)
            {
                graph[i][j].g = __INT_MAX__;
            }
        }
    }
//--------------------------------------------------------------------------------------------------------------
    std::vector<char> resultPath;
    int tempX = endX, tempY = endY;
    while( !( tempX == beginX && tempY == beginY ) )
    {
        //-------------------------------------------------
        #ifdef DEBUG
            printf("Back going [%d,%d] to [%d,%d]\n",tempX,tempY,beginX,beginY);
            getchar();
        #endif
        //-------------------------------------------------
        resultPath.push_back( direction[ graph[tempX][tempY].enterDirection ] );
        switch (graph[tempX][tempY].enterDirection)
        {
            case UP:    tempX += 1; break;
            case DOWN:  tempX -= 1; break;
            case LEFT:  tempY += 1; break;
            case RIGHT: tempY -= 1; break;
        }
    }
    for(int i = 0; i < height + 2; i++)
    {
        free(graph[i]);
    }
    free(graph);
    return resultPath;
}