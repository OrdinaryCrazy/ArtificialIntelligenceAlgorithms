#include <cstdlib>
// #define DEBUG
struct grid
{
    int obstacle;       // 记录节点是否可以通行
    int x;
    int y;              // 记录节点坐标，方便优先队列使用
    int f;
    int g;              // 已知最小g值，初始化为无穷大
    int h;              // 启发函数值，可以在一开始就计算得到
    int enterDirection; // 记录上一个节点是按哪个方向移动进入当前节点，用于最后输出最优路径
    //-------------------------------------------------------
    // 优先队列优先级比较算符定义
    friend bool operator <(const grid &a, const grid &b)
    {
        return a.f > b.f;
    }
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
            graph[i][j].x = i;
            graph[i][j].y = j;
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
#define UP      0
#define DOWN    1
#define LEFT    2
#define RIGHT   3
char direction[4] = {'U','D','L','R'};
std::vector<char> AStarSearch(grid** graph, int height, int width, int beginX, int beginY, int endX, int endY)
{
    graph[beginX][beginY].g = 0;
    for(int i = 1; i <= height; i++)
    {
        for(int j = 1; j <= width; j++)
        {
            graph[i][j].h = abs(i - endX) + abs(j - endY);
        }
    }
    graph[beginX][beginY].f = graph[beginX][beginY].g + graph[beginX][beginY].h;
//--------------------------------------------------------------------------------------------------------------
    std::priority_queue<grid> expandCandidate;
    expandCandidate.push(graph[beginX][beginY]);

    while(true){
        grid searchingGrid = expandCandidate.top();
        expandCandidate.pop();
        //-------------------------------------------------
        #ifdef DEBUG
            printf("Searching [%d,%d] from %c\n",searchingGrid.x,searchingGrid.y,direction[searchingGrid.enterDirection]);
        #endif
        //-------------------------------------------------
        if(searchingGrid.x == endX && searchingGrid.y == endY)
        {
            //-------------------------------------------------
            #ifdef DEBUG
                printf("Found\n");
            #endif
            //-------------------------------------------------
            break;
        }
        int newg = graph[searchingGrid.x][searchingGrid.y].g + 1;
        // Up
        if( graph[searchingGrid.x - 1][searchingGrid.y].obstacle == 0 && graph[searchingGrid.x - 1][searchingGrid.y].g > newg )
        {
            graph[searchingGrid.x - 1][searchingGrid.y].g = newg;
            graph[searchingGrid.x - 1][searchingGrid.y].f = newg + graph[searchingGrid.x - 1][searchingGrid.y].h;
            graph[searchingGrid.x - 1][searchingGrid.y].enterDirection = UP;
            expandCandidate.push(graph[searchingGrid.x - 1][searchingGrid.y]);
        }
        // Left
        if( graph[searchingGrid.x][searchingGrid.y - 1].obstacle == 0 && graph[searchingGrid.x][searchingGrid.y - 1].g > newg )
        {
            graph[searchingGrid.x][searchingGrid.y - 1].g = newg;
            graph[searchingGrid.x][searchingGrid.y - 1].f = newg + graph[searchingGrid.x][searchingGrid.y - 1].h;
            graph[searchingGrid.x][searchingGrid.y - 1].enterDirection = LEFT;
            expandCandidate.push(graph[searchingGrid.x][searchingGrid.y - 1]);
        }
        // Right
        if( graph[searchingGrid.x][searchingGrid.y + 1].obstacle == 0 && graph[searchingGrid.x][searchingGrid.y + 1].g > newg )
        {
            graph[searchingGrid.x][searchingGrid.y + 1].g = newg;
            graph[searchingGrid.x][searchingGrid.y + 1].f = newg + graph[searchingGrid.x][searchingGrid.y + 1].h;
            graph[searchingGrid.x][searchingGrid.y + 1].enterDirection = RIGHT;
            expandCandidate.push(graph[searchingGrid.x][searchingGrid.y + 1]);
        }
        // Down
        if( graph[searchingGrid.x + 1][searchingGrid.y].obstacle == 0 && graph[searchingGrid.x + 1][searchingGrid.y].g > newg )
        {
            graph[searchingGrid.x + 1][searchingGrid.y].g = newg;
            graph[searchingGrid.x + 1][searchingGrid.y].f = newg + graph[searchingGrid.x + 1][searchingGrid.y].h;
            graph[searchingGrid.x + 1][searchingGrid.y].enterDirection = DOWN;
            expandCandidate.push(graph[searchingGrid.x + 1][searchingGrid.y]);
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