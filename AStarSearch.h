struct grid
{
    int obstacle;
    int x;
    int y;
    int f;
    int g;
    int h;
    int enterDirection;
    //-------------------------------------------------------
    friend bool operator <(const grid &a, const grid &b)
    {
        return a.f > b.f;
    }
};
typedef struct grid grid;
//==================================================================================================
grid** buildGraph(int height, int width, char* inputFile)
{   
    grid** graph = (grid**)malloc(sizeof(grid) * (height + 2) * (width + 2));
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
            int temp;
            fscanf(inputf,"%d",&temp);
            fgetc(inputf);
            graph[i][j].obstacle = temp;
            graph[i][j].x = i;
            graph[i][j].y = j;
            graph[i][j].g = __INT_MAX__;
        }
    }
    return graph;
}
//==================================================================================================
#include <queue>
#include <cmath>
char direction[4] = {'U','D','L','R'};
std::vector<char> AStarSearch(grid** graph, int height, int width, int beginX, int beginY, int endX, int endY)
{
    graph[beginX][beginY].g = 0;
    for(int i = 1; i <= height; i++)
    {
        for(int j = 1; j <= width; j++)
        {
            graph[i][j].h = abs(graph[i][j].x - graph[endX][endY].x) + abs(graph[i][j].y - graph[endX][endY].y);
        }
    }
    graph[beginX][beginY].f = graph[beginX][beginY].g + graph[beginX][beginY].h;

    std::priority_queue<grid> expandCandidate;
    expandCandidate.push(graph[beginX][beginY]);

    while(true){
        grid searchingGrid = expandCandidate.pop();
        if(searchingGrid.x == endX && searchingGrid.y == endY)
        {
            break;
        }
        int newCost;
        
    }
    std::vector<char> resultPath;
    return resultPath;
}