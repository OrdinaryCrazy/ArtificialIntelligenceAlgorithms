#include <stdio.h>
#include <vector>
#include <sys/time.h>

#include "AStarSearch.h"

#define GRID_HEIGHT 18
#define GRID_WIDTH  25
#define BEGIN_X     1 + 1
#define BEGIN_Y     0 + 1
#define END_X       16 + 1
#define END_Y       24 + 1

int main(int argc,char* argv[])
{
    grid** graph = buildGraph(GRID_HEIGHT, GRID_WIDTH, argv[1]);

    struct timeval beginTime, endTime;
    gettimeofday(&beginTime, NULL);
    std::vector<char> expandPath = AStarSearch(graph, BEGIN_X, BEGIN_Y, END_X, END_Y);
    gettimeofday(&endTime, NULL);

    FILE* outputFile = fopen("./output_A.txt","w");
    fprintf(outputFile, "%ld\n",(endTime.tv_sec - beginTime.tv_sec) + (endTime.tv_usec - beginTime.tv_usec)/1e-6);
    for(int i = 0; i < expandPath.size(); i++)
    {
        fputc(expandPath[i], outputFile);
    }
    fputc("\n", outputFile);
    fprintf(outputFile, "%d",expandPath.size());

    return 0;
}