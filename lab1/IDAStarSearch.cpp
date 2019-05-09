#include <stdio.h>
#include <vector>
#include <sys/time.h>

#include "IDAStarSearch.h"

#define GRID_HEIGHT 30 // 18
#define GRID_WIDTH  60 // 25 
#define BEGIN_X     1 + 1
#define BEGIN_Y     0 + 1
#define END_X       28 + 1 // 16 + 1
#define END_Y       59 + 1 // 24 + 1

int main(int argc,char* argv[])
{
    grid** graph = buildGraph(GRID_HEIGHT, GRID_WIDTH, argv[1]);

    struct timeval beginTime, endTime;
    gettimeofday(&beginTime, NULL);
    std::vector<char> expandPath = IDAStarSearch(graph, GRID_HEIGHT, GRID_WIDTH, BEGIN_X, BEGIN_Y, END_X, END_Y);
    gettimeofday(&endTime, NULL);

    FILE* outputFile = fopen("./output_IDA.txt","w");
    fprintf(outputFile, "%lf\n",(endTime.tv_sec - beginTime.tv_sec) + (endTime.tv_usec - beginTime.tv_usec)/1e6);
    for(int i = expandPath.size() - 1; i >= 0 ; i--)
    {
        fputc(expandPath[i], outputFile);
    }
    fputc('\n', outputFile);
    fprintf(outputFile, "%ld",(long)(expandPath.size()));
    fclose(outputFile);
    return 0;
}