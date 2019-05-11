struct typeCount
{
    int maxC;
    int minC;
};
typedef struct typeCount typeCount;
//---------------------------------------------------------------
// 棋型：活四 011110
typeCount PerfectFour(int** board)
{
    typeCount result;
    result.maxC = result.minC = 0;
    // UP-DOWN----------------------------------------------------
    for(int i = 0; i < 10; i++)
    {
        for(int j = 0; j < 15; j++)
        {
            if( board[i    ][j] == 0                &&
                board[i + 1][j] != 0                &&
                board[i + 1][j] == board[i + 2][j]  &&  
                board[i + 2][j] == board[i + 3][j]  &&
                board[i + 3][j] == board[i + 4][j]  &&
                board[i + 5][j] == 0  
                )
            {   // 011110
                if(board[i + 1][j] == 1)    result.maxC++;
                else                        result.minC++;
            }
        }
    }
    // RIGHT-LEFT----------------------------------------------------
    for(int i = 0; i < 15; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            if( board[i][j    ] == 0                &&
                board[i][j + 1] != 0                &&
                board[i][j + 1] == board[i][j + 2]  &&  
                board[i][j + 2] == board[i][j + 3]  &&
                board[i][j + 3] == board[i][j + 4]  &&
                board[i][j + 5] == 0  
                )
            {
                if(board[i][j + 1] == 1)    result.maxC++;
                else                        result.minC++;
            }
        }
    }
    // UL-DR----------------------------------------------------
    for(int i = 0; i < 10; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            if( board[i    ][j    ] == 0                    &&  
                board[i + 1][j + 1] != 0                    && 
                board[i + 1][j + 1] == board[i + 2][j + 2]  &&  
                board[i + 2][j + 2] == board[i + 3][j + 3]  &&
                board[i + 3][j + 3] == board[i + 4][j + 4]  &&
                board[i + 5][j + 5] == 0  
                )
            {
                if(board[i + 1][j + 1] == 1)    result.maxC++;
                else                            result.minC++;
            }
        }
    }
    // UR-DL----------------------------------------------------
    for(int i = 5; i < 15; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            if( board[i    ][j    ] == 0                    &&  
                board[i - 1][j + 1] != 0                    && 
                board[i - 1][j + 1] == board[i - 2][j + 2]  &&  
                board[i - 2][j + 2] == board[i - 3][j + 3]  &&
                board[i - 3][j + 3] == board[i - 4][j + 4]  &&
                board[i - 5][j + 5] == 0  
                )
            {
                if(board[i - 1][j + 1] == 1)    result.maxC++;
                else                            result.minC++;
            }
        }
    }
    //----------------------------------------------------
    return result;
}

// 棋型：冲四 011112 or 10111 or 11011 or 11101 or 2111110
typeCount ThreatFour(int** board)
{
    typeCount result;
    result.maxC = result.minC = 0;
    // UP-DOWN----------------------------------------------------
    for(int i = 0; i < 10; i++)
    {
        for(int j = 0; j < 15; j++)
        {
            if( board[i    ][j] == 0                    &&  
                board[i + 1][j] != 0                    &&  
                board[i + 1][j] == board[i + 2][j]      &&  
                board[i + 2][j] == board[i + 3][j]      &&
                board[i + 3][j] == board[i + 4][j]      &&
                board[i + 5][j] == (-1)*board[i + 1][j] 
                )
            {   // 011112
                if(board[i + 1][j] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i    ][j] == (-1)*board[i + 1][j] &&  
                board[i + 1][j] != 0                    &&  
                board[i + 1][j] == board[i + 2][j]      &&  
                board[i + 2][j] == board[i + 3][j]      &&
                board[i + 3][j] == board[i + 4][j]      &&
                board[i + 5][j] == 0 
                )
            {   // 2111110
                if(board[i + 1][j] == 1)    result.maxC++;
                else                        result.minC++;
            }
        }
    }
    for(int i = 0; i < 11; i++)
    {
        for(int j = 0; j < 15; j++)
        {
            if( board[i    ][j] != 0                    &&  
                board[i    ][j] == board[i + 2][j]      &&  
                board[i + 1][j] == 0                    &&
                board[i + 2][j] == board[i + 3][j]      &&  
                board[i + 3][j] == board[i + 4][j]      
                )
            {   // 10111
                if(board[i + 2][j] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i    ][j] != 0                    &&
                board[i    ][j] == board[i + 1][j]      &&  
                board[i + 1][j] == board[i + 3][j]      &&  
                board[i + 2][j] == 0                    &&
                board[i + 3][j] == board[i + 4][j]     
                )
            {   // 11011
                if(board[i + 1][j] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i    ][j] != 0                    &&
                board[i    ][j] == board[i + 1][j]      &&  
                board[i + 1][j] == board[i + 2][j]      &&  
                board[i + 2][j] == board[i + 4][j]      &&
                board[i + 3][j] == 0      
                )
            {   // 11101
                if(board[i + 1][j] == 1)    result.maxC++;
                else                        result.minC++;
            }
        }
    }
    // RIGHT-LEFT----------------------------------------------------
    for(int i = 0; i < 15; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            if( board[i][j    ] == 0                    &&  
                board[i][j + 1] != 0                    &&  
                board[i][j + 1] == board[i][j + 2]      &&  
                board[i][j + 2] == board[i][j + 3]      &&
                board[i][j + 3] == board[i][j + 4]      &&
                board[i][j + 5] == (-1)*board[i][j + 1] 
                )
            {   // 011112
                if(board[i][j + 1] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i][j    ] == (-1)*board[i][j + 1] &&  
                board[i][j + 1] != 0                    &&  
                board[i][j + 1] == board[i][j + 2]      &&  
                board[i][j + 2] == board[i][j + 3]      &&
                board[i][j + 3] == board[i][j + 4]      &&
                board[i][j + 5] == 0 
                )
            {   // 2111110
                if(board[i][j + 1] == 1)    result.maxC++;
                else                        result.minC++;
            }
        }
    }
    for(int i = 0; i < 15; i++)
    {
        for(int j = 0; j < 11; j++)
        {
            if( board[i][j    ] != 0                    &&  
                board[i][j    ] == board[i][j + 2]      &&  
                board[i][j + 1] == 0                    &&
                board[i][j + 2] == board[i][j + 3]      &&  
                board[i][j + 3] == board[i][j + 4]      
                )
            {   // 10111
                if(board[i][j + 2] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i][j    ] == board[i][j + 1]      &&
                board[i][j + 1] != 0                    &&  
                board[i][j + 1] == board[i][j + 3]      &&  
                board[i][j + 2] == 0                    &&
                board[i][j + 3] == board[i][j + 4]     
                )
            {   // 11011
                if(board[i][j + 1] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i][j    ] == board[i][j + 1]      &&  
                board[i][j + 1] != 0                    &&  
                board[i][j + 1] == board[i][j + 2]      &&  
                board[i][j + 2] == board[i][j + 4]      &&
                board[i][j + 3] == 0      
                )
            {   // 11101
                if(board[i][j + 1] == 1)    result.maxC++;
                else                        result.minC++;
            }
        }
    }
    // UL-DR----------------------------------------------------
    for(int i = 0; i < 10; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            if( board[i    ][j    ] == 0                        &&  
                board[i + 1][j + 1] != 0                        &&  
                board[i + 1][j + 1] == board[i + 2][j + 2]      &&  
                board[i + 2][j + 2] == board[i + 3][j + 3]      &&
                board[i + 3][j + 3] == board[i + 4][j + 4]      &&
                board[i + 5][j + 5] == (-1)*board[i + 1][j + 1] 
                )
            {   // 011112
                if(board[i + 1][j + 1] == 1)    result.maxC++;
                else                            result.minC++;
            }
            if( board[i    ][j    ] == (-1)*board[i + 1][j + 1] &&  
                board[i + 1][j + 1] != 0                        &&  
                board[i + 1][j + 1] == board[i + 2][j + 2]      &&  
                board[i + 2][j + 2] == board[i + 3][j + 3]      &&
                board[i + 3][j + 3] == board[i + 4][j + 4]      &&
                board[i + 5][j + 5] == 0 
                )
            {   // 2111110
                if(board[i + 1][j + 1] == 1)    result.maxC++;
                else                            result.minC++;
            }
        }
    }
    for(int i = 0; i < 11; i++)
    {
        for(int j = 0; j < 11; j++)
        {
            if( board[i    ][j    ] == board[i + 2][j + 2]      && 
                board[i    ][j    ] != 0                        && 
                board[i + 1][j + 1] == 0                        &&
                board[i + 2][j + 2] == board[i + 3][j + 3]      &&  
                board[i + 3][j + 3] == board[i + 4][j + 4]      
                )
            {   // 10111
                if(board[i + 2][j + 2] == 1)    result.maxC++;
                else                            result.minC++;
            }
            if( board[i    ][j    ] == board[i + 1][j + 1]      && 
                board[i + 1][j + 1] != 0                        && 
                board[i + 1][j + 1] == board[i + 3][j + 3]      && 
                board[i + 2][j + 2] == 0                        &&
                board[i + 3][j + 3] == board[i + 4][j + 4]     
                )
            {   // 11011
                if(board[i + 1][j + 1] == 1)    result.maxC++;
                else                            result.minC++;
            }
            if( board[i    ][j    ] == board[i + 1][j + 1]      &&  
                board[i + 1][j + 1] != 0                        && 
                board[i + 1][j + 1] == board[i + 2][j + 2]      && 
                board[i + 2][j + 2] == board[i + 4][j + 4]      &&
                board[i + 3][j + 3] == 0      
                )
            {   // 11101
                if(board[i + 1][j + 1] == 1)    result.maxC++;
                else                            result.minC++;
            }
        }
    }
    // UR-DL----------------------------------------------------
    for(int i = 5; i < 15; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            if( board[i    ][j    ] == 0                        &&  
                board[i - 1][j + 1] != 0                        &&  
                board[i - 1][j + 1] == board[i - 2][j + 2]      &&  
                board[i - 2][j + 2] == board[i - 3][j + 3]      &&
                board[i - 3][j + 3] == board[i - 4][j + 4]      &&
                board[i - 5][j + 5] == (-1)*board[i - 1][j + 1] 
                )
            {   // 011112
                if(board[i - 1][j + 1] == 1)    result.maxC++;
                else                            result.minC++;
            }
            if( board[i    ][j    ] == (-1)*board[i - 1][j + 1] &&  
                board[i - 1][j + 1] != 0                        &&  
                board[i - 1][j + 1] == board[i - 2][j + 2]      &&  
                board[i - 2][j + 2] == board[i - 3][j + 3]      &&
                board[i - 3][j + 3] == board[i - 4][j + 4]      &&
                board[i - 5][j + 5] == 0 
                )
            {   // 211110
                if(board[i - 1][j + 1] == 1)    result.maxC++;
                else                            result.minC++;
            }
        }
    }
    for(int i = 4; i < 15; i++)
    {
        for(int j = 0; j < 11; j++)
        {
            if( board[i    ][j    ] == board[i - 2][j + 2]      &&  
                board[i    ][j    ] != 0                        &&  
                board[i - 1][j + 1] == 0                        &&
                board[i - 2][j + 2] == board[i - 3][j + 3]      &&  
                board[i - 3][j + 3] == board[i - 4][j + 4]      
                )
            {   // 10111
                if(board[i - 2][j + 2] == 1)    result.maxC++;
                else                            result.minC++;
            }
            if( board[i    ][j    ] == board[i - 1][j + 1]      &&  
                board[i - 1][j + 1] != 0                        &&  
                board[i - 1][j + 1] == board[i - 3][j + 3]      &&  
                board[i - 2][j + 2] == 0                        &&
                board[i - 3][j + 3] == board[i - 4][j + 4]     
                )
            {   // 11011
                if(board[i - 1][j + 1] == 1)    result.maxC++;
                else                            result.minC++;
            }
            if( board[i    ][j    ] == board[i - 1][j + 1]      &&  
                board[i - 1][j + 1] != 0                        &&  
                board[i - 1][j + 1] == board[i - 2][j + 2]      &&  
                board[i - 2][j + 2] == board[i - 4][j + 4]      &&
                board[i - 3][j + 3] == 0      
                )
            {   // 11101
                if(board[i - 1][j + 1] == 1)    result.maxC++;
                else                            result.minC++;
            }
        }
    }
    //----------------------------------------------------
    return result;
}

// 棋型：活三 01110 or 010110 or 011010
typeCount ThreatThree(int** board)
{
    typeCount result;
    result.maxC = result.minC = 0;
    // UP-DOWN----------------------------------------------------
    for(int i = 0; i < 10; i++)
    {
        for(int j = 0; j < 15; j++)
        {
            if( board[i    ][j] == 0                    &&  
                board[i + 1][j] != 0                    &&  
                board[i + 1][j] == board[i + 3][j]      &&  
                board[i + 2][j] == 0                    &&
                board[i + 3][j] == board[i + 4][j]      &&
                board[i + 5][j] == 0
                )
            {   // 010110
                if(board[i + 1][j] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i    ][j] == 0                    &&  
                board[i + 1][j] != 0                    &&  
                board[i + 1][j] == board[i + 2][j]      &&  
                board[i + 2][j] == board[i + 4][j]      &&
                board[i + 3][j] == 0                    &&
                board[i + 5][j] == 0 
                )
            {   // 011010
                if(board[i + 1][j] == 1)    result.maxC++;
                else                        result.minC++;
            }
        }
    }
    for(int i = 0; i < 11; i++)
    {
        for(int j = 0; j < 15; j++)
        {
            if( board[i    ][j] == 0                    &&  
                board[i + 1][j] != 0                    &&
                board[i + 1][j] == board[i + 2][j]      &&
                board[i + 2][j] == board[i + 3][j]      &&  
                board[i + 4][j] == 0      
                )
            {   // 01110
                if(board[i + 2][j] == 1)    result.maxC++;
                else                        result.minC++;
            }
        }
    }
    // RIGHT-LEFT----------------------------------------------------
    for(int i = 0; i < 15; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            if( board[i][j    ] == 0                    &&  
                board[i][j + 1] != 0                    &&  
                board[i][j + 1] == board[i][j + 3]      &&  
                board[i][j + 2] == 0                    &&
                board[i][j + 3] == board[i][j + 4]      &&
                board[i][j + 5] == 0
                )
            {   // 010110
                if(board[i][j + 1] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i][j    ] == 0                    &&  
                board[i][j + 1] != 0                    &&  
                board[i][j + 1] == board[i][j + 2]      &&  
                board[i][j + 2] == board[i][j + 4]      &&
                board[i][j + 3] == 0                    &&
                board[i][j + 5] == 0 
                )
            {   // 011010
                if(board[i][j + 1] == 1)    result.maxC++;
                else                        result.minC++;
            }
        }
    }
    for(int i = 0; i < 15; i++)
    {
        for(int j = 0; j < 11; j++)
        {
            if( board[i][j    ] == 0                    &&  
                board[i][j + 1] != 0                    &&
                board[i][j + 1] == board[i][j + 2]      &&
                board[i][j + 2] == board[i][j + 3]      &&  
                board[i][j + 4] == 0      
                )
            {   // 01110
                if(board[i][j + 2] == 1)    result.maxC++;
                else                        result.minC++;
            }
        }
    }
    // UL-DR----------------------------------------------------
    for(int i = 0; i < 10; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            if( board[i    ][j    ] == 0                    &&  
                board[i + 1][j + 1] != 0                    &&  
                board[i + 1][j + 1] == board[i + 3][j + 3]  &&  
                board[i + 2][j + 2] == 0                    &&
                board[i + 3][j + 3] == board[i + 4][j + 4]  &&
                board[i + 5][j + 5] == 0
                )
            {   // 010110
                if(board[i + 1][j + 1] == 1)    result.maxC++;
                else                            result.minC++;
            }
            if( board[i    ][j    ] == 0                    &&  
                board[i + 1][j + 1] != 0                    &&  
                board[i + 1][j + 1] == board[i + 2][j + 2]  &&  
                board[i + 2][j + 2] == board[i + 4][j + 4]  &&
                board[i + 3][j + 3] == 0                    &&
                board[i + 5][j + 5] == 0 
                )
            {   // 011010
                if(board[i + 1][j + 1] == 1)    result.maxC++;
                else                            result.minC++;
            }
        }
    }
    for(int i = 0; i < 11; i++)
    {
        for(int j = 0; j < 11; j++)
        {
            if( board[i    ][j    ] == 0                    &&  
                board[i + 1][j + 1] != 0                    &&
                board[i + 1][j + 1] == board[i + 2][j + 2]  &&
                board[i + 2][j + 2] == board[i + 3][j + 3]  &&  
                board[i + 4][j + 4] == 0      
                )
            {   // 01110
                if(board[i + 2][j + 2] == 1)    result.maxC++;
                else                            result.minC++;
            }
        }
    }
    // UR-DL----------------------------------------------------
    for(int i = 5; i < 15; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            if( board[i    ][j    ] == 0                    &&  
                board[i - 1][j + 1] != 0                    &&  
                board[i - 1][j + 1] == board[i - 3][j + 3]  &&  
                board[i - 2][j + 2] == 0                    &&
                board[i - 3][j + 3] == board[i - 4][j + 4]  &&
                board[i - 5][j + 5] == 0
                )
            {   // 010110
                if(board[i - 1][j + 1] == 1)    result.maxC++;
                else                            result.minC++;
            }
            if( board[i    ][j    ] == 0                    &&  
                board[i - 1][j + 1] != 0                    &&  
                board[i - 1][j + 1] == board[i - 2][j + 2]  &&  
                board[i - 2][j + 2] == board[i - 4][j + 4]  &&
                board[i - 3][j + 3] == 0                    &&
                board[i - 5][j + 5] == 0 
                )
            {   // 011010
                if(board[i - 1][j + 1] == 1)    result.maxC++;
                else                            result.minC++;
            }
        }
    }
    for(int i = 4; i < 15; i++)
    {
        for(int j = 0; j < 11; j++)
        {
            if( board[i    ][j    ] == 0                    &&  
                board[i - 1][j + 1] != 0                    &&
                board[i - 1][j + 1] == board[i - 2][j + 2]  &&
                board[i - 2][j + 2] == board[i - 3][j + 3]  &&  
                board[i - 4][j + 4] == 0      
                )
            {   // 01110
                if(board[i - 2][j + 2] == 1)    result.maxC++;
                else                            result.minC++;
            }
        }
    }
    //----------------------------------------------------
    return result;
}

// 棋型：眠三 001112 or 211100 or 010112 or 011012 or 10011 or 11001 or 10101 or 2011102
typeCount TryThree(int** board)
{
    typeCount result;
    result.maxC = result.minC = 0;
    // UP-DOWN----------------------------------------------------
    for(int i = 0; i < 9; i++)
    {
        for(int j = 0; j < 15; j++)
        {
            if( board[i    ][j] == (-1)*board[i + 2][j] &&  
                board[i + 1][j] == 0                    &&  
                board[i + 2][j] != 0                    &&
                board[i + 2][j] == board[i + 3][j]      &&
                board[i + 3][j] == board[i + 4][j]      &&
                board[i + 5][j] == 0                    &&
                board[i + 6][j] == (-1)*board[i + 2][j] 
                )
            {   // 2011102
                if(board[i + 2][j] == 1)    result.maxC++;
                else                        result.minC++;
            }
        }
    }
    for(int i = 0; i < 10; i++)
    {
        for(int j = 0; j < 15; j++)
        {
            if( board[i    ][j] == 0                    &&  
                board[i + 1][j] == 0                    &&  
                board[i + 2][j] != 0                    &&
                board[i + 2][j] == board[i + 3][j]      &&
                board[i + 3][j] == board[i + 4][j]      &&
                board[i + 5][j] == (-1)*board[i + 2][j]
                )
            {   // 001112
                if(board[i + 2][j] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i    ][j] == (-1)*board[i + 2][j] &&  
                board[i + 1][j] != 0                    &&  
                board[i + 1][j] == board[i + 2][j]      &&  
                board[i + 2][j] == board[i + 3][j]      &&
                board[i + 4][j] == 0                    &&
                board[i + 5][j] == 0
                )
            {   // 211100
                if(board[i + 2][j] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i    ][j] == 0                    &&  
                board[i + 1][j] != 0                    &&  
                board[i + 1][j] == board[i + 3][j]      &&  
                board[i + 2][j] == 0                    &&
                board[i + 3][j] == board[i + 4][j]      &&
                board[i + 5][j] == (-1)*board[i + 1][j]
                )
            {   // 010112
                if(board[i + 1][j] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i    ][j] == 0                    &&  
                board[i + 1][j] != 0                    &&  
                board[i + 1][j] == board[i + 2][j]      &&  
                board[i + 2][j] == board[i + 4][j]      &&
                board[i + 3][j] == 0                    &&
                board[i + 5][j] == (-1)*board[i + 2][j]
                )
            {   // 011012
                if(board[i + 2][j] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i    ][j] == (-1)*board[i + 1][j] &&  
                board[i + 1][j] != 0                    &&  
                board[i + 1][j] == board[i + 3][j]      &&  
                board[i + 2][j] == 0                    &&
                board[i + 3][j] == board[i + 4][j]      &&
                board[i + 5][j] == 0
                )
            {   // 210110
                if(board[i + 1][j] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i    ][j] == (-1)*board[i + 2][j] &&  
                board[i + 1][j] != 0                    &&  
                board[i + 1][j] == board[i + 2][j]      &&  
                board[i + 2][j] == board[i + 4][j]      &&
                board[i + 3][j] == 0                    &&
                board[i + 5][j] == 0
                )
            {   // 211010
                if(board[i + 2][j] == 1)    result.maxC++;
                else                        result.minC++;
            }
        }
    }
    for(int i = 0; i < 11; i++)
    {
        for(int j = 0; j < 15; j++)
        {
            if( board[i    ][j] == board[i + 3][j]      &&  
                board[i    ][j] != 0                    &&  
                board[i + 1][j] == 0                    &&
                board[i + 2][j] == 0                    &&  
                board[i + 3][j] == board[i + 4][j]
                )
            {   // 10011
                if(board[i    ][j] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i    ][j] == board[i + 2][j]      &&  
                board[i    ][j] != 0                    &&  
                board[i + 1][j] == 0                    &&
                board[i + 2][j] == board[i + 4][j]      &&  
                board[i + 3][j] == 0      
                )
            {   // 10101
                if(board[i + 2][j] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i    ][j] == board[i + 1][j]      &&  
                board[i + 1][j] != board[i + 4][j]      &&
                board[i + 1][j] == board[i + 4][j]      &&
                board[i + 2][j] == 0                    &&  
                board[i + 3][j] == 0      
                )
            {   // 11001
                if(board[i + 1][j] == 1)    result.maxC++;
                else                        result.minC++;
            }
        }
    }
    // RIGHT-LEFT----------------------------------------------------
    for(int i = 0; i < 15; i++)
    {
        for(int j = 0; j < 9; j++)
        {
            if( board[i][j    ] == (-1)*board[i][j + 2] &&  
                board[i][j + 1] == 0                    &&  
                board[i][j + 2] != 0                    &&
                board[i][j + 2] == board[i][j + 3]      &&
                board[i][j + 3] == board[i][j + 4]      &&
                board[i][j + 5] == 0                    &&
                board[i][j + 6] == (-1)*board[i][j + 2] 
                )
            {   // 2011102
                if(board[i][j + 2] == 1)    result.maxC++;
                else                        result.minC++;
            }
        }
    }
    for(int i = 0; i < 15; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            if( board[i][j    ] == 0                    &&  
                board[i][j + 1] == 0                    &&  
                board[i][j + 2] != 0                    &&
                board[i][j + 2] == board[i][j + 3]      &&
                board[i][j + 3] == board[i][j + 4]      &&
                board[i][j + 5] == (-1)*board[i][j + 2]
                )
            {   // 001112
                if(board[i][j + 2] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i][j    ] == (-1)*board[i][j + 2] &&  
                board[i][j + 1] != 0                    &&  
                board[i][j + 1] == board[i][j + 2]      &&  
                board[i][j + 2] == board[i][j + 3]      &&
                board[i][j + 4] == 0                    &&
                board[i][j + 5] == 0
                )
            {   // 211100
                if(board[i][j + 2] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i][j    ] == 0                    &&  
                board[i][j + 1] != 0                    &&  
                board[i][j + 1] == board[i][j + 3]      &&  
                board[i][j + 2] == 0                    &&
                board[i][j + 3] == board[i][j + 4]      &&
                board[i][j + 5] == (-1)*board[i][j + 1]
                )
            {   // 010112
                if(board[i][j + 1] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i][j    ] == 0                    &&  
                board[i][j + 1] != 0                    &&  
                board[i][j + 1] == board[i][j + 2]      &&  
                board[i][j + 2] == board[i][j + 4]      &&
                board[i][j + 3] == 0                    &&
                board[i][j + 5] == (-1)*board[i][j + 2]
                )
            {   // 011012
                if(board[i][j + 2] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i][j    ] == (-1)*board[i][j + 1] &&  
                board[i][j + 1] != 0                    &&  
                board[i][j + 1] == board[i][j + 3]      &&  
                board[i][j + 2] == 0                    &&
                board[i][j + 3] == board[i][j + 4]      &&
                board[i][j + 5] == 0
                )
            {   // 210110
                if(board[i][j + 1] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i][j    ] == (-1)*board[i][j + 2] &&  
                board[i][j + 1] != 0                    &&  
                board[i][j + 1] == board[i][j + 2]      &&  
                board[i][j + 2] == board[i][j + 4]      &&
                board[i][j + 3] == 0                    &&
                board[i][j + 5] == 0
                )
            {   // 211010
                if(board[i][j + 2] == 1)    result.maxC++;
                else                        result.minC++;
            }
        }
    }
    for(int i = 0; i < 15; i++)
    {
        for(int j = 0; j < 11; j++)
        {
            if( board[i][j    ] == board[i][j + 3]      &&  
                board[i][j    ] != 0                    &&  
                board[i][j + 1] == 0                    &&
                board[i][j + 2] == 0                    &&  
                board[i][j + 3] == board[i][j + 4]
                )
            {   // 10011
                if(board[i    ][j] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i][j    ] == board[i][j + 2]      &&  
                board[i][j    ] != 0                    &&  
                board[i][j + 1] == 0                    &&
                board[i][j + 2] == board[i][j + 4]      &&  
                board[i][j + 3] == 0      
                )
            {   // 10101
                if(board[i][j + 2] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i][j    ] == board[i][j + 1]      &&  
                board[i][j + 1] != 0                    &&
                board[i][j + 1] == board[i][j + 4]      &&
                board[i][j + 2] == 0                    &&  
                board[i][j + 3] == 0      
                )
            {   // 11001
                if(board[i][j + 1] == 1)    result.maxC++;
                else                        result.minC++;
            }
        }
    }
    // UL-DR----------------------------------------------------
    for(int i = 0; i < 9; i++)
    {
        for(int j = 0; j < 9; j++)
        {
            if( board[i    ][j    ] == (-1)*board[i + 2][j + 2] &&  
                board[i + 1][j + 1] == 0                        &&  
                board[i + 2][j + 2] != 0                        &&
                board[i + 2][j + 2] == board[i + 3][j + 3]      &&
                board[i + 3][j + 3] == board[i + 4][j + 4]      &&
                board[i + 5][j + 5] == 0                        &&
                board[i + 6][j + 6] == (-1)*board[i + 2][j + 2] 
                )
            {   // 2011102
                if(board[i + 2][j + 2] == 1)    result.maxC++;
                else                            result.minC++;
            }
        }
    }
    for(int i = 0; i < 10; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            if( board[i    ][j    ] == 0                        &&  
                board[i + 1][j + 1] == 0                        &&  
                board[i + 2][j + 2] != 0                        &&
                board[i + 2][j + 2] == board[i + 3][j + 3]      &&
                board[i + 3][j + 3] == board[i + 4][j + 4]      &&
                board[i + 5][j + 5] == (-1)*board[i + 2][j + 2]
                )
            {   // 001112
                if(board[i + 2][j + 2] == 1)    result.maxC++;
                else                            result.minC++;
            }
            if( board[i    ][j    ] == (-1)*board[i + 2][j + 2] &&  
                board[i + 1][j + 1] != 0                        &&  
                board[i + 1][j + 1] == board[i + 2][j + 2]      &&  
                board[i + 2][j + 2] == board[i + 3][j + 3]      &&
                board[i + 4][j + 4] == 0                        &&
                board[i + 5][j + 2] == 0
                )
            {   // 211100
                if(board[i + 2][j + 2] == 1)    result.maxC++;
                else                            result.minC++;
            }
            if( board[i    ][j    ] == 0                        &&  
                board[i + 1][j + 1] != 0                        &&  
                board[i + 1][j + 1] == board[i + 3][j + 3]      &&  
                board[i + 2][j + 2] == 0                        &&
                board[i + 3][j + 3] == board[i + 4][j + 4]      &&
                board[i + 5][j + 5] == (-1)*board[i + 1][j + 1]
                )
            {   // 010112
                if(board[i + 1][j + 1] == 1)    result.maxC++;
                else                            result.minC++;
            }
            if( board[i    ][j    ] == 0                        &&  
                board[i + 1][j + 1] != 0                        &&  
                board[i + 1][j + 1] == board[i + 2][j + 2]      &&  
                board[i + 2][j + 2] == board[i + 4][j + 4]      &&
                board[i + 3][j + 3] == 0                        &&
                board[i + 5][j + 5] == (-1)*board[i + 2][j + 2]
                )
            {   // 011012
                if(board[i + 2][j + 2] == 1)    result.maxC++;
                else                            result.minC++;
            }
            if( board[i    ][j    ] == (-1)*board[i + 1][j + 1] &&  
                board[i + 1][j + 1] != 0                        &&  
                board[i + 1][j + 1] == board[i + 3][j + 3]      &&  
                board[i + 2][j + 2] == 0                        &&
                board[i + 3][j + 3] == board[i + 4][j + 4]      &&
                board[i + 5][j + 5] == 0
                )
            {   // 210110
                if(board[i + 1][j + 1] == 1)    result.maxC++;
                else                            result.minC++;
            }
            if( board[i    ][j    ] == (-1)*board[i + 2][j + 2] &&  
                board[i + 1][j + 1] != 0                        &&  
                board[i + 1][j + 1] == board[i + 2][j + 2]      &&  
                board[i + 2][j + 2] == board[i + 4][j + 4]      &&
                board[i + 3][j + 3] == 0                        &&
                board[i + 5][j + 5] == 0
                )
            {   // 211010
                if(board[i + 2][j + 2] == 1)    result.maxC++;
                else                            result.minC++;
            }
        }
    }
    for(int i = 0; i < 11; i++)
    {
        for(int j = 0; j < 11; j++)
        {
            if( board[i    ][j    ] == board[i + 3][j + 3]  &&  
                board[i    ][j    ] != 0                    &&  
                board[i + 1][j + 1] == 0                    &&
                board[i + 2][j + 2] == 0                    &&  
                board[i + 3][j + 3] == board[i + 4][j + 4]
                )
            {   // 10011
                if(board[i    ][j] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i    ][j    ] == board[i + 2][j + 2]  &&  
                board[i    ][j    ] != 0                    &&  
                board[i + 1][j + 1] == 0                    &&
                board[i + 2][j + 2] == board[i + 4][j + 4]  &&  
                board[i + 3][j + 3] == 0      
                )
            {   // 10101
                if(board[i + 2][j + 2] == 1)    result.maxC++;
                else                            result.minC++;
            }
            if( board[i    ][j    ] == board[i + 1][j + 1]  &&  
                board[i + 1][j + 1] != 0                    &&
                board[i + 1][j + 1] == board[i + 4][j + 4]  &&
                board[i + 2][j + 2] == 0                    &&  
                board[i + 3][j + 3] == 0      
                )
            {   // 11001
                if(board[i + 1][j + 1] == 1)    result.maxC++;
                else                            result.minC++;
            }
        }
    }
    // UR-DL----------------------------------------------------
    for(int i = 6; i < 15; i++)
    {
        for(int j = 0; j < 9; j++)
        {
            if( board[i    ][j    ] == (-1)*board[i - 2][j + 2] &&  
                board[i - 1][j + 1] == 0                        &&  
                board[i - 2][j + 2] != 0                        &&
                board[i - 2][j + 2] == board[i - 3][j + 3]      &&
                board[i - 3][j + 3] == board[i - 4][j + 4]      &&
                board[i - 5][j + 5] == 0                        &&
                board[i - 6][j + 6] == (-1)*board[i - 2][j + 2] 
                )
            {   // 2011102
                if(board[i - 2][j + 2] == 1)    result.maxC++;
                else                            result.minC++;
            }
        }
    }
    for(int i = 5; i < 15; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            if( board[i    ][j    ] == 0                        &&  
                board[i - 1][j + 1] == 0                        &&  
                board[i - 2][j + 2] != 0                        &&
                board[i - 2][j + 2] == board[i - 3][j + 3]      &&
                board[i - 3][j + 3] == board[i - 4][j + 4]      &&
                board[i - 5][j + 5] == (-1)*board[i - 2][j + 2]
                )
            {   // 001112
                if(board[i - 2][j + 2] == 1)    result.maxC++;
                else                            result.minC++;
            }
            if( board[i    ][j    ] == (-1)*board[i - 2][j + 2] &&  
                board[i - 1][j + 1] != 0                        &&  
                board[i - 1][j + 1] == board[i - 2][j + 2]      &&  
                board[i - 2][j + 2] == board[i - 3][j + 3]      &&
                board[i - 4][j + 4] == 0                        &&
                board[i - 5][j + 2] == 0
                )
            {   // 211100
                if(board[i - 2][j + 2] == 1)    result.maxC++;
                else                            result.minC++;
            }
            if( board[i    ][j    ] == 0                        &&  
                board[i - 1][j + 1] != 0                        &&  
                board[i - 1][j + 1] == board[i - 3][j + 3]      &&  
                board[i - 2][j + 2] == 0                        &&
                board[i - 3][j + 3] == board[i - 4][j + 4]      &&
                board[i - 5][j + 5] == (-1)*board[i - 1][j + 1]
                )
            {   // 010112
                if(board[i - 1][j + 1] == 1)    result.maxC++;
                else                            result.minC++;
            }
            if( board[i    ][j    ] == 0                        &&  
                board[i - 1][j + 1] != 0                        &&  
                board[i - 1][j + 1] == board[i - 2][j + 2]      &&  
                board[i - 2][j + 2] == board[i - 4][j + 4]      &&
                board[i - 3][j + 3] == 0                        &&
                board[i - 5][j + 5] == (-1)*board[i - 2][j + 2]
                )
            {   // 011012
                if(board[i - 2][j + 2] == 1)    result.maxC++;
                else                            result.minC++;
            }
            if( board[i    ][j    ] == (-1)*board[i - 1][j + 1] &&  
                board[i - 1][j + 1] != 0                        &&  
                board[i - 1][j + 1] == board[i - 3][j + 3]      &&  
                board[i - 2][j + 2] == 0                        &&
                board[i - 3][j + 3] == board[i - 4][j + 4]      &&
                board[i - 5][j + 5] == 0
                )
            {   // 210110
                if(board[i - 1][j + 1] == 1)    result.maxC++;
                else                            result.minC++;
            }
            if( board[i    ][j    ] == (-1)*board[i - 2][j + 2] &&  
                board[i - 1][j + 1] != 0                        &&  
                board[i - 1][j + 1] == board[i - 2][j + 2]      &&  
                board[i - 2][j + 2] == board[i - 4][j + 4]      &&
                board[i - 3][j + 3] == 0                        &&
                board[i - 5][j + 5] == 0
                )
            {   // 211010
                if(board[i - 2][j + 2] == 1)    result.maxC++;
                else                            result.minC++;
            }
        }
    }
    for(int i = 4; i < 15; i++)
    {
        for(int j = 0; j < 11; j++)
        {
            if( board[i    ][j    ] == board[i - 3][j + 3]  &&  
                board[i    ][j    ] != 0                    &&  
                board[i - 1][j + 1] == 0                    &&
                board[i - 2][j + 2] == 0                    &&  
                board[i - 3][j + 3] == board[i - 4][j + 4]
                )
            {   // 10011
                if(board[i    ][j] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i    ][j    ] == board[i - 2][j + 2]  &&  
                board[i    ][j    ] != 0                    &&  
                board[i - 1][j + 1] == 0                    &&
                board[i - 2][j + 2] == board[i - 4][j + 4]  &&  
                board[i - 3][j + 3] == 0      
                )
            {   // 10101
                if(board[i - 2][j + 2] == 1)    result.maxC++;
                else                            result.minC++;
            }
            if( board[i    ][j    ] == board[i - 1][j + 1]  &&  
                board[i - 1][j + 1] != 0                    &&
                board[i - 1][j + 1] == board[i - 4][j + 4]  &&
                board[i - 2][j + 2] == 0                    &&  
                board[i - 3][j + 3] == 0      
                )
            {   // 11001
                if(board[i - 1][j + 1] == 1)    result.maxC++;
                else                            result.minC++;
            }
        }
    }
    //----------------------------------------------------
    return result;
}

// 棋型：活二 00110 or 01100 or 01010 or 010010
typeCount GoodTwo(int** board)
{
    typeCount result;
    result.maxC = result.minC = 0;
    // UP-DOWN----------------------------------------------------
    for(int i = 0; i < 10; i++)
    {
        for(int j = 0; j < 15; j++)
        {
            if( board[i    ][j] == 0                &&  
                board[i + 5][j] == 0                &&
                board[i + 1][j] != 0                &&  
                board[i + 1][j] == board[i + 4][j]  &&  
                board[i + 2][j] == 0                &&
                board[i + 3][j] == 0  
                )
            {   // 010010
                if(board[i + 1][j] == 1)    result.maxC++;
                else                        result.minC++;
            }
        }
    }
    for(int i = 0; i < 11; i++)
    {
        for(int j = 0; j < 15; j++)
        {
            if( board[i    ][j] == 0                &&  
                board[i + 1][j] != 0                &&  
                board[i + 1][j] == board[i + 2][j]  &&  
                board[i + 4][j] == 0                &&
                board[i + 3][j] == 0  
                )
            {   // 01100
                if(board[i + 1][j] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i    ][j] == 0                &&  
                board[i + 1][j] == 0                &&  
                board[i + 2][j] != 0                &&
                board[i + 2][j] == board[i + 3][j]  &&
                board[i + 4][j] == 0  
                )
            {   // 00110
                if(board[i + 2][j] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i    ][j] == 0                &&  
                board[i + 1][j] != 0                &&  
                board[i + 1][j] == board[i + 3][j]  &&  
                board[i + 2][j] == 0                &&
                board[i + 4][j] == 0  
                )
            {   // 01010
                if(board[i + 1][j] == 1)    result.maxC++;
                else                        result.minC++;
            }
        }
    }
    // RIGHT-LEFT----------------------------------------------------
    for(int i = 0; i < 15; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            if( board[i][j    ] == 0                &&  
                board[i][j + 5] == 0                &&
                board[i][j + 1] != 0                &&  
                board[i][j + 1] == board[i][j + 4]  &&  
                board[i][j + 2] == 0                &&
                board[i][j + 3] == 0  
                )
            {   // 010010
                if(board[i][j + 1] == 1)    result.maxC++;
                else                        result.minC++;
            }
        }
    }
    for(int i = 0; i < 15; i++)
    {
        for(int j = 0; j < 11; j++)
        {
            if( board[i][j    ] == 0                &&  
                board[i][j + 1] != 0                &&  
                board[i][j + 1] == board[i][j + 2]  &&  
                board[i][j + 4] == 0                &&
                board[i][j + 3] == 0  
                )
            {   // 01100
                if(board[i][j + 1] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i][j    ] == 0                &&  
                board[i][j + 1] == 0                &&  
                board[i][j + 2] != 0                &&
                board[i][j + 2] == board[i][j + 3]  &&
                board[i][j + 4] == 0  
                )
            {   // 00110
                if(board[i][j + 2] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i][j    ] == 0                &&  
                board[i][j + 1] != 0                &&  
                board[i][j + 1] == board[i][j + 3]  &&  
                board[i][j + 2] == 0                &&
                board[i][j + 4] == 0  
                )
            {   // 01010
                if(board[i][j + 1] == 1)    result.maxC++;
                else                        result.minC++;
            }
        }
    }
    // UL-DR----------------------------------------------------
    for(int i = 0; i < 10; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            if( board[i    ][j    ] == 0                    &&  
                board[i + 5][j + 5] == 0                    &&
                board[i + 1][j + 1] != 0                    &&  
                board[i + 1][j + 1] == board[i + 4][j + 4]  &&  
                board[i + 2][j + 2] == 0                    &&
                board[i + 3][j + 3] == 0  
                )
            {   // 010010
                if(board[i + 1][j + 1] == 1)    result.maxC++;
                else                            result.minC++;
            }
        }
    }
    for(int i = 0; i < 11; i++)
    {
        for(int j = 0; j < 11; j++)
        {
            if( board[i    ][j    ] == 0                    &&  
                board[i + 1][j + 1] != 0                    &&  
                board[i + 1][j + 1] == board[i + 2][j + 2]  &&  
                board[i + 4][j + 4] == 0                    &&
                board[i + 3][j + 3] == 0  
                )
            {   // 01100
                if(board[i + 1][j + 1] == 1)    result.maxC++;
                else                            result.minC++;
            }
            if( board[i    ][j    ] == 0                    &&  
                board[i + 1][j + 1] == 0                    &&  
                board[i + 2][j + 2] != 0                    &&
                board[i + 2][j + 2] == board[i + 3][j + 3]  &&
                board[i + 4][j + 4] == 0  
                )
            {   // 00110
                if(board[i + 2][j + 2] == 1)    result.maxC++;
                else                            result.minC++;
            }
            if( board[i    ][j    ] == 0                    &&  
                board[i + 1][j + 1] != 0                    &&  
                board[i + 1][j + 1] == board[i + 3][j + 3]  &&  
                board[i + 2][j + 2] == 0                    &&
                board[i + 4][j + 4] == 0  
                )
            {   // 01010
                if(board[i + 1][j + 1] == 1)    result.maxC++;
                else                            result.minC++;
            }
        }
    }
    // UR-DL----------------------------------------------------
    for(int i = 5; i < 15; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            if( board[i    ][j    ] == 0                    &&  
                board[i - 5][j + 5] == 0                    &&
                board[i - 1][j + 1] != 0                    &&  
                board[i - 1][j + 1] == board[i - 4][j + 4]  &&  
                board[i - 2][j + 2] == 0                    &&
                board[i - 3][j + 3] == 0  
                )
            {   // 010010
                if(board[i - 1][j + 1] == 1)    result.maxC++;
                else                            result.minC++;
            }
        }
    }
    for(int i = 4; i < 15; i++)
    {
        for(int j = 0; j < 11; j++)
        {
            if( board[i    ][j    ] == 0                    &&  
                board[i - 1][j + 1] != 0                    &&  
                board[i - 1][j + 1] == board[i - 2][j + 2]  &&  
                board[i - 4][j + 4] == 0                    &&
                board[i - 3][j + 3] == 0  
                )
            {   // 01100
                if(board[i - 1][j + 1] == 1)    result.maxC++;
                else                            result.minC++;
            }
            if( board[i    ][j    ] == 0                    &&  
                board[i - 1][j + 1] == 0                    &&  
                board[i - 2][j + 2] != 0                    &&
                board[i - 2][j + 2] == board[i - 3][j + 3]  &&
                board[i - 4][j + 4] == 0  
                )
            {   // 00110
                if(board[i - 2][j + 2] == 1)    result.maxC++;
                else                            result.minC++;
            }
            if( board[i    ][j    ] == 0                    &&  
                board[i - 1][j + 1] != 0                    &&  
                board[i - 1][j + 1] == board[i - 3][j + 3]  &&  
                board[i - 2][j + 2] == 0                    &&
                board[i - 4][j + 4] == 0  
                )
            {   // 01010
                if(board[i - 1][j + 1] == 1)    result.maxC++;
                else                            result.minC++;
            }
        }
    }
    //----------------------------------------------------
    return result;
}

// 棋型：眠二 000112 or 211000 or 010012 or 10001 or 2010102 or 2011002
typeCount LimitedTwo(int** board)
{
    typeCount result;
    result.maxC = result.minC = 0;
    // UP-DOWN----------------------------------------------------
    for(int i = 0; i < 9; i++)
    {
        for(int j = 0; j < 15; j++)
        {
            if( board[i    ][j] == board[i + 2][j]*(-1) &&
                board[i + 6][j] == board[i + 2][j]*(-1) &&  
                board[i + 5][j] == 0                    &&
                board[i + 1][j] == 0                    &&  
                board[i + 2][j] != 0                    &&
                board[i + 2][j] == board[i + 3][j]      &&
                board[i + 4][j] == 0 
                )
            {   // 2011002
                if(board[i + 2][j] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i    ][j] == board[i + 2][j]*(-1) &&
                board[i + 6][j] == board[i + 2][j]*(-1) &&  
                board[i + 5][j] == 0                    &&
                board[i + 1][j] == 0                    &&  
                board[i + 2][j] != 0                    &&
                board[i + 2][j] == board[i + 4][j]      &&
                board[i + 3][j] == 0 
                )
            {   // 2010102
                if(board[i + 2][j] == 1)    result.maxC++;
                else                        result.minC++;
            }
        }
    }
    for(int i = 0; i < 10; i++)
    {
        for(int j = 0; j < 15; j++)
        {
            if( board[i    ][j] == 0                    && 
                board[i + 5][j] == board[i + 3][j]*(-1) &&
                board[i + 1][j] == 0                    &&  
                board[i + 2][j] == 0                    &&
                board[i + 4][j] != 0                    &&
                board[i + 4][j] == board[i + 3][j]
                )
            {   // 000112
                if(board[i + 3][j] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i    ][j] == board[i + 2][j]*(-1) &&
                board[i + 5][j] == 0                    &&
                board[i + 4][j] == 0                    &&  
                board[i + 2][j] != 0                    &&
                board[i + 2][j] == board[i + 1][j]      &&
                board[i + 3][j] == 0 
                )
            {   // 211000
                if(board[i + 2][j] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i    ][j] == 0                    &&
                board[i + 5][j] == board[i + 4][j]*(-1) &&
                board[i + 1][j] != 0                    &&  
                board[i + 1][j] == board[i + 4][j]      &&  
                board[i + 2][j] == 0                    &&
                board[i + 3][j] == 0 
                )
            {   // 010012
                if(board[i + 1][j] == 1)    result.maxC++;
                else                        result.minC++;
            }
        }
    }
    for(int i = 0; i < 11; i++)
    {
        for(int j = 0; j < 15; j++)
        {
            if( board[i    ][j] == board[i + 4][j]  &&  
                board[i    ][j] != 0                &&  
                board[i + 1][j] == 0                &&  
                board[i + 2][j] == 0                &&
                board[i + 3][j] == 0 
                )
            {   // 10001
                if(board[i][j] == 1)    result.maxC++;
                else                    result.minC++;
            }
        }
    }
    // RIGHT-LEFT----------------------------------------------------
    for(int i = 0; i < 15; i++)
    {
        for(int j = 0; j < 9; j++)
        {
            if( board[i][j    ] == board[i][j + 2]*(-1) &&
                board[i][j + 6] == board[i][j + 2]*(-1) &&  
                board[i][j + 5] == 0                    &&
                board[i][j + 1] == 0                    &&  
                board[i][j + 2] != 0                    &&
                board[i][j + 2] == board[i][j + 3]      &&
                board[i][j + 4] == 0  
                )
            {   // 2011002
                if(board[i][j + 2] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i][j    ] == board[i][j + 2]*(-1) &&
                board[i][j + 6] == board[i][j + 2]*(-1) &&  
                board[i][j + 5] == 0                    &&
                board[i][j + 1] == 0                    &&  
                board[i][j + 2] != 0                    &&
                board[i][j + 2] == board[i][j + 4]      &&
                board[i][j + 3] == 0 
                )
            {   // 2010102
                if(board[i][j + 2] == 1)    result.maxC++;
                else                        result.minC++;
            }
        }
    }
    for(int i = 0; i < 15; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            if( board[i][j    ] == 0                    && 
                board[i][j + 5] == board[i][j + 3]*(-1) &&
                board[i][j + 1] == 0                    &&  
                board[i][j + 2] == 0                    &&
                board[i][j + 4] != 0                    &&
                board[i][j + 4] == board[i][j + 3]  
                )
            {   // 000112
                if(board[i][j + 3] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i][j    ] == board[i][j + 2]*(-1) &&
                board[i][j + 5] == 0                    &&
                board[i][j + 4] == 0                    &&  
                board[i][j + 2] != 0                    &&
                board[i][j + 2] == board[i][j + 1]      &&
                board[i][j + 3] == 0  
                )
            {   // 211000
                if(board[i][j + 2] == 1)    result.maxC++;
                else                        result.minC++;
            }
            if( board[i][j    ] == 0                    &&
                board[i][j + 5] == board[i][j + 4]*(-1) &&
                board[i][j + 1] != 0                    &&  
                board[i][j + 1] == board[i][j + 4]      &&  
                board[i][j + 2] == 0  &&
                board[i][j + 3] == 0  
                )
            {   // 010012
                if(board[i][j + 1] == 1)    result.maxC++;
                else                        result.minC++;
            }
        }
    }
    for(int i = 0; i < 15; i++)
    {
        for(int j = 0; j < 11; j++)
        {
            if( board[i][j    ] != 0                &&  
                board[i][j    ] == board[i][j + 4]  &&  
                board[i][j + 1] == 0                &&  
                board[i][j + 2] == 0                &&
                board[i][j + 3] == 0  
                )
            {   // 10001
                if(board[i][j] == 1)    result.maxC++;
                else                    result.minC++;
            }
        }
    }
    // UL-DR----------------------------------------------------
    for(int i = 0; i < 9; i++)
    {
        for(int j = 0; j < 9; j++)
        {
            if( board[i    ][j    ] == board[i + 2][j + 2]*(-1) &&
                board[i + 6][j + 6] == board[i + 2][j + 2]*(-1) &&  
                board[i + 5][j + 5] == 0                        &&
                board[i + 1][j + 1] == 0                        &&  
                board[i + 2][j + 2] != 0                        &&
                board[i + 2][j + 2] == board[i + 3][j + 3]      &&
                board[i + 4][j + 4] == 0  
                )
            {   // 2011002
                if(board[i + 2][j + 2] == 1)    result.maxC++;
                else                            result.minC++;
            }
            if( board[i    ][j    ] == board[i + 2][j + 2]*(-1) &&
                board[i + 6][j + 6] == board[i + 2][j + 2]*(-1) &&  
                board[i + 5][j + 5] == 0                        &&
                board[i + 1][j + 1] == 0                        &&  
                board[i + 2][j + 2] != 0                        &&
                board[i + 2][j + 2] == board[i + 4][j + 4]      &&
                board[i + 3][j + 3] == 0  
                )
            {   // 2010102
                if(board[i + 2][j + 2] == 1)    result.maxC++;
                else                            result.minC++;
            }
        }
    }
    for(int i = 0; i < 10; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            if( board[i    ][j    ] == 0                        && 
                board[i + 5][j + 5] == board[i + 3][j + 3]*(-1) &&
                board[i + 1][j + 1] == 0                        &&  
                board[i + 2][j + 2] == 0                        &&
                board[i + 4][j + 4] != 0                        &&
                board[i + 4][j + 4] == board[i + 3][j + 3]  
                )
            {   // 000112
                if(board[i + 3][j + 3] == 1)    result.maxC++;
                else                            result.minC++;
            }
            if( board[i    ][j    ] == board[i + 2][j + 2]*(-1) &&
                board[i + 5][j + 5] == 0                        &&
                board[i + 4][j + 4] == 0                        &&  
                board[i + 2][j + 2] != 0                        &&
                board[i + 2][j + 2] == board[i + 1][j + 1]      &&
                board[i + 3][j + 3] == 0  
                )
            {   // 211000
                if(board[i + 2][j + 2] == 1)    result.maxC++;
                else                            result.minC++;
            }
            if( board[i    ][j    ] == 0                        &&
                board[i + 5][j + 5] == board[i + 4][j + 4]*(-1) &&
                board[i + 1][j + 1] != 0                        &&  
                board[i + 1][j + 1] == board[i + 4][j + 4]      &&  
                board[i + 2][j + 2] == 0                        &&
                board[i + 3][j + 3] == 0  
                )
            {   // 010012
                if(board[i + 1][j + 1] == 1)    result.maxC++;
                else                            result.minC++;
            }
        }
    }
    for(int i = 0; i < 11; i++)
    {
        for(int j = 0; j < 11; j++)
        {
            if( board[i    ][j    ] == board[i + 4][j + 4]  &&  
                board[i    ][j    ] != 0                    &&  
                board[i + 1][j + 1] == 0                    &&  
                board[i + 2][j + 2] == 0                    &&
                board[i + 3][j + 3] == 0  
                )
            {   // 10001
                if(board[i][j] == 1)    result.maxC++;
                else                    result.minC++;
            }
        }
    }
    // UR-DL----------------------------------------------------
    for(int i = 6; i < 15; i++)
    {
        for(int j = 0; j < 9; j++)
        {
            if( board[i    ][j    ] == board[i - 2][j + 2]*(-1) &&
                board[i - 6][j + 6] == board[i - 2][j + 2]*(-1) &&  
                board[i - 5][j + 5] == 0                        &&
                board[i - 1][j + 1] == 0                        &&  
                board[i - 2][j + 2] != 0                        &&
                board[i - 2][j + 2] == board[i - 3][j + 3]      &&
                board[i - 4][j + 4] == 0  
                )
            {   // 2011002
                if(board[i - 2][j + 2] == 1)    result.maxC++;
                else                            result.minC++;
            }
            if( board[i    ][j    ] == board[i - 2][j + 2]*(-1) &&
                board[i - 6][j + 6] == board[i - 2][j + 2]*(-1) &&  
                board[i - 5][j + 5] == 0                        &&
                board[i - 1][j + 1] == 0                        &&  
                board[i - 2][j + 2] != 0                        &&
                board[i - 2][j + 2] == board[i - 4][j + 4]      &&
                board[i - 3][j + 3] == 0  
                )
            {   // 2010102
                if(board[i - 2][j + 2] == 1)    result.maxC++;
                else                            result.minC++;
            }
        }
    }
    for(int i = 5; i < 15; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            if( board[i    ][j    ] == 0                        && 
                board[i - 5][j + 5] == board[i - 3][j + 3]*(-1) &&
                board[i - 1][j + 1] == 0                        &&  
                board[i - 2][j + 2] == 0                        &&
                board[i - 4][j + 4] != 0                        &&
                board[i - 4][j + 4] == board[i - 3][j + 3]  
                )
            {   // 000112
                if(board[i - 3][j + 3] == 1)    result.maxC++;
                else                            result.minC++;
            }
            if( board[i    ][j    ] == board[i - 2][j + 2]*(-1) &&
                board[i - 5][j + 5] == 0                        &&
                board[i - 4][j + 4] == 0                        &&  
                board[i - 2][j + 2] != 0                        &&
                board[i - 2][j + 2] == board[i - 1][j + 1]      &&
                board[i - 3][j + 3] == 0  
                )
            {   // 211000
                if(board[i - 2][j + 2] == 1)    result.maxC++;
                else                            result.minC++;
            }
            if( board[i    ][j    ] == 0                        &&
                board[i - 5][j + 5] == board[i - 4][j + 4]*(-1) &&
                board[i - 1][j + 1] != 0                        &&  
                board[i - 1][j + 1] == board[i - 4][j + 4]      &&  
                board[i - 2][j + 2] == 0                        &&
                board[i - 3][j + 3] == 0  
                )
            {   // 010012
                if(board[i - 1][j + 1] == 1)    result.maxC++;
                else                            result.minC++;
            }
        }
    }
    for(int i = 4; i < 15; i++)
    {
        for(int j = 0; j < 11; j++)
        {
            if( board[i    ][j    ] != 0                    &&  
                board[i    ][j    ] == board[i - 4][j + 4]  &&  
                board[i - 1][j + 1] == 0                    &&  
                board[i - 2][j + 2] == 0                    &&
                board[i - 3][j + 3] == 0  
                )
            {   // 10001
                if(board[i][j] == 1)    result.maxC++;
                else                    result.minC++;
            }
        }
    }
    //----------------------------------------------------
    return result;
}