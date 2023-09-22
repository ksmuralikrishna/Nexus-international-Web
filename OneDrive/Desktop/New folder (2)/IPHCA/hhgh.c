#include <stdio.h>
int main()
{
    int tm;
    printf("nte");
    scanf("%d", &tm);
    if(tm<=50)
    {
        goto Fail;
    }
    else
    {
        goto Pass;
    }
    Pass:
        printf("p");
    goto End;
    Fail:
        printf("f");
    End:
return 0;
}