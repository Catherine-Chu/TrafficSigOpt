import math
import sys
block = [4, 4, 3, 5, 3, 4, 3]
fr1=open("./result.txt","r")
fr2=open("./result2.txt","r")
fw1=open("./final.txt","w")

g_upQ = []
g_downQ = []
g_dist = [637.789591088,428.654872829,399.577902292,814.455991752,978.273287482,900.826634264]
g_upV = []
g_upmaxv = 0
g_downV = []
g_downmaxv = 0
g_fullQ = []
g_Q = []
minPhase = [[35,35,8,10],[8,35,8,35],[10,35,35],[10,35,10,8,35],[10,35,35],[10,35,10,35],[10,35,35]] # 最小相位时长

# data="98.0,0.4591836734693877,0.47959183673469385,0.5408163265306123,0.35714285714285715,0.5408163265306123,0.35714285714285715,0.5317421460639554,49.36058162745286,73.31739828292888,73.63636199757457,64.1563268370589,71.60951881121207,44.824303560409135"

data="300,0.11666666666666667,0.21603616411218218,0.85,0.11666666666666667,0.85,0.3626273526890298,0.11666666666666667,258.45162329222103,278.79512501170456,26.096486098573592,35.529888588754204,11.568241312667737,11.899374162481195"
def readFiles():
    global g_upmaxv,g_downmaxv
    xp=0
    yp=0
    f1_lines=fr1.readlines()
    for line in f1_lines:
        line=line.strip()
        if(line.isdigit()):
            xp=int(line)
            g_fullQ.append([])
            g_Q.append([])
        else:
            tmp=line.split()
            yp=int(tmp[0])
            g_fullQ[xp].append(float(tmp[2]))
            g_Q[xp].append(float(tmp[1]))

    '''调整第一个路口相位的特殊性'''
    pos =len(g_Q[0]) - 1
    tmp1=g_Q[0][pos]
    tmp2=g_fullQ[0][pos]
    while pos>0:
        g_Q[0][pos]=g_Q[0][pos-1]
        g_fullQ[0][pos]=g_fullQ[0][pos-1]
        pos-=1
    g_Q[0][0]=tmp1
    g_fullQ[0][0]=tmp2

    f2_lines=fr2.readlines()
    for line in f2_lines:
        line=line.strip()
        if(line.isdigit()):
            xp=int(line)
            yp=0
        else:
            tmp=line.split()
            if(yp==0):
                '''up'''
                g_upQ.append(float(tmp[1]))
                g_upV.append(float(tmp[2]))
                yp=1
            else:
                '''down'''
                g_downQ.append(float(tmp[1]))
                g_downV.append(float(tmp[2]))
                yp=0
    g_upmaxv=max(g_upV)
    g_downmaxv=max(g_downV)

def getLamda(list, q):
    lamda = []
    Qi = []
    for i in range(n):
        Qi.append(0)
        for j in range(block[i]):
            if j != 1:
                Qi[i] += q[i][j]
    for i in range(n):
        lamda.append([])
        for j in range(block[i]):
            if j == 1:
                lamda[i].append(list[i + 1])
            else:
                lamda[i].append((1 - list[i + 1]) * q[i][j] / Qi[i])
    return lamda

def summ(i,beg):
    sum=0
    for j in range(block[i]):
        if j<beg or j==1:
            continue
        else:
            sum+=g_Q[i][j]
    return sum
def summ1(i,beg):
    sum=0
    for j in range(block[i]):
        if j<beg or j==1:
            continue
        else:
            sum+=minPhase[i][j]
    return sum

if __name__ == '__main__':

    lists=data.split(',')
    for i in range(len(lists)):
        lists[i]=float(lists[i])

    readFiles()
    C=lists[0]
    n=7
    lamda=getLamda(lists,g_Q)
    result=[]
    diff=[]
    for i in range(n):
        result.append([])
        sum=0
        op=0
        for k in range(block[i]):
            if k==1:
                result[i].append(int(C*lamda[i][k]))
                print("aaaaaa%f"%result[i][k])
            else:
                a=int(C * lamda[i][k])
                b=minPhase[i][k]
                if a<=b:
                    result[i].append(b)
                elif a>b:
                    result[i].append(min(C-sum-summ1(i, k+1),a))
                # result[i].append(max(int(C*lamda[i][k]),minPhase[i][k]))

                # print("%f,%f"%(C*lamda[i][k],minPhase[i][k]))
                op+=float(result[i][k])/C
                # print(op)
                k1=k+1
                while k1 < block[i]:
                    if k1!=1:
                        lamda[i][k1]=(1 - op-lamda[i][1]) * g_Q[i][k1] / summ(i,k1)
                        # print("%d,%d,%f"%(i,k1,lamda[i][k1]))
                    k1+=1
            sum+=result[i][k]
        # print(sum)
        if sum<C:
            result[i][1]=result[i][1]+C-sum
        elif sum >C:
            print("error%d,%d"%(i,(sum-C)))
    tmp=result[0][0]
    for i in range(block[0]-1):
        result[0][i]=result[0][i+1]
    result[0][block[0]-1]=tmp

    diff.append(0)
    sum=0
    for i in range(n-1):
        sum+=lists[n+1+i]
        dela=math.ceil(sum-result[i+1][0])%C
        if dela<0:
            dela=C+dela
        diff.append(dela)

    for i in range(len(result)):
        fw1.write("%d" % diff[i])
        for j in range(len(result[i])):
            fw1.write(",%d"%result[i][j])
        fw1.write(";")

    fr1.close()
    fr2.close()
    fw1.flush()
    fw1.close()
    sys.exit(0)
