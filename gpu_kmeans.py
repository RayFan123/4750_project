import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule
import math
import time
import matplotlib.pyplot as plt
from scipy import signal
from prettytable import PrettyTable
from sklearn.cluster import KMeans

class GPU:
    def __init__(self):
        self.mod = self.getSourceModule()
        pass
    def getSourceModule(self):
        kernelwrapper = """#include <stdio.h>
            __global__ void dist(float* A, const float* C, int* Y, int n, int m, int k){
                int tid = blockIdx.x * blockDim.x + threadIdx.x;
                if (tid < n){
                    float d = 999999.0f;
                    int index = 0;
                    float current = 0.0f;
                    for (int j = 0; j < k; j ++){
                        current = 0;
                        for (int i = 0; i < m; i ++){
                            current += pow((A[tid * m + i] - C[j * m + i]), 2);
                        }
                        //printf("  dist:%f;d:%f  ", current,d);
                        if(current < d){
                            //printf("change");
                            d = current;
                            index = j;
                        }
                    }
                    Y[tid] = index;
                    //printf(" index:%d ", index);
                }
            }        
            
            __global__ void Mean_initial(float* C, int m, int k){
                int tid = blockIdx.x * blockDim.x + threadIdx.x;
                if (tid < k * m){
                    C[tid] = 0.0f;
                }
            }
            __global__ void Sum(float* A, int* B, float* C, int* count, int n, int m, int k){
                int tid = blockIdx.x * blockDim.x + threadIdx.x;
                int index = B[tid];
                if (tid < n){
                    for (int i = 0; i < m; i ++){
                        atomicAdd(&(C[index * m + i]), A[tid * m + i]);
                    }   
                    atomicAdd(&(count[index]),1);
                }
            }
            __global__ void Mean(float* C, int* count, int m, int k){
                int tid = blockIdx.x * blockDim.x + threadIdx.x;
                if (tid < k){
                    for (int i = 0; i < m; i ++){
                        C[tid * m + i] =  C[tid * m + i] / count[tid];
                    }
                }
            }
                
        """
        return SourceModule(kernelwrapper)

# 加载数据

def loadDataSet(fileName):  # 解析文件，按tab分割字段，得到一个浮点数字类型的矩阵
    dataMat = []  # 文件的最后一个字段是类别标签
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)  # 将每个元素转成float类型
        dataMat.append(fltLine)
    return dataMat


# 计算欧几里得距离
def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))  # 求两个向量之间的距离

# 构建聚簇中心，取k个(此例中为4)随机质心
def randCent(dataSet, k):
    m = np.shape(dataSet)[1]
    n = np.shape(dataSet)[0]
    index = np.random.randint(0,n,k)
    centroids = np.zeros((k,m))  # 每个质心有n个坐标值，总共要k个质心
    for j in range(k):
        centroids[j, :] = dataSet[index[j]]
    return centroids


def cluster_assign_python(dataSet, centroids, clusterAssment, distMeans,clusterChanged):
    m = np.shape(dataSet)[0]
    k = np.shape(centroids)[0]
    for i in range(m):  # 把每一个数据点划分到离它最近的中心点
        minDist = np.inf
        minIndex = -1
        for j in range(k):
            distJI = distMeans(centroids[j, :], dataSet[i, :])
            #print(distJI)
            if distJI < minDist:
                minDist = distJI
                minIndex = j  # 如果第i个数据点到第j个中心点更近，则将i归属为j
        if (clusterAssment[i] != minIndex):
            clusterChanged = True  # 如果分配发生变化，则需要继续迭代
            clusterAssment[i] = minIndex
              # 并将第i个数据点的分配情况存入字典
    return clusterAssment,clusterChanged

def cluster_assign_CUDA(dataSet, centroids, clusterAssment, distMeans,clusterChanged):
    n = np.shape(dataSet)[0]
    k = np.shape(centroids)[0]
    m = np.shape(dataSet)[1]
    s_len = int(math.ceil(n / 256))
    blockDim = (256, 1, 1)
    gridDim = (s_len, 1, 1)
    temp = clusterAssment.copy()
    A_gpu = cuda.mem_alloc(dataSet.nbytes)
    C_gpu = cuda.mem_alloc(centroids.nbytes)
    Y_gpu = cuda.mem_alloc(clusterAssment.nbytes)
    cuda.memcpy_htod(A_gpu, dataSet)
    cuda.memcpy_htod(C_gpu, centroids)
    func = mod.get_function("dist")
    func(A_gpu, C_gpu, Y_gpu, np.int32(n), np.int32(m),np.int32(k), block=blockDim, grid=gridDim)
    cuda.memcpy_dtoh(clusterAssment, Y_gpu)
    if ((temp == clusterAssment).all()):
        pass
    else:
        clusterChanged = True
    return clusterAssment, clusterChanged

def Mean_cal_CUDA(dataSet,clusterAssment,centroids,k):
    n = np.shape(dataSet)[0]
    m = np.shape(dataSet)[1]
    s_len = int(math.ceil(n / 128))
    blockDim = (128, 1, 1)
    gridDim = (s_len, 1, 1)
    count = np.zeros(k)
    count = count.astype(np.int32)
    count_gpu = cuda.mem_alloc(count.nbytes)
    A_gpu = cuda.mem_alloc(dataSet.nbytes)
    B_gpu = cuda.mem_alloc(clusterAssment.nbytes)
    Y_gpu = cuda.mem_alloc(centroids.nbytes)
    cuda.memcpy_htod(count_gpu, count)
    cuda.memcpy_htod(A_gpu, dataSet)
    cuda.memcpy_htod(B_gpu, clusterAssment)
    func = mod.get_function("Mean")
    func(A_gpu, B_gpu, Y_gpu, count_gpu, np.int32(n), np.int32(m),np.int32(k), block=blockDim, grid=gridDim)
    cuda.memcpy_dtoh(centroids, Y_gpu)
    cuda.memcpy_dtoh(count, count_gpu)
    #print(sum(count))
    return centroids
# k-means 聚类算法
def kMeans(dataSet, k, gpu, means, init_centroids,distMeans=distEclud):
    start = time.time()
    m = np.shape(dataSet)[0]
    centroids = init_centroids.copy()
    clusterAssment = np.zeros(m).astype(np.int32)  # 用于存放该样本属于哪类及质心距离
    # clusterAssment第一列存放该数据所属的中心点，第二列是该数据到中心点的距离
    clusterChanged = True  # 用来判断聚类是否已经收敛
    iter = 0
    while clusterChanged:
        clusterChanged = False
        if(gpu == True):
            clusterAssment,clusterChanged = cluster_assign_CUDA(dataSet, centroids, clusterAssment, distMeans, clusterChanged)
        else:
            clusterAssment, clusterChanged = cluster_assign_python(dataSet, centroids, clusterAssment, distMeans,clusterChanged)
        #print("clusterAssment",clusterAssment)
        if means == True:
            centroids = Mean_cal_CUDA(dataSet, clusterAssment, centroids, k)
        else:
            for cent in range(k):  # 重新计算中心点
                ptsInClust = dataSet[np.nonzero((clusterAssment == cent).astype(int))]  # 去第一列等于cent的所有列
                centroids[cent, :] = np.mean(ptsInClust, axis=0)  # 算出这些数据的中心点
        iter += 1
    print(iter)
    end = time.time()
    return centroids, clusterAssment,end - start,iter

def Kmeans_gpu_integrate(dataSet, k, init_centroids):
    start = time.time()
    n = np.shape(dataSet)[0]
    m = np.shape(dataSet)[1]
    centroids = init_centroids.copy()
    clusterAssment = np.zeros(n).astype(np.int32)
    clusterChanged = True
    s_len = int(math.ceil(n / 256))
    blockDim = (256, 1, 1)
    gridDim = (s_len, 1, 1)
    A_gpu = cuda.mem_alloc(dataSet.nbytes)
    C_gpu = cuda.mem_alloc(centroids.nbytes)
    Y_gpu = cuda.mem_alloc(clusterAssment.nbytes)
    count = np.zeros(k).astype(np.int32)
    z = count.copy()
    count_gpu = cuda.mem_alloc(count.nbytes)
    cuda.memcpy_htod(A_gpu, dataSet)
    cuda.memcpy_htod(C_gpu, centroids)
    iter = 0
    while clusterChanged:
        clusterChanged = False
        temp = clusterAssment.copy()

        func1 = mod.get_function("dist")
        func1(A_gpu, C_gpu, Y_gpu, np.int32(n), np.int32(m), np.int32(k), block=blockDim, grid=gridDim)

        func2 = mod.get_function("Mean_initial")
        func2(C_gpu, np.int32(m), np.int32(k), block=(256,1,1), grid=(1,1,1))
        cuda.memcpy_htod(count_gpu, z)

        func3 = mod.get_function("Sum")
        func3(A_gpu, Y_gpu, C_gpu, count_gpu, np.int32(n), np.int32(m),np.int32(k), block=blockDim, grid=gridDim)

        func4 = mod.get_function("Mean")
        func4(C_gpu, count_gpu, np.int32(m), np.int32(k), block=(32,1,1), grid=(1,1,1))
        # cuda.memcpy_dtoh(count, count_gpu)
        # print(count)
        cuda.memcpy_dtoh(clusterAssment, Y_gpu)
        cuda.memcpy_dtoh(centroids, C_gpu)
        # print(centroids)
        if ((temp == clusterAssment).all()):
            pass
        else:
            clusterChanged = True
        iter += 1
    cuda.memcpy_dtoh(centroids, C_gpu)
    print(iter)
    end = time.time()
    return centroids, clusterAssment, end - start, iter
# --------------------测试----------------------------------------------------
# 用测试数据及测试kmeans算法
m = 5
k = 5
x = np.random.normal(loc=1, scale=2, size=(100, m))
y = np.random.normal(loc=3, scale=1, size=(100, m))
dataset = np.vstack((x,y)).astype(np.float32)
init_centroids = randCent(dataset, k).astype(np.float32)
inst = GPU()
mod = inst.mod

#myCentroids, clustAssign,t1,iter1 = kMeans(dataset, k, True, True, init_centroids)
myCentroids2, clustAssign2,t2,iter2 = Kmeans_gpu_integrate(dataset, k, init_centroids)
myCentroids, clustAssign,t3,iter3 = kMeans(dataset, k, True, False, init_centroids)
#myCentroids, clustAssign,t2,iter2 = kMeans(dataset, 3, False,init_centroids)

start = time.time()
cluster = KMeans(n_clusters=k,random_state=0).fit(dataset)
end = time.time()
print(t2,t3,end-start)
print(myCentroids)
print(clustAssign)
