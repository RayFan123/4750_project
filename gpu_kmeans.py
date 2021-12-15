# encoding: utf-8
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
                int t = threadIdx.x;
                if (tid < n){
                    float d = 999999.9f;
                    int index = 0;
                    float current = 0.0f;
                    for (int j = 0; j < k; j ++){
                        current = 0;
                        for (int i = 0; i < m; i ++){
                            current += pow((A[tid * m + i] - C[j * m + i]), 2);
                            //current += pow((data[t][i] - C[j * m + i]), 2);
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
            __global__ void dist_shared(float* A, const float* C, int* Y, int n, int m, int k){
                int tid = blockIdx.x * blockDim.x + threadIdx.x;
                int t = threadIdx.x;
                __shared__ float data[256][8];
                if (tid < n){
                    for (int i = 0; i< m; i ++){
                        data[t][i] = A[tid * m + i];
                    }
                }
                __syncthreads();
                if (tid < n){
                    float d = 999999.9f;
                    int index = 0;
                    float current = 0.0f;
                    for (int j = 0; j < k; j ++){
                        current = 0;
                        for (int i = 0; i < m; i ++){
                            //current += pow((A[tid * m + i] - C[j * m + i]), 2);
                            current += pow((data[t][i] - C[j * m + i]), 2);
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
            
            __global__ void sum_shared(float* A, float* C, int* Y, int n, int m, int k,int* count){
                // for the shared memory, 1st index is for block size, 2nd index for feature number
                __shared__ float data[512][8];
                int tid = threadIdx.x;
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if(i < n && Y[i] == k){
                    atomicAdd(&(count[k]),1);
                    for (int a=0;a<m;a++){
                        data[tid][a] = A[i * m + a];
                    }
                }
                else{
                    for (int a=0;a<m;a++){
                        data[tid][a] = 0.0f;
                    }
                }
                __syncthreads();

                for (unsigned int stride = blockDim.x/2; stride>0; stride>>=1){
                    if (tid < stride) {
                        for (int a=0;a<m;a++){
                            data[tid][a] += data[tid + stride][a];
                        }
                    }
                    __syncthreads();
                }

                if (tid < m){
                    atomicAdd(&(C[k*m + tid]),data[0][tid]);
                    //C[k*m + tid] = data[0][tid];
                }
            }
            
            __global__ void sum_partial(float* A, float* diff, int* Y_1, int* Y_2, int n, int m, int k, int* d_count){
                int tid = blockIdx.x * blockDim.x + threadIdx.x;
                if (tid < n){
                    int old = Y_1[tid];
                    int cur = Y_2[tid];
                    if (old >= k){
                        //printf(" cur:%d ", cur);
                        //printf(" count:%d ", d_count[cur]);
                        for (int i = 0;i < m;i++){
                            atomicAdd( & (diff[cur * m + i]), A[tid * m + i]);
                            atomicAdd( & (d_count[cur]), 1);
                        }
                    }
                    else{
                        if (old != cur){
                            for (int i = 0;i < m;i++){
                                atomicAdd( & (diff[old * m + i]), -A[tid * m + i]);
                                atomicAdd( & (diff[cur * m + i]), A[tid * m + i]);
                                atomicAdd( & (d_count[old]), -1);
                                atomicAdd( & (d_count[cur]), 1);
                            }
                        }
                    }
                }
            }
            
            __global__ void mean_partial(float* C, float* diff, int m, int k, int* count, int* d_count){
                int tid = threadIdx.x;
                if (tid < k){
                    for (int i = 0; i<m;i++){
                        C[tid * m + i] = (C[tid * m + i] * count[tid] + diff[tid * m + i]) / (count[tid] + d_count[tid]);
                    }
                    count[tid] += d_count[tid];
                }
            }
            
            __global__ void update(int* Y_1, int* Y_2,int n){
                int tid = blockIdx.x * blockDim.x + threadIdx.x;
                if (tid < n){
                    Y_1[tid] = Y_2[tid];
                }
            }
        """
        return SourceModule(kernelwrapper)

def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


def randCent(dataSet, k):
    m = np.shape(dataSet)[1]
    n = np.shape(dataSet)[0]
    index = np.random.randint(0,n,k)
    centroids = np.zeros((k,m))
    for j in range(k):
        centroids[j, :] = dataSet[index[j]]
    return centroids


def cluster_assign_python(dataSet, centroids, clusterAssment, distMeans,clusterChanged):
    m = np.shape(dataSet)[0]
    k = np.shape(centroids)[0]
    for i in range(m):
        minDist = np.inf
        minIndex = -1
        for j in range(k):
            distJI = distMeans(centroids[j, :], dataSet[i, :])
            #print(distJI)
            if distJI < minDist:
                minDist = distJI
                minIndex = j
        if (clusterAssment[i] != minIndex):
            clusterChanged = True
            clusterAssment[i] = minIndex

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

def kMeans(dataSet, k, gpu, means, init_centroids,distMeans=distEclud):
    start = time.time()
    m = np.shape(dataSet)[0]
    centroids = init_centroids.copy()
    clusterAssment = np.zeros(m).astype(np.int32)

    clusterChanged = True
    iter = 0
    while clusterChanged:
        clusterChanged = False
        if(gpu == True):
            temp = np.copy(clusterAssment)
            clusterAssment,clusterChanged = cluster_assign_CUDA(dataSet, centroids, clusterAssment, distMeans, clusterChanged)
            chan = np.count_nonzero(temp-clusterAssment)
            # print(chan)
        else:
            clusterAssment, clusterChanged = cluster_assign_python(dataSet, centroids, clusterAssment, distMeans,clusterChanged)
        #print("clusterAssment",clusterAssment)
        if means == True:
            centroids = Mean_cal_CUDA(dataSet, clusterAssment, centroids, k)
        else:
            for cent in range(k):
                ptsInClust = dataSet[np.nonzero((clusterAssment == cent).astype(int))]
                centroids[cent, :] = np.mean(ptsInClust, axis=0)
        iter += 1
    end = time.time()
    print(iter)
    return centroids, clusterAssment,end - start,iter

def Kmeans_gpu_integrate(dataSet, k, init_centroids, use_share):
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
    z = np.zeros(k).astype(np.int32)
    count_gpu = cuda.mem_alloc(count.nbytes)
    cuda.memcpy_htod(A_gpu, dataSet)
    cuda.memcpy_htod(C_gpu, centroids)
    time_ = 0
    iter = 0
    while clusterChanged:
        clusterChanged = False
        temp = clusterAssment.copy()
        cstart = cuda.Event()
        end = cuda.Event()
        cstart.record()
        func1 = mod.get_function("dist")
        func1(A_gpu, C_gpu, Y_gpu, np.int32(n), np.int32(m), np.int32(k), block=blockDim, grid=gridDim)
        end.record()
        end.synchronize()
        time_ = time_ + cstart.time_till(end) * 1e-3

        func2 = mod.get_function("Mean_initial")
        func2(C_gpu, np.int32(m), np.int32(k), block=(256,1,1), grid=(1,1,1))
        cuda.memcpy_htod(count_gpu, z)

        if use_share == True:
            func3 = mod.get_function("sum_shared")
            for cent in range(k):
                func3(A_gpu, C_gpu, Y_gpu, np.int32(n), np.int32(m), np.int32(cent), count_gpu, block=(512, 1, 1),
                      grid=(int(math.ceil(n / 512)), 1, 1))
        else:
            func3 = mod.get_function("Sum")
            func3(A_gpu, Y_gpu, C_gpu, count_gpu, np.int32(n), np.int32(m),np.int32(k), block=blockDim, grid=gridDim)


        func4 = mod.get_function("Mean")
        func4(C_gpu, count_gpu, np.int32(m), np.int32(k), block=(32,1,1), grid=(1,1,1))
        # cuda.memcpy_dtoh(count, count_gpu)
        # print(count)
        cuda.memcpy_dtoh(clusterAssment, Y_gpu)
        # cuda.memcpy_dtoh(centroids, C_gpu)
        # print(centroids)
        if ((temp == clusterAssment).all()):
            pass
        else:
            clusterChanged = True
        iter += 1
    cuda.memcpy_dtoh(centroids, C_gpu)
    print(iter)
    end = time.time()
    return centroids, clusterAssment, end - start, iter,time_

def Kmeans_gpu_partial(dataSet, k, init_centroids):
    start = time.time()
    n = np.shape(dataSet)[0]
    m = np.shape(dataSet)[1]
    centroids = init_centroids.copy()
    clusterAssment = np.zeros(n).astype(np.int32)
    diff = np.zeros((k,m)).astype(np.float32)
    clusterChanged = True
    s_len = int(math.ceil(n / 256))
    blockDim = (256, 1, 1)
    gridDim = (s_len, 1, 1)
    A_gpu = cuda.mem_alloc(dataSet.nbytes)
    C_gpu = cuda.mem_alloc(centroids.nbytes)
    Diff_gpu = cuda.mem_alloc(centroids.nbytes)
    Y_gpu = cuda.mem_alloc(clusterAssment.nbytes)
    Y1_gpu = cuda.mem_alloc(clusterAssment.nbytes)

    count = np.zeros(k).astype(np.int32)
    z = np.zeros(k).astype(np.int32)
    count_gpu = cuda.mem_alloc(count.nbytes)
    D_count_gpu = cuda.mem_alloc(count.nbytes)
    cuda.memcpy_htod(Y1_gpu, (np.ones(n)+k).astype(np.int32))
    cuda.memcpy_htod(A_gpu, dataSet)
    cuda.memcpy_htod(C_gpu, centroids)
    cuda.memcpy_htod(count_gpu, z)
    iter = 0
    time_ = 0
    while clusterChanged:
        cstart = cuda.Event()
        end = cuda.Event()
        cstart.record()
        clusterChanged = False
        temp = clusterAssment.copy()

        func1 = mod.get_function("dist")
        func1(A_gpu, C_gpu, Y_gpu, np.int32(n), np.int32(m), np.int32(k), block=blockDim, grid=gridDim)
        # end.record()
        # end.synchronize()
        #time_ = time_ + cstart.time_till(end) * 1e-3
        cuda.memcpy_htod(D_count_gpu, z)
        cuda.memcpy_htod(Diff_gpu, diff)
        func3 = mod.get_function("sum_partial")
        func3(A_gpu, Diff_gpu, Y1_gpu, Y_gpu, np.int32(n), np.int32(m),np.int32(k),  D_count_gpu, block=(512, 1, 1),grid=(int(math.ceil(n / 512)), 1, 1))


        func4 = mod.get_function("mean_partial")
        func4(C_gpu, Diff_gpu, np.int32(m), np.int32(k), count_gpu, D_count_gpu, block=(32,1,1), grid=(1,1,1))


        func2 = mod.get_function("update")
        func2(Y1_gpu, Y_gpu, np.int32(n), block=blockDim, grid=gridDim)
        # cuda.memcpy_dtoh(count, count_gpu)
        # print(count)
        cuda.memcpy_dtoh(clusterAssment, Y_gpu)
        end.record()
        end.synchronize()
        time_ = time_ + cstart.time_till(end) * 1e-3
        #cuda.memcpy_dtoh(centroids, C_gpu)
        # print(centroids)
        if ((temp == clusterAssment).all()):
            pass
        else:
            clusterChanged = True
        iter += 1
    cuda.memcpy_dtoh(centroids, C_gpu)
    print(iter)
    end = time.time()
    return centroids, clusterAssment, end - start, iter,time_



m = 8
k = 5
n = [100,1000,10000,100000,1000000]
#y = np.random.normal(loc=3, scale=1, size=(50000, m))


t = [0,0,0,0,0]
iter = [0,0,0,0,0]
inst = GPU()
mod = inst.mod
for j in range(5):
    x = np.random.normal(loc=1, scale=2, size=(n[j], m))
    dataset = x.astype(np.float32)
    for i in range(10):
        init_centroids = randCent(dataset, k).astype(np.float32)
        myCentroids, clustAssign,t_t,iter_t = kMeans(dataset, k, True, False, init_centroids)
        t[j] += t_t
        iter[j] += iter_t
    t[j] /= 10
    iter[j] /=10

# myCentroids, clustAssign,t,iter,time_ = Kmeans_gpu_partial(dataset, k, init_centroids)
#myCentroids2, clustAssign2,t2,iter2,time_1 = Kmeans_gpu_integrate(dataset, k, init_centroids,False)
#myCentroids2, clustAssign2,t2,iter2,time_1 = Kmeans_gpu_integrate(dataset, k, init_centroids,True)
#myCentroids3, clustAssign3,t3,iter3 = kMeans(dataset, k, True, False, init_centroids)
#myCentroids, clustAssign,t2,iter2 = kMeans(dataset, 3, False,init_centroids)

# start = time.time()
# cluster = KMeans(n_clusters=k,max_iter=2000,n_init = 1,init = init_centroids,algorithm="full",tol=0.00000001).fit(dataset)
# end = time.time()

# print(cluster.n_iter_)
# print(time_,t,time_1,t2,end-start)
# print(myCentroids)
# print(cluster.cluster_centers_)

print(t)
print(iter)
