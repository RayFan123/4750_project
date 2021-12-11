# 4750_project
This py file now only implement 2 methods for kmeans speedup using gpu.  

We split the work load of each iteration in two parts, first is calculating distance between each datapoint and centroids, and assign datapoints accordingly. The second part is to update the centroids based on the new assignment.   

We first use the GPU kernel called dist() to calculate distance and assign the points, then there are 3 ways of updating centroids. We can use the Numpy function np.nonzero() and np.mean() to update, or we can use np.nonezero() first to separate datas into groups and use gpu function means() to calculate each centroids. But in practice this yields even worse performance than before.  

So instead we introduce another function called Kmeans_gpu_integrate(). in this function, once we setup the memory space in the GPU, we run the multiple kernel functions for each iteration without transfering between device memory and host memory all the time, therefore reduce over head.
