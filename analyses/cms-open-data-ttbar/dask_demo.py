import dask
import dask.array
import dask.distributed
import os

DASK_SCHEDULER_URI = os.getenv("DASK_SCHEDULER_URI", "tcp://127.0.0.1:8080")
client = dask.distributed.Client(DASK_SCHEDULER_URI)

x = dask.array.random.random((10000, 10000), chunks=(1000, 1000))
y = x + x.T
z = y[::2, 5000:].mean(axis=1)

result = z.compute()
print(result)
