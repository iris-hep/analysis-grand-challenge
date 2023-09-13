def get_client(af="coffea_casa"):
    if af == "coffea_casa":
        from dask.distributed import Client

        client = Client("tls://localhost:8786")

    elif af == "EAF":
        from htcdaskgateway import HTCGateway

        gateway = HTCGateway()
        cluster = gateway.new_cluster()
        cluster.scale(10)
        print("Please allow up to 60 seconds for HTCondor worker jobs to start")
        print(f"Cluster dashboard: https://dask-gateway.fnal.gov/clusters/{str(cluster.name)}/status")

        client = cluster.get_client()

    elif af == "cmsaf-dev":
        from dask_gateway import Gateway
        
        gateway = Gateway()
        clusters = gateway.list_clusters()
        # by default you will have at least one Dask gateway cluster, but here could be more
        # to shut down cluster please use: `cluster.shutdown()`
        cluster = gateway.connect(clusters[0].name)
        # adjust number of workers manually
        cluster.scale(50)
        
        client = cluster.get_client()
        
    elif af == "local":
        from dask.distributed import Client

        client = Client()

    else:
        raise NotImplementedError(f"unknown analysis facility: {af}")

    return client

def get_triton_client(triton_url):
    
    import tritonclient.grpc as grpcclient
    triton_client = grpcclient.InferenceServerClient(url=triton_url)
    
    return triton_client
