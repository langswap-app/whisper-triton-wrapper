import grpc
import tritonclient.grpc as grpcclient
from tritonclient.grpc import service_pb2
from tritonclient.grpc import service_pb2_grpc

class BaseTritonClient:

    def __init__(self, url: str, model_name: str) -> None:
        self._url = url
        self._model_name = model_name
        self._client = grpcclient.InferenceServerClient(url=self._url)

    def load_model(self, model_name: str = None) -> None:
        # Create a gRPC stub for communicating with the server
        channel = grpc.insecure_channel(self._url)
        stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)

        # Set the model name
        model_name = self._model_name if model_name is None else model_name

        # Create a request to unload the model
        request = service_pb2.RepositoryModelLoadRequest(
            model_name=model_name,
        )

        # Send the request
        response = stub.RepositoryModelLoad(request)

        return None

    def unload_model(self, model_name: str = None) -> None:
        # Create a gRPC stub for communicating with the server
        channel = grpc.insecure_channel(self._url)
        stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)

        # Set the model name
        model_name = self._model_name if model_name is None else model_name

        # Create a request to unload the model
        request = service_pb2.RepositoryModelUnloadRequest(
            model_name=model_name,
        )

        # Send the request
        response = stub.RepositoryModelUnload(request)

        return None