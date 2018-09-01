import os
import torch
from torch import nn, optim
import numpy as np
import sys
import logging
from cloudpickle import CloudPickler
import tempfile
import tarfile
import docker
import shutil
import time
import torchvision.models as models

model_name = "pytorch-model"

PYTORCH_WEIGHTS_RELATIVE_PATH = "pytorch_weights.pkl"
PYTORCH_MODEL_RELATIVE_PATH = "pytorch_model.pkl"

def build_model(name,
                model_data_path,
                base_image,
                pkgs_to_install=None):
    """Build a new model container Docker image with the provided data"

    This method builds a new Docker image from the provided base image with the local directory
    specified by ``model_data_path`` copied into the image. The Dockerfile that gets generated
    to build the image is equivalent to the following::

        FROM <base_image>
        COPY <model_data_path> /model/

    The newly built image is then pushed to the specified container registry. If no container
    registry is specified, the image will be pushed to the default DockerHub registry. Clipper
    will tag the newly built image with the tag [<registry>]/<name>:<version>.

    This method can be called without being connected to a Clipper cluster.

    Parameters
    ----------
    name : str
        The name of the deployed model.
    model_data_path : str
        A path to a local directory. The contents of this directory will be recursively copied
        into the Docker container.
        base_image : str
        The base Docker image to build the new model image from. This
        image should contain all code necessary to run a Clipper model
        container RPC client.
    pkgs_to_install : list (of strings), optional
        A list of the names of packages to install, using pip, in the container.
        The names must be strings.
    Returns
    -------
    str :
        The fully specified tag of the newly built image. This will include the
        container registry if specified.

    Raises
    ------
    :py:exc:`clipper.ClipperException`

    Note
    ----
    Both the model name and version must be valid DNS-1123 subdomains. Each must consist of
    lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric
    character (e.g. 'example.com', regex used for validation is
    '[a-z0-9]([-a-z0-9]*[a-z0-9])?\Z'.
    """

    run_cmd = ''
    if pkgs_to_install:
        run_as_lst = 'RUN apt-get -y install build-essential && pip install'.split(
            ' ')
        run_cmd = ' '.join(run_as_lst + pkgs_to_install)
    with tempfile.NamedTemporaryFile(
            mode="w+b", suffix="tar") as context_file:
        # Create build context tarfile
        with tarfile.TarFile(
                fileobj=context_file, mode="w") as context_tar:
            context_tar.add(model_data_path)
            # From https://stackoverflow.com/a/740854/814642
            try:
                df_contents = StringIO(
                    str.encode(
                        "FROM {container_name}\n{run_command}\nCOPY {data_path} /model/\n".
                        format(
                            container_name=base_image,
                            data_path=model_data_path,
                            run_command=run_cmd)))
                df_tarinfo = tarfile.TarInfo('Dockerfile')
                df_contents.seek(0, os.SEEK_END)
                df_tarinfo.size = df_contents.tell()
                df_contents.seek(0)
                context_tar.addfile(df_tarinfo, df_contents)
            except TypeError:
                df_contents = StringIO(
                    "FROM {container_name}\n{run_command}\nCOPY {data_path} /model/\n".
                    format(
                        container_name=base_image,
                        data_path=model_data_path,
                        run_command=run_cmd))
                df_tarinfo = tarfile.TarInfo('Dockerfile')
                df_contents.seek(0, os.SEEK_END)
                df_tarinfo.size = df_contents.tell()
                df_contents.seek(0)
                context_tar.addfile(df_tarinfo, df_contents)
        # Exit Tarfile context manager to finish the tar file
        # Seek back to beginning of file for reading
        context_file.seek(0)
        image = "{name}".format(name=name)
        docker_client = docker.from_env()
        logger.warning("Building model Docker image with model data from {}".format(model_data_path))
        image_result, build_logs = docker_client.images.build(
            fileobj=context_file, custom_context=True, tag=image)
        for b in build_logs:
            if 'stream' in b and b['stream'] != '\n':  #log build steps only
                logger.warning(b['stream'].rstrip())

    return image

def build_and_deploy_model(name,
                           input_type,
                           model_data_path,
                           base_image,
                           labels=None,
                           container_registry=None,
                           num_replicas=1,
                           batch_size=-1,
                           pkgs_to_install=None):
    """Build a new model container Docker image with the provided data and deploy it as
    a model to Clipper.

    This method does two things.

    1. Builds a new Docker image from the provided base image with the local directory specified
    by ``model_data_path`` copied into the image by calling
    :py:meth:`clipper_admin.ClipperConnection.build_model`.

    2. Registers and deploys a model with the specified metadata using the newly built
    image by calling :py:meth:`clipper_admin.ClipperConnection.deploy_model`.

    Parameters
    ----------
    name : str
        The name of the deployed model
    input_type : str
        The type of the request data this endpoint can process. Input type can be
        one of "integers", "floats", "doubles", "bytes", or "strings". See the
        `User Guide <http://clipper.ai/user_guide/#input-types>`_ for more details
        on picking the right input type for your application.
    model_data_path : str
        A path to a local directory. The contents of this directory will be recursively copied
        into the Docker container.
    base_image : str
        The base Docker image to build the new model image from. This
        image should contain all code necessary to run a Clipper model
        container RPC client.
    labels : list(str), optional
        A list of strings annotating the model. These are ignored by Clipper
        and used purely for user annotations.
        container_registry : str, optional
        The Docker container registry to push the freshly built model to. Note
        that if you are running Clipper on Kubernetes, this registry must be accesible
        to the Kubernetes cluster in order to fetch the container from the registry.
    num_replicas : int, optional
        The number of replicas of the model to create. The number of replicas
        for a model can be changed at any time with
        :py:meth:`clipper.ClipperConnection.set_num_replicas`.
    batch_size : int, optional
        The user-defined query batch size for the model. Replicas of the model will attempt
        to process at most `batch_size` queries simultaneously. They may process smaller
        batches if `batch_size` queries are not immediately available.
        If the default value of -1 is used, Clipper will adaptively calculate the batch size for
        individual replicas of this model.
    pkgs_to_install : list (of strings), optional
        A list of the names of packages to install, using pip, in the container.
        The names must be strings.
    Raises
    ------
    :py:exc:`clipper.UnconnectedException`
    :py:exc:`clipper.ClipperException`
    """

    image = build_model(name, model_data_path, base_image, pkgs_to_install)

if sys.version_info < (3, 0):
    try:
        from cStringIO import StringIO
    except ImportError:
        from StringIO import StringIO
    PY3 = False
else:
    from io import BytesIO as StringIO
    PY3 = True

logger = logging.getLogger(__name__)

def serialize_object(obj):
    s = StringIO()
    c = CloudPickler(s, 2)
    c.dump(obj)
    return s.getvalue()


def save_python_function(name, func):
    predict_fname = "func.pkl"

    # Serialize function
    s = StringIO()
    c = CloudPickler(s, 2)
    c.dump(func)
    serialized_prediction_function = s.getvalue()

    # Set up serialization directory
    serialization_dir = os.path.abspath(tempfile.mkdtemp(suffix='clipper'))
    logger.warning("Saving function to {}".format(serialization_dir))

    # Write out function serialization
    func_file_path = os.path.join(serialization_dir, predict_fname)
    if sys.version_info < (3, 0):
        with open(func_file_path, "w") as serialized_function_file:
            serialized_function_file.write(serialized_prediction_function)
    else:
        with open(func_file_path, "wb") as serialized_function_file:
            serialized_function_file.write(serialized_prediction_function)
    logging.warning("Serialized and supplied predict function")
    return serialization_dir

class ClipperException(Exception):
    """A generic exception indicating that Clipper encountered a problem."""

    def __init__(self, msg, *args):
        self.msg = msg
        super(Exception, self).__init__(msg, *args)

def deploy_pytorch_model(name,
                         input_type,
                         func,
                         pytorch_model,
                         base_image="default",
                         labels=None,
                         registry=None,
                         num_replicas=1,
                         batch_size=-1,
                         pkgs_to_install=None):
    """Deploy a Python function with a PyTorch model.

    Parameters
    ----------
    clipper_conn : :py:meth:`clipper_admin.ClipperConnection`
        A ``ClipperConnection`` object connected to a running Clipper cluster.
    name : str
        The name to be assigned to both the registered application and deployed model.
    input_type : str
        The input_type to be associated with the registered app and deployed model.
        One of "integers", "floats", "doubles", "bytes", or "strings".
    func : function
        The prediction function. Any state associated with the function will be
        captured via closure capture and pickled with Cloudpickle.
    pytorch_model : pytorch model object
        The Pytorch model to save.
    base_image : str, optional
        The base Docker image to build the new model image from. This
        image should contain all code necessary to run a Clipper model
        container RPC client.
    labels : list(str), optional
        A list of strings annotating the model. These are ignored by Clipper
        and used purely for user annotations.
    registry : str, optional
        The Docker container registry to push the freshly built model to. Note
        that if you are running Clipper on Kubernetes, this registry must be accesible
        to the Kubernetes cluster in order to fetch the container from the registry.
    num_replicas : int, optional
        The number of replicas of the model to create. The number of replicas
        for a model can be changed at any time with
        :py:meth:`clipper.ClipperConnection.set_num_replicas`.
    batch_size : int, optional
        The user-defined query batch size for the model. Replicas of the model will attempt
        to process at most `batch_size` queries simultaneously. They may process smaller
        batches if `batch_size` queries are not immediately available.
        If the default value of -1 is used, Clipper will adaptively calculate the batch size for individual
        replicas of this model.
    pkgs_to_install : list (of strings), optional
        A list of the names of packages to install, using pip, in the container.
        The names must be strings.

    Example
    -------
    Define a pytorch nn module and save the model::
    
        from clipper_admin import ClipperConnection, DockerContainerManager
        from clipper_admin.deployers.pytorch import deploy_pytorch_model
        from torch import nn

        clipper_conn = ClipperConnection(DockerContainerManager())

        # Connect to an already-running Clipper cluster
        clipper_conn.connect()
        model = nn.Linear(1, 1)

        # Define a shift function to normalize prediction inputs
        def predict(model, inputs):
            pred = model(shift(inputs))
            pred = pred.data.numpy()
            return [str(x) for x in pred]


        deploy_pytorch_model(
            clipper_conn,
            name="example",
            version=1,
            input_type="doubles",
            func=predict,
            pytorch_model=model)
    """

    serialization_dir = save_python_function(name, func)

    # save Torch model
    torch_weights_save_loc = os.path.join(serialization_dir,
                                          PYTORCH_WEIGHTS_RELATIVE_PATH)

    torch_model_save_loc = os.path.join(serialization_dir,
                                        PYTORCH_MODEL_RELATIVE_PATH)
    print(torch_weights_save_loc)

    try:
        torch.save(pytorch_model.state_dict(), torch_weights_save_loc)
        serialized_model = serialize_object(pytorch_model)
        with open(torch_model_save_loc, "wb") as serialized_model_file:
            serialized_model_file.write(serialized_model)
        logger.warning("Torch model saved")

        py_minor_version = (sys.version_info.major, sys.version_info.minor)
        # Check if Python 2 or Python 3 image
        if base_image == "default":
            if py_minor_version < (3, 0):
                logger.warning("Using Python 2 base image")
                base_image = "pytorch"
            elif py_minor_version == (3, 5):
                logger.warning("Using Python 3.5 base image")
                base_image = "pytorch"
            elif py_minor_version == (3, 6):
                logger.warning("Using Python 3.6 base image")
                base_image = "pytorch"
            else:
                msg = (
                    "PyTorch deployer only supports Python 2.7, 3.5, and 3.6. "
                    "Detected {major}.{minor}").format(
                        major=sys.version_info.major,
                        minor=sys.version_info.minor)
                logger.error(msg)
                # Remove temp files
                shutil.rmtree(serialization_dir)
                raise ClipperException(msg)

        # Deploy model
        build_and_deploy_model(
            name, input_type, serialization_dir, base_image, labels,
            registry, num_replicas, batch_size, pkgs_to_install)

    except Exception as e:
        raise ClipperException("Error saving torch model: %s" % e)

    # Remove temp files
    shutil.rmtree(serialization_dir)

cur_dir = os.path.dirname(os.path.abspath(__file__))

def normalize(x):
    return x.astype(np.double) / 255.0

def parsedata(train_path, pos_label):
    trainData = np.genfromtxt(train_path, delimiter=',', dtype=int)
    records = trainData[:, 1:]
    labels = trainData[:, :1]
    transformedlabels = [objective(ele, pos_label) for ele in labels]
    return (records, transformedlabels)

def predict(model, xs):
    # preds = []
    # for x in xs:
    #     p = model(x).data.numpy().tolist()[0]
    #     preds.append(str(p))
    # return preds
    preds = model(xs)
    return preds

def deploy_and_test_model(model,
                          predict_fn=predict):
    deploy_pytorch_model(model_name, "integers",
                         predict_fn, model)

    time.sleep(5)

# Define a simple NN model
class BasicNN(nn.Module):
    def __init__(self):
        super(BasicNN, self).__init__()
        self.net = nn.Linear(28 * 28, 2)

    def forward(self, x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        x = x.float()
        x = Variable(x)
        x = x.view(1, 1, 28, 28)
        x = x / 255.0
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        output = self.net(x.float())
        return F.softmax(output)

# def train(model):
#     model.train()
#     optimizer = optim.SGD(model.parameters(), lr=0.001)
#     for epoch in range(10):
#         for i, d in enumerate(train_loader, 1):
#             image, j = d
#             optimizer.zero_grad()
#             output = model(image)
#             loss = F.cross_entropy(output,
#                                    Variable(
#                                        torch.LongTensor([train_y[i - 1]])))
#             loss.backward()
#             optimizer.step()
#     return model

trained_models = ['resnet18', 'alexnet', 'vgg16', 'squeezenet1_0', 'densenet161']

#for model_name in trained_models:
#    model = getattr(models, model_name)()
#    input = torch.randn(1, 3, 224, 224)
#    output = model(input)
#    print(output)

def main():
    pos_label = 3

    #train_path = os.path.join(cur_dir, "data/train.data")
    #train_x, train_y = parsedata(train_path, pos_label)
    #train_x = normalize(train_x)
    #train_loader = TrainingDataset(train_x, train_y)


    #model = BasicNN()
    model = models.densenet201()
    #nn_model = train(model)

    deploy_and_test_model(model)

if __name__ == '__main__':
	main()
