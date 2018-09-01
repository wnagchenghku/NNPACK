import torch
import docker
import os
import sys
import logging
import torchvision.models as models
import cloud
from cloudpickle import CloudPickler

PYTORCH_WEIGHTS_RELATIVE_PATH = "pytorch_weights.pkl"
PYTORCH_MODEL_RELATIVE_PATH = "pytorch_model.pkl"

trained_models = ['resnet18', 'densenet201']

logger = logging.getLogger(__name__)

class ClipperException(Exception):
    """A generic exception indicating that Clipper encountered a problem."""

    def __init__(self, msg, *args):
        self.msg = msg
        super(Exception, self).__init__(msg, *args)

if sys.version_info < (3, 0):
    try:
        from cStringIO import StringIO
    except ImportError:
        from StringIO import StringIO
    PY3 = False
else:
    from io import BytesIO as StringIO
    PY3 = True

def serialize_object(obj):
    s = StringIO()
    c = CloudPickler(s, 2)
    c.dump(obj)
    return s.getvalue()

def build_model(name,
	            model_data_path,
                base_image,
                container_registry=None,
                pkgs_to_install=None):

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
        if container_registry is not None:
            image = "{reg}/{image}".format(
                reg=container_registry, image=image)
        docker_client = docker.from_env()
        logger.info("Building model Docker image with model data from {}".format(model_data_path))
        image_result, build_logs = docker_client.images.build(
            fileobj=context_file, custom_context=True, tag=image)
        for b in build_logs:
            if 'stream' in b and b['stream'] != '\n':  #log build steps only
                logger.info(b['stream'].rstrip())

    return image

def build_and_deploy_model(name,
                           input_type,
                           model_data_path,
                           base_image,
                           labels=None,
                           container_registry=None,
                           pkgs_to_install=None):

    image = build_model(name, model_data_path, base_image,
                        container_registry, pkgs_to_install)

def save_python_function(name, func):
    predict_fname = "func.pkl"

    # Serialize function
    s = StringIO()
    c = CloudPickler(s, 2)
    c.dump(func)
    serialized_prediction_function = s.getvalue()

    # Set up serialization directory
    # serialization_dir = os.path.abspath(tempfile.mkdtemp(suffix='clipper'))
    serialization_dir = "model/{}/".format(name)
    os.makedirs(serialization_dir)
    logger.info("Saving function to {}".format(serialization_dir))

    # Write out function serialization
    func_file_path = os.path.join(serialization_dir, predict_fname)
    if sys.version_info < (3, 0):
        with open(func_file_path, "w") as serialized_function_file:
            serialized_function_file.write(serialized_prediction_function)
    else:
        with open(func_file_path, "wb") as serialized_function_file:
            serialized_function_file.write(serialized_prediction_function)
    logging.info("Serialized and supplied predict function")
    return serialization_dir

def deploy_pytorch_model(name,
                         func,
                         pytorch_model):
    serialization_dir = save_python_function(name, func)

    # save Torch model
    torch_weights_save_loc = os.path.join(serialization_dir,
                                          PYTORCH_WEIGHTS_RELATIVE_PATH)

    torch_model_save_loc = os.path.join(serialization_dir,
                                        PYTORCH_MODEL_RELATIVE_PATH)

    try:
        torch.save(pytorch_model.state_dict(), torch_weights_save_loc)
        serialized_model = serialize_object(pytorch_model)
        with open(torch_model_save_loc, "wb") as serialized_model_file:
            serialized_model_file.write(serialized_model)

        # Deploy model
        # build_and_deploy_model(
        #     name, version, input_type, serialization_dir, base_image, labels,
        #     registry, pkgs_to_install)

    except Exception as e:
        raise ClipperException("Error saving torch model: %s" % e)

def predict(model, xs):
    # preds = []
    # for x in xs:
    #     p = model(x).data.numpy().tolist()[0]
    #     preds.append(str(p))
    # return preds
    preds = model(xs)
    return preds

def deploy_and_test_model(model,
                          model_name,
                          predict_fn=predict):
    deploy_pytorch_model(model_name, predict_fn, model)

def main():

	for model_name in trained_models:
		deploy_and_test_model(getattr(models, model_name)(), model_name)

if __name__ == '__main__':
	main()