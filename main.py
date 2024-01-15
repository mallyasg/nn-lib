from absl import app
from absl import flags
from absl import logging
import jax
from clu import platform
from ml_collections import config_flags

from trainer import train

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store training artifacts')
config_flags.DEFINE_config_file(
    'model_config',
    None,
    'File path containing model name and its associated hyperparameters',
    lock_config=True)
config_flags.DEFINE_config_file(
    'data_config',
    None,
    'File path containing data for training and its associated metadata',
    lock_config=True)


def main(argv):
  logging.info(f'JAX process: {jax.process_index()} / {jax.process_count()}')
  logging.info(f'JAX local devices: {jax.local_devices()}')

  platform.work_unit().set_task_status(f'process_index: {jax.process_index()}, '
                                       f'process_count: {jax.process_count()}')

  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       FLAGS.workdir, 'workdir')

  train.train_and_evaluate(FLAGS.model_config, FLAGS.data_config, FLAGS.workdir)


if __name__ == '__main__':
  flags.mark_flags_as_required(['model_config', 'data_config', 'workdir'])
  app.run(main)
