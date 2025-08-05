import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image  # 替代 scipy.misc.toimage（已弃用）

class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        with self.writer.as_default():
            tf.summary.scalar(name=tag, data=value, step=step)
            self.writer.flush()

    def image_summary(self, tag, images, step, max_outputs=3):
        """Log a list of images (H, W, C) in [0,255] or [0,1]."""
        with self.writer.as_default():
            # Convert images to float32 in [0,1] if necessary
            images = np.array(images)
            if images.dtype != np.float32:
                images = images.astype(np.float32) / 255.0
            tf.summary.image(name=tag, data=images, step=step, max_outputs=max_outputs)
            self.writer.flush()
