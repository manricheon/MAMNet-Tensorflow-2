import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from model.common import *

class Trainer:
    def __init__(self,
                 model,
                 loss,
                 learning_rate,
                 checkpoint_path,
                 args):

        self.now = None
        self.loss = loss
        self.learning_rate = learning_rate
        self.args = args

        if checkpoint_path:
            self.ckpt_path = checkpoint_path
        else:
            self.ckpt_path = os.path.join(self.args.exp_dir, self.args.exp_name, 'ckpt')        

        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              psnr=tf.Variable(-1.0),
                                              optimizer=Adam(learning_rate),
                                              model=model)

        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                         directory=self.ckpt_path,
                                         max_to_keep=100)

        self.restore(self.ckpt_path)


    @property
    def model(self):
        return self.checkpoint.model

    def train(self, train_dataset, valid_dataset, save_best_only=False):
        loss_mean = Mean()

        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint

        self.now = time.perf_counter()

        for lr, hr in train_dataset.take(self.args.num_iter - ckpt.step.numpy()):
            ckpt.step.assign_add(1)
            step = ckpt.step.numpy()

            loss = self.train_step(lr, hr)
            loss_mean(loss)

            loss_value = loss_mean.result()
            loss_mean.reset_states()

            lr_value = ckpt.optimizer._decayed_lr('float32').numpy()

            duration = time.perf_counter() - self.now                
            self.now = time.perf_counter()               
            
            if step % self.args.log_freq == 0:
                tf.summary.scalar('loss', loss_value, step=step)
                tf.summary.scalar('lr', lr_value, step=step)

            if step % self.args.print_freq == 0:
                 print(f'{step}/{self.args.num_iter}: loss = {loss_value.numpy():.3f} , lr = {lr_value:.6f} ({duration:.2f}s)')

            if step % self.args.valid_freq == 0:
                psnr_value = self.evaluate(valid_dataset)
                ckpt.psnr = psnr_value
                tf.summary.scalar('psnr', psnr_value, step=step)

                print(f'{step}/{self.args.num_iter}: loss = {loss_value.numpy():.3f}, lr = {lr_value:.6f}, PSNR = {psnr_value.numpy():3f}')

            if step % self.args.save_freq == 0:
                # save weights only
                save_path = self.ckpt_path + '/weights-' + str(step) + '.h5'
                self.checkpoint.model.save_weights(filepath=save_path, save_format='h5')

                # save ckpt (weights + other train status)
                ckpt_mgr.save(checkpoint_number=step)

            

    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.checkpoint.model(lr, training=True)
            loss_value = self.loss(hr, sr)

        gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
        self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))

        return loss_value

    def evaluate(self, dataset):
        return evaluate_chop(self.checkpoint.model, dataset)

    def restore(self, ckpt_path):
        if os.path.isdir(ckpt_path) is False:
            self.checkpoint.restore(ckpt_path).expect_partial()
            print(f'Model restored from checkpoint path {ckpt_path}')
        else:
            if self.checkpoint_manager.latest_checkpoint:
                self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
                print(f'Model checkpoint restored at step {self.checkpoint.step.numpy()}')
            else:
                print('No Model restored')


class MAMNetTrainer(Trainer):
    def __init__(self,
                 model,
                 ckpt_path,
                 args):
        super().__init__(model=model,
                         loss=MeanAbsoluteError(),
                         learning_rate=ExponentialDecay(args.lr_init, args.lr_decay_step, args.lr_decay_ratio, staircase=True),
                         checkpoint_path=ckpt_path,
                         args=args)

    def train(self, train_dataset, valid_dataset, save_best_only=True):
        super().train(train_dataset, valid_dataset, save_best_only)