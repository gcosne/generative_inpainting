""" common model for DCGAN """
import logging

import cv2
import neuralgym as ng
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope

from neuralgym.models import Model
from neuralgym.ops.summary_ops import scalar_summary, images_summary
from neuralgym.ops.layers import resize

from inpaint_ops import conv2d_sn, gan_hinge_loss
from inpaint_ops import gated_conv, gated_deconv, gen_conv
from inpaint_ops import random_mask
from inpaint_ops import resize_mask_like, contextual_attention


logger = logging.getLogger()


class InpaintCAModel(Model):
    def __init__(self):
        super().__init__('InpaintCAModel')

    def build_inpaint_net(self, x, mask, config=None, reuse=False,
                          training=True, padding='SAME', name='inpaint_net'):
        """Inpaint network.

        Args:
            x: incomplete image, [-1, 1]
            mask: mask region {0, 1}
        Returns:
            [-1, 1] as predicted image
        """
        xin = x
        offset_flow = None
        ones_x = tf.ones_like(x)[:, :, :, 0:1]
        x = tf.concat([x, ones_x, ones_x*mask], axis=3)

        # two stage network
        cnum = 24
        with tf.variable_scope(name, reuse=reuse), \
                arg_scope([gated_conv, gated_deconv, gen_conv],
                          training=training, padding=padding):
            # stage1
            x = gated_conv(x, cnum, 5, 1, name='conv1')
            x = gated_conv(x, 2*cnum, 3, 2, name='conv2_downsample')
            x = gated_conv(x, 2*cnum, 3, 1, name='conv3')
            x = gated_conv(x, 4*cnum, 3, 2, name='conv4_downsample')
            x = gated_conv(x, 4*cnum, 3, 1, name='conv5')
            x = gated_conv(x, 4*cnum, 3, 1, name='conv6')
            mask_s = resize_mask_like(mask, x)
            x = gated_conv(x, 4*cnum, 3, rate=2, name='conv7_atrous')
            x = gated_conv(x, 4*cnum, 3, rate=4, name='conv8_atrous')
            x = gated_conv(x, 4*cnum, 3, rate=8, name='conv9_atrous')
            x = gated_conv(x, 4*cnum, 3, rate=16, name='conv10_atrous')
            x = gated_conv(x, 4*cnum, 3, 1, name='conv11')
            x = gated_conv(x, 4*cnum, 3, 1, name='conv12')
            x = gated_deconv(x, 2*cnum, name='conv13_upsample')
            x = gated_conv(x, 2*cnum, 3, 1, name='conv14')
            x = gated_deconv(x, cnum, name='conv15_upsample')
            x = gated_conv(x, cnum//2, 3, 1, name='conv16')
            x = gen_conv(x, 3, 3, 1, activation=tf.nn.tanh, name='conv17')
            x_stage1 = x
            # return x_stage1, None, None

            # stage2, paste result as input
            # x = tf.stop_gradient(x)
            x = x*mask + xin*(1.-mask)
            x.set_shape(xin.get_shape().as_list())
            xnow = tf.concat([x, ones_x, ones_x*mask], axis=3)
            # conv branch
            x = gated_conv(xnow, cnum, 5, 1, name='xconv1')
            x = gated_conv(x, cnum, 3, 2, name='xconv2_downsample')
            x = gated_conv(x, 2*cnum, 3, 1, name='xconv3')
            x = gated_conv(x, 2*cnum, 3, 2, name='xconv4_downsample')
            x = gated_conv(x, 4*cnum, 3, 1, name='xconv5')
            x = gated_conv(x, 4*cnum, 3, 1, name='xconv6')
            x = gated_conv(x, 4*cnum, 3, rate=2, name='xconv7_atrous')
            x = gated_conv(x, 4*cnum, 3, rate=4, name='xconv8_atrous')
            x = gated_conv(x, 4*cnum, 3, rate=8, name='xconv9_atrous')
            x = gated_conv(x, 4*cnum, 3, rate=16, name='xconv10_atrous')
            x_hallu = x
            # attention branch
            x = gated_conv(xnow, cnum, 5, 1, name='pmconv1')
            x = gated_conv(x, cnum, 3, 2, name='pmconv2_downsample')
            x = gated_conv(x, 2*cnum, 3, 1, name='pmconv3')
            x = gated_conv(x, 4*cnum, 3, 2, name='pmconv4_downsample')
            x = gated_conv(x, 4*cnum, 3, 1, name='pmconv5')
            x = gated_conv(x, 4*cnum, 3, 1, name='pmconv6',
                           activation=tf.nn.relu)
            x, offset_flow = contextual_attention(x, x, mask_s, 3, 1, rate=2)
            x = gated_conv(x, 4*cnum, 3, 1, name='pmconv9')
            x = gated_conv(x, 4*cnum, 3, 1, name='pmconv10')
            pm = x
            x = tf.concat([x_hallu, pm], axis=3)
            # upsample
            x = gated_conv(x, 4*cnum, 3, 1, name='allconv11')
            x = gated_conv(x, 4*cnum, 3, 1, name='allconv12')
            x = gated_deconv(x, 2*cnum, name='allconv13_upsample')
            x = gated_conv(x, 2*cnum, 3, 1, name='allconv14')
            x = gated_deconv(x, cnum, name='allconv15_upsample')
            x = gated_conv(x, cnum//2, 3, 1, name='allconv16')
            x = gen_conv(x, 3, 3, 1, activation=tf.nn.tanh, name='allconv17')
            x_stage2 = x
        return x_stage1, x_stage2, offset_flow

    def build_sn_patch_gan_discriminator(self, x, mask,
                                         reuse=False, training=True):
        ones_x = tf.ones_like(x)[:, :, :, 0:1]
        x = tf.concat([x, ones_x*mask], axis=3)
        with tf.variable_scope('discriminator', reuse=reuse):
            cnum = 64
            x = conv2d_sn(x, cnum, 5, 1, name='sn_conv1')
            x = conv2d_sn(x, cnum*2, 5, 2, name='sn_conv2')
            x = conv2d_sn(x, cnum*4, 5, 2, name='sn_conv3')
            x = conv2d_sn(x, cnum*4, 5, 2, name='sn_conv4')
            x = conv2d_sn(x, cnum*4, 5, 2, name='sn_conv5')
            x = conv2d_sn(x, cnum*4, 5, 2, name='sn_conv6')
            return x

    def build_graph_with_losses(self, batch_data, config, training=True,
                                summary=False, reuse=False, batch_mask=None):
        
        #print('INPAINT MODEL')
        #print('config.CUSTOM_MASK = ',config.CUSTOM_MASK)
        batch_pos = batch_data / 127.5 - 1.
        #print('This is the batch_pos shape',batch_pos.shape)
        
        if batch_mask == None:
            # generate mask, 1 represents masked point
            mask = random_mask(config)
            # print('THIS IS RANDOM_MASK_SHAPE',mask.shape)
        else:
            channels = tf.unstack (batch_mask, axis=-1)
            mask    = channels[0]
            mask    = tf.expand_dims(mask,axis = -1)
            
        batch_incomplete = batch_pos*(1.-mask)
        # inpaint
        x1, x2, offset_flow = self.build_inpaint_net(
            batch_incomplete, mask, config, reuse=reuse,
            training=training, padding=config.PADDING)

        if config.PRETRAIN_COARSE_NETWORK:
            batch_predicted = x1
            logger.info('Set batch_predicted to x1.')
        else:
            batch_predicted = x2
            logger.info('Set batch_predicted to x2.')

        losses = {}
        # apply mask and complete image
        batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask)

        losses['l1_loss'] = config.COARSE_L1_ALPHA * tf.reduce_mean(tf.abs(batch_pos - x1))
        losses['l1_loss'] += tf.reduce_mean(tf.abs(batch_pos - x2))

        if summary:
            scalar_summary('losses/l1_loss', losses['l1_loss'])
            viz_img = [batch_pos, batch_incomplete, batch_complete]
            if offset_flow is not None:
                viz_img.append(
                    resize(offset_flow, scale=4,
                           func=tf.image.resize_nearest_neighbor))
            images_summary(
                tf.concat(viz_img, axis=2),
                'raw_incomplete_predicted_complete', config.VIZ_MAX_OUT)

        # gan
        if config.GAN == 'sn_patch_gan':
            # fake
            Dsn_Gz = self.build_sn_patch_gan_discriminator(
                batch_complete, mask, training=training, reuse=tf.AUTO_REUSE)
            # real
            Dsn_x = self.build_sn_patch_gan_discriminator(
                batch_pos, mask, training=training, reuse=tf.AUTO_REUSE)
            g_loss, d_loss = gan_hinge_loss(Dsn_x, Dsn_Gz)
            losses['g_loss'] = config.GAN_LOSS_ALPHA * g_loss
            losses['g_loss'] += config.L1_LOSS_ALPHA * losses['l1_loss']
            losses['d_loss'] = d_loss
            scalar_summary('losses/g_loss', losses['g_loss'])

        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'inpaint_net')
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        return g_vars, d_vars, losses

    def build_infer_graph(self, batch_data, config, name='val',mask=None):
        """
        """
        if mask is None:
            mask = random_mask(config, name=name+'mask_c')
            
        batch_pos = batch_data / 127.5 - 1.
        edges = None

        batch_incomplete = batch_pos*(1.-mask)
        # inpaint
        x1, x2, offset_flow = self.build_inpaint_net(
            batch_incomplete, mask, config, reuse=True,
            training=False, padding=config.PADDING)

        if config.PRETRAIN_COARSE_NETWORK:
            batch_predicted = x1
            logger.info('Set batch_predicted to x1.')
        else:
            batch_predicted = x2
            logger.info('Set batch_predicted to x2.')

        # apply mask and reconstruct
        batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask)

        # global image visualization
        viz_img = [batch_pos, batch_incomplete, batch_complete]
        if offset_flow is not None:
            viz_img.append(
                resize(offset_flow, scale=4,
                       func=tf.image.resize_nearest_neighbor))
        images_summary(
            tf.concat(viz_img, axis=2),
            name+'_raw_incomplete_complete', config.VIZ_MAX_OUT)

        return batch_complete

    def build_static_infer_graph(self, batch_data, config, name,mask=None):
        """
        """
        # generate mask, 1 represents masked point
        return self.build_infer_graph(batch_data, config, name,mask)

    def build_server_graph(self, batch_data, reuse=False, is_training=False):
        """
        """
        # generate mask, 1 represents masked point
        batch_raw, masks_raw = tf.split(batch_data, 2, axis=2)
        masks = tf.cast(masks_raw[0:1, :, :, 0:1] > 127.5, tf.float32)

        batch_pos = batch_raw / 127.5 - 1.
        batch_incomplete = batch_pos * (1. - masks)
        # inpaint
        x1, x2, flow = self.build_inpaint_net(
            batch_incomplete, masks, reuse=reuse, training=is_training,
            config=None)
        batch_predict = x2
        # apply mask and reconstruct
        batch_complete = batch_predict*masks + batch_incomplete*(1-masks)
        return batch_complete