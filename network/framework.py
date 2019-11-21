import tensorflow as tf
import numpy as np
import tflearn
from tqdm import tqdm

from . import transform
from .utils import MultiGPUs
from .spatial_transformer import Dense3DSpatialTransformer, Fast3DTransformer
from .recursive_cascaded_networks import RecursiveCascadedNetworks


def set_tf_keys(feed_dict, **kwargs):
    ret = dict([(k + ':0', v) for k, v in feed_dict.items()])
    ret.update([(k + ':0', v) for k, v in kwargs.items()])
    return ret


def masked_mean(arr, mask):
    return tf.reduce_sum(arr * mask) / (tf.reduce_sum(mask) + 1e-9)


class FrameworkUnsupervised:
    net_args = {'class': RecursiveCascadedNetworks}
    framework_name = 'gaffdfrm'

    def __init__(self, devices, image_size, segmentation_class_value, validation=False, fast_reconstruction=False):
        network_class = self.net_args.get('class', RecursiveCascadedNetworks)
        self.summaryType = self.net_args.pop('summary', 'basic')

        self.reconstruction = Fast3DTransformer() if fast_reconstruction else Dense3DSpatialTransformer()

        # input place holder
        img1 = tf.placeholder(dtype=tf.float32, shape=[
                              None, 128, 128, 128, 1], name='voxel1')
        img2 = tf.placeholder(dtype=tf.float32, shape=[
                              None, 128, 128, 128, 1], name='voxel2')
        seg1 = tf.placeholder(dtype=tf.float32, shape=[
                              None, 128, 128, 128, 1], name='seg1')
        seg2 = tf.placeholder(dtype=tf.float32, shape=[
                              None, 128, 128, 128, 1], name='seg2')
        point1 = tf.placeholder(dtype=tf.float32, shape=[
                                None, 6, 3], name='point1')
        point2 = tf.placeholder(dtype=tf.float32, shape=[
                                None, 6, 3], name='point2')

        bs = tf.shape(img1)[0]
        augImg1, preprocessedImg2 = img1 / 255.0, img2 / 255.0

        aug = self.net_args.pop('augmentation', None)
        if aug is None:
            imgs = img1.shape.as_list()[1:4]
            control_fields = transform.sample_power(
                -0.4, 0.4, 3, tf.stack([bs, 5, 5, 5, 3])) * (np.array(imgs) // 4)
            augFlow = transform.free_form_fields(imgs, control_fields)

            def augmentation(x):
                return tf.cond(tflearn.get_training_mode(), lambda: self.reconstruction([x, augFlow]),
                               lambda: x)

            def augmenetation_pts(incoming):
                def aug(incoming):
                    aug_pt = tf.cast(transform.warp_points(
                        augFlow, incoming), tf.float32)
                    pt_mask = tf.cast(tf.reduce_all(
                        incoming >= 0, axis=-1, keep_dims=True), tf.float32)
                    return aug_pt * pt_mask - (1 - pt_mask)
                return tf.cond(tflearn.get_training_mode(), lambda: aug(incoming), lambda: incoming)
            
            augImg2 = augmentation(preprocessedImg2)
            augSeg2 = augmentation(seg2)
            augPt2 = augmenetation_pts(point2)
        elif aug == 'identity':
            augFlow = tf.zeros(
                tf.stack([tf.shape(img1)[0], 128, 128, 128, 3]), dtype=tf.float32)
            augImg2 = preprocessedImg2
            augSeg2 = seg2
            augPt2 = point2
        else:
            raise NotImplementedError('Augmentation {}'.format(aug))

        learningRate = tf.placeholder(tf.float32, [], 'learningRate')
        if not validation:
            adamOptimizer = tf.train.AdamOptimizer(learningRate)

        self.segmentation_class_value = segmentation_class_value
        self.network = network_class(
            self.framework_name, framework=self, fast_reconstruction=fast_reconstruction, **self.net_args)
        net_pls = [augImg1, augImg2, seg1, augSeg2, point1, augPt2]
        if devices == 0:
            with tf.device("/cpu:0"):
                self.predictions = self.network(*net_pls)
                if not validation:
                    self.adamOpt = adamOptimizer.minimize(
                        self.predictions["loss"])
        else:
            gpus = MultiGPUs(devices)
            if validation:
                self.predictions = gpus(self.network, net_pls)
            else:
                self.predictions, self.adamOpt = gpus(
                    self.network, net_pls, opt=adamOptimizer)
        self.build_summary(self.predictions)

    @property
    def data_args(self):
        return self.network.data_args

    def build_summary(self, predictions):
        self.loss = tf.reduce_mean(predictions['loss'])
        for k in predictions:
            if k.find('loss') != -1:
                tf.summary.scalar(k, tf.reduce_mean(predictions[k]))
        self.summaryOp = tf.summary.merge_all()

        if self.summaryType == 'full':
            tf.summary.scalar('dice_score', tf.reduce_mean(
                self.predictions['dice_score']))
            tf.summary.scalar('landmark_dist', masked_mean(
                self.predictions['landmark_dist'], self.predictions['pt_mask']))
            preds = tf.reduce_sum(
                tf.cast(self.predictions['jacc_score'] > 0, tf.float32))
            tf.summary.scalar('jacc_score', tf.reduce_sum(
                self.predictions['jacc_score']) / (preds + 1e-8))
            self.summaryExtra = tf.summary.merge_all()
        else:
            self.summaryExtra = self.summaryOp

    def get_predictions(self, *keys):
        return dict([(k, self.predictions[k]) for k in keys])

    def validate_clean(self, sess, generator, keys=None):
        for fd in generator:
            _ = fd.pop('id1')
            _ = fd.pop('id2')
            _ = sess.run(self.get_predictions(*keys),
                         feed_dict=set_tf_keys(fd))

    def validate(self, sess, generator, keys=None, summary=False, predict=False, show_tqdm=False):
        if keys is None:
            keys = ['dice_score', 'landmark_dist', 'pt_mask', 'jacc_score']
            # if self.segmentation_class_value is not None:
            #     for k in self.segmentation_class_value:
            #         keys.append('jacc_{}'.format(k))
        full_results = dict([(k, list()) for k in keys])
        if not summary:
            full_results['id1'] = []
            full_results['id2'] = []
            if predict:
                full_results['seg1'] = []
                full_results['seg2'] = []
                full_results['img1'] = []
                full_results['img2'] = []
        tflearn.is_training(False, sess)
        if show_tqdm:
            generator = tqdm(generator)
        for fd in generator:
            id1 = fd.pop('id1')
            id2 = fd.pop('id2')
            results = sess.run(self.get_predictions(
                *keys), feed_dict=set_tf_keys(fd))
            if not summary:
                results['id1'] = id1
                results['id2'] = id2
                if predict:
                    results['seg1'] = fd['seg1']
                    results['seg2'] = fd['seg2']
                    results['img1'] = fd['voxel1']
                    results['img2'] = fd['voxel2']
            mask = np.where([i and j for i, j in zip(id1, id2)])
            for k, v in results.items():
                full_results[k].append(v[mask])
        if 'landmark_dist' in full_results and 'pt_mask' in full_results:
            pt_mask = full_results.pop('pt_mask')
            full_results['landmark_dist'] = [arr * mask for arr,
                                             mask in zip(full_results['landmark_dist'], pt_mask)]
        for k in full_results:
            full_results[k] = np.concatenate(full_results[k], axis=0)
            if summary:
                full_results[k] = full_results[k].mean()

        return full_results
