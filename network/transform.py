from __future__ import division
import tensorflow as tf
import tflearn
import numpy as np


def get_coef(u):
    return tf.stack([((1 - u) ** 3) / 6, (3 * (u ** 3) - 6 * (u ** 2) + 4) / 6,
                     (-3 * (u ** 3) + 3 * (u ** 2) + 3 * u + 1) / 6, (u ** 3) / 6], axis=1)


def sample_power(lo, hi, k, size=None):
    r = (hi - lo) / 2
    center = (hi + lo) / 2
    r = r ** (1 / k)
    points = (tf.random_uniform(size, dtype=tf.float32) - 0.5) * 2 * r
    points = (tf.abs(points) ** k) * tf.sign(points)
    return points + center


def pad_3d(mat, pad):
    return tf.pad(mat, [[0, 0], [pad, pad], [pad, pad], [pad, pad], [0, 0]], "CONSTANT")


def resize_linear(target_shape, control_fields):
    _, n, m, t, _ = control_fields.shape.as_list()
    assert n == m and m == t
    assert target_shape % n == 0
    factor = target_shape // n
    ret_n = target_shape

    def interpolate_axis(arr):
        ret = arr
        expand_shape = ret.shape.as_list()
        expand_shape[3] = (expand_shape[3] + 1) * factor
        expand_points_l = tf.reshape(
            tf.tile(tf.expand_dims(
                tf.concat([ret[:, :, :, 0:1, :], ret[:, :, :, :-1, :], ret[:, :, :, -2:-1, :]], axis=3), 4),
                [1, 1, 1, 1, factor, 1]), [-1] + list(expand_shape[1:]))
        expand_points_r = tf.reshape(
            tf.tile(tf.expand_dims(
                tf.concat([ret[:, :, :, 1:2, :], ret[:, :, :, 1:, :], ret[:, :, :, -1:, :]], axis=3), 4),
                [1, 1, 1, 1, factor, 1]), [-1] + list(expand_shape[1:]))
        d = factor // 2
        u = np.zeros(ret_n)
        for i in range(ret_n):
            p = min(max(0, (i - d) // factor), n - 2)
            u[i] = (p + 1.5) - (i + 0.5) / factor
        u = u.reshape((1, 1, 1, ret_n, 1))
        ret = u * expand_points_l[:, :, :, d: d + ret_n] + \
            (1 - u) * expand_points_r[:, :, :, d: d + ret_n]
        return ret

    ret = interpolate_axis(tf.transpose(control_fields, [0, 1, 3, 2, 4]))
    ret = interpolate_axis(tf.transpose(ret, [0, 3, 2, 1, 4]))
    ret = interpolate_axis(tf.transpose(ret, [0, 3, 1, 2, 4]))
    return ret


def meshgrids(shape, flatten=True, name=None):
    with tf.name_scope(name, "meshgrid", [shape]):
        indices_x = tf.range(0, shape[1])
        indices_y = tf.range(0, shape[2])
        indices_z = tf.range(0, shape[3])
        indices = tf.stack(tf.meshgrid(indices_x, indices_y,
                                       indices_z, indexing='ij'), axis=-1)
        indices = tf.tile(tf.expand_dims(indices, axis=0),
                          tf.stack([shape[0], 1, 1, 1, 1]))
        indices = tf.cast(indices, tf.float32)
        if flatten:
            return tf.reshape(indices, tf.stack([shape[0], shape[1] * shape[2] * shape[3], 3]))
        else:
            return indices


def meshgrids_like(tensor, flatten=True, name=None):
    return meshgrids(tf.shape(tensor), flatten, name)


def warp_points(flow_fields, pts):
    '''
    Arguments
    ----------------
    flow_fields : [batch, X, Y, Z, 3]
    pts: [batch, 6, 3] 
    '''
    moving_pts = meshgrids_like(flow_fields, flatten=False) + flow_fields
    shape = tf.shape(flow_fields)
    moving_pts = tf.reshape(moving_pts, tf.stack(
        [shape[0], shape[1] * shape[2] * shape[3], 1, 3]))
    distance = tf.sqrt(tf.reduce_sum(
        (moving_pts - tf.expand_dims(pts, axis=1)) ** 2, axis=-1))
    closest = tf.cast(tf.argmin(distance, axis=1), tf.int32)
    fixed_pts = tf.stack([tf.div(closest, shape[2] * shape[3]), tf.mod(
        tf.div(closest, shape[3]), shape[2]), tf.mod(closest, shape[3])], axis=2)
    return fixed_pts


def free_form_fields(shape, control_fields, padding='same'):
    '''Calculate a flow fields based on 3-order B-Spline interpolation of control points.

    Arguments
    --------------
    shape : list of 3 integers, flow field shape `(x, y, z)`
    control_fields : 5d tensor with 3 channels `(batch_size, n, m, t, 3)`

    Output
    --------------
    5d tensor with 3 channels `(batch_size, x, y, z, 3)`
    '''
    interpolate_range = 4

    control_fields = tf.convert_to_tensor(control_fields, dtype=tf.float32)
    _, n, m, t, _ = control_fields.shape.as_list()
    if padding == 'same':
        control_fields = pad_3d(control_fields, 1)
    elif padding == 'valid':
        n -= 2
        m -= 2
        t -= 2
    control_fields = tf.reshape(tf.transpose(
        control_fields, (1, 2, 3, 0, 4)), [n + 2, m + 2, t + 2, -1])

    assert shape[0] % (n - 1) == 0
    s_x = shape[0] // (n - 1)
    u_x = (tf.range(0, s_x, dtype=tf.float32) + 0.5) / s_x  # s_x
    coef_x = get_coef(u_x)  # (s_x, 4)

    shape_cf = control_fields.shape.as_list()
    flow = tf.concat([tf.matmul(coef_x,
                                tf.reshape(control_fields[i: i + interpolate_range], [interpolate_range, -1]))
                      for i in range(0, n - 1)],
                     axis=0)

    assert shape[1] % (m - 1) == 0
    s_y = shape[1] // (m - 1)
    u_y = (tf.range(0, s_y, dtype=tf.float32) + 0.5) / s_y  # s_y
    coef_y = get_coef(u_y)  # (s_y, 4)

    flow = tf.reshape(tf.transpose(flow), [shape_cf[1], -1])
    flow = tf.concat([tf.matmul(coef_y,
                                tf.reshape(flow[i: i + interpolate_range], [interpolate_range, -1]))
                      for i in range(0, m - 1)],
                     axis=0)
    # print(flow.shape)
    assert shape[2] % (t - 1) == 0
    s_z = shape[2] // (t - 1)
    u_z = (tf.range(0, s_z, dtype=tf.float32) + 0.5) / s_z  # s_y
    coef_z = get_coef(u_z)  # (s_y, 4)

    flow = tf.reshape(tf.transpose(flow), [shape_cf[2], -1])
    flow = tf.concat([tf.matmul(coef_z,
                                tf.reshape(flow[i: i + interpolate_range], [interpolate_range, -1]))
                      for i in range(0, t - 1)],
                     axis=0)
    # print(flow.shape)
    flow = tf.reshape(flow, [shape[2], -1, 3, shape[1], shape[0]])
    flow = tf.transpose(flow, [1, 4, 3, 0, 2])
    return flow
