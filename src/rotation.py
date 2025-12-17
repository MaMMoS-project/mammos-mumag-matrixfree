import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation


def get_rot_matix(euler_angles: jnp.ndarray) -> jnp.ndarray:
    rotation = Rotation.from_euler("xyz", euler_angles)
    q = rotation.as_quat(scalar_first=True)
    """
    Transformation from quaternion to the corresponding rotation matrix.
    Reference: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    """
    #! this is a sclar first quaternion
    return jnp.array(
        [
            [
                q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3],
                2 * q[1] * q[2] - 2 * q[0] * q[3],
                2 * q[1] * q[3] + 2 * q[0] * q[2],
            ],
            [
                2 * q[1] * q[2] + 2 * q[0] * q[3],
                q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3],
                2 * q[2] * q[3] - 2 * q[0] * q[1],
            ],
            [
                2 * q[1] * q[3] - 2 * q[0] * q[2],
                2 * q[2] * q[3] + 2 * q[0] * q[1],
                q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3],
            ],
        ]
    )
