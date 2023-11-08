"""Utilities for affine transformations.

One day it will be a standalone module maybe. ---------------------------
"""
import numpy

# ############################
# BASIC AFFINE TRANSFORMATIONS
# ############################

# #
# Translation
# #

def get_affine_3d_translation(translation: numpy.ndarray) -> numpy.ndarray(shape=(4, 4)):
    """Get the affine matrix for a translation.

    Parameters
    ----------
    translation : numpy.ndarray
        A 3D vector of the translation.
    
    Returns
    -------
    numpy.ndarray
        The affine matrix for the translation.
    """
    affine = numpy.eye(4)
    affine[0:3, 3] = translation
    return affine

# #
# Mirror
# #

def get_affine_3d_mirror_plane(plane_normal: numpy.ndarray, plane_normal_source: numpy.ndarray) -> numpy.ndarray(shape=(4, 4)):
    """Get the affine matrix for a mirroring with respect to q plane.
    
    Parameters
    ----------
    plane_normal : numpy.ndarray
        A 3D (unit) vector describing the normal of the plane.
    plane_normal_source : numpy.ndarray
        A 3D vector describing the point source of the normal.
        The plane will therefore pass through this point and 
        be orthogonal to the normal.

    Returns
    -------
    numpy.ndarray
        The affine matrix for the mirroring.
    """
    # Transform to unit vector
    plane_normal = plane_normal.astype(numpy.float32)
    plane_normal_source = plane_normal_source.astype(numpy.float32)
    if numpy.linalg.norm(plane_normal) == 0:
        raise ValueError("Plane normal cannot be zero vector.")
    plane_normal /= numpy.linalg.norm(plane_normal)
    # Translate to origin from center of rotation
    translation_matrix = get_affine_3d_translation(-plane_normal_source)
    # Mirror
    mirror_matrix = numpy.eye(4)
    mirror_matrix[0:3, 0:3] -= 2 * numpy.outer(plane_normal, plane_normal)
    # Translate back to center of rotation
    translation_matrix2 = get_affine_3d_translation(plane_normal_source)
    # Combine
    final_affine = numpy.matmul(numpy.matmul(translation_matrix2, mirror_matrix), translation_matrix)
    return final_affine


# #
# Scale
# #

def get_affine_3d_scale(scale: numpy.ndarray) -> numpy.ndarray(shape=(4, 4)):
    """Get the affine matrix for a 3D scaling.

    Parameters
    ----------
    scale : numpy.ndarray
        A 3D vector describing the scaling along each axis.

    Returns
    -------
    numpy.ndarray
        The affine matrix for the scaling.
    """
    affine = numpy.eye(4)
    affine[0, 0] = scale[0]
    affine[1, 1] = scale[1]
    affine[2, 2] = scale[2]
    return affine


# #
# Rotation
# #

def get_affine_3d_rotation_around_axis(axis_of_rotation: numpy.ndarray, rotation: float, rotation_units: str = "rad") -> numpy.ndarray(shape=(4, 4)):
    """Get the rotation matrix for a 3D rotation around an axis that passes through the origin.

    Parameters
    ----------
    rotation : float
        The angle of rotation (in radians by default).
    axis_of_rotation : numpy.ndarray
        A 3D (unit) vector describing the axis of rotation.
    rotation_units : str, optional
        The units of rotation. Either "rad" or "deg".
        Defaults to "rad".

    Returns
    -------
    numpy.ndarray
        The rotation matrix.
    """
    # Convert to radians if necessary
    if rotation_units == "deg":
        rotation = numpy.deg2rad(rotation)
    # Transform to unit vector
    axis_of_rotation = axis_of_rotation.astype(numpy.float32)
    if numpy.linalg.norm(axis_of_rotation) == 0:
        raise ValueError("Axis of rotation cannot be zero vector.")
    axis_of_rotation /= numpy.linalg.norm(axis_of_rotation)
    # Make matrix
    x = axis_of_rotation[0]
    y = axis_of_rotation[1]
    z = axis_of_rotation[2]
    c = numpy.cos(rotation)
    s = numpy.sin(rotation)
    rotation_matrix = numpy.array([
        [x*x*(1-c)+c,   x*y*(1-c)-z*s, x*z*(1-c)+y*s, 0.0],
        [y*x*(1-c)+z*s,   y*y*(1-c)+c, y*z*(1-c)-x*s, 0.0],
        [z*x*(1-c)-y*s, z*y*(1-c)+x*s,   z*z*(1-c)+c, 0.0],
        [          0.0,           0.0,           0.0, 1.0]
    ])
    return rotation_matrix
    

def get_affine_3d_rotation_around_vector(vector: numpy.ndarray, vector_source: numpy.ndarray, rotation: float, rotation_units: str = "rad") -> numpy.ndarray(shape=(4, 4)):
    """Get the rotation matrix for a 3D rotation.

    Parameters
    ----------
    rotation : float
        The angle of rotation around the vector (in radians by default).
    vector : numpy.ndarray
        A 3D vector describing the axis of rotation.
    vector_source : numpy.ndarray
        A 3D vector describing the point source of the vector.
    rotation_units : str, optional
        The units of rotation. Either "rad" or "deg".
        Defaults to "rad".

    Returns
    -------
    numpy.ndarray
        The rotation matrix.
    """
    # Convert to radians if necessary
    if rotation_units == "deg":
        rotation = numpy.deg2rad(rotation)
    # Transform to unit vector
    vector = vector.astype(numpy.float32)
    vector_source = vector_source.astype(numpy.float32) 
    if numpy.linalg.norm(vector) == 0:
        raise ValueError("Parameter vector cannot be zero vector.")
    vector /= numpy.linalg.norm(vector)
    # Translate to origin from center of rotation
    translation_matrix = get_affine_3d_translation(-vector_source)
    # Rotate around the axis
    rotation_matrix = get_affine_3d_rotation_around_axis(vector, rotation)
    # Translate back to center of rotation
    translation_matrix2 = get_affine_3d_translation(vector_source)
    # Combine
    final_affine = numpy.matmul(numpy.matmul(translation_matrix2, rotation_matrix), translation_matrix)
    return final_affine


# ########################
# UTILITY TO APPLY AFFINES
# ########################

def compose_affines(affines: list[numpy.ndarray]) -> numpy.ndarray:
    """Composes affines to be applied in sequence.

    Given a list of [affine1, affine2, ..., affineN], this function
    will return the affine that is equivalent to applying the
    affines in the list in sequence, starting from affine1.

    y = affineN * ... * affine2 * affine1 * x
    
    y = affineN * ( ... ( affine2 * ( affine1 * x ) ) )

    Parameters
    ----------
    affines : list
        A list of affines to be applied in sequence.
    
    Returns
    -------
    numpy.ndarray
        The composed affine.
    """
    # Check input
    if len(affines) == 0:
        raise ValueError("List of affines cannot be empty.")
    # Compose
    # composition starts from the last one as the most internal one
    final_affine = affines[-1]
    for affine in affines[-2::-1]:
        final_affine = numpy.matmul(final_affine, affine)
    return final_affine

def apply_affine_3d(affine: numpy.ndarray, points: numpy.ndarray) -> numpy.ndarray:
    """Transform the input points with the specified affine.

    Parameters
    ----------
    points : numpy.ndarray
        The points to transform.
        points.shape = (3,) or (3, N) where N is the number of locations.
        If points.shape = (N,3) with N = {1,2,>3} then it will be transposed automatically.

    affine : numpy.ndarray
        The affine transformation matrix to use.

    Returns
    -------
    numpy.ndarray
        The transformed points, in shape (3,) or (3,N), NOT (N,3).
        To go back to (N,3), use output.T.
    """
    _single_point = False
    # Input pointa rejection
    if points.ndim > 2:
        raise ValueError(f"Data must be a (x,y,z) array. Got {points.ndim}D array.")
    if len(points.shape) == 1:
        _single_point = True
        points = points.reshape(points.shape[0],1)
    if points.shape[1] == 3 and (points.shape[0] in [1,2] or points.shape[0] > 3):
        # here we have an (N,3) array, so we transpose it
        # (3,3) is not considered because it is assumed to be correct
        points = points.T
    if points.shape[0] != 3:
        raise ValueError(f"Data must have 3 rows. Got {points.shape[0]} rows.")
    # Input affine rejection
    if affine.shape != (4,4):
        raise ValueError(f"Affine must be a 4x4 matrix. Got {affine.shape} matrix.")
    # Transform
    points = numpy.vstack((points, numpy.ones(points.shape[1])))
    points = numpy.matmul(affine, points)
    return points[0:3, :]