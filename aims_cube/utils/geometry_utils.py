import numpy as np
from scipy.optimize import linprog, OptimizeResult


def get_vertices(lattice_matrix):
    """
    Given a 3x3 lattice matrix, returns the 8 vertices (Cartesian coordinates) of the parallelepiped.
    """
    origin = np.zeros(3)
    a1, a2, a3 = lattice_matrix  # columns are lattice vectors
    vertices = [
        origin, a1, a2, a3,
        a1 + a2, a1 + a3, a2 + a3,
        a1 + a2 + a3
    ]
    return np.array(vertices)


# This vertex-based approach is easy to prove applicable to all CONVEX polyhedrons.
def find_translation_vector(contained_matrix, containing_matrix):
    """
    Finds the translation vector x that translates the parallelepiped defined
    by contained_matrix to fit fully within the parallelepiped defined by
    containing_matrix.
    """
    # Get vertices of the contained parallelepiped (Cartesian)
    contained_vertices = get_vertices(contained_matrix)

    # Transformation from Cartesian coordinate system to the fractional coordinate system of the containing parallelepiped
    transform_matrix = np.linalg.inv(containing_matrix)

    # Set up the linear programming problem
    A_ub_list = []  # noqa
    b_ub_list = []
    for v in contained_vertices:
        # (v + x) * t > 0
        # x * t > - v * t
        # tt * xt > -tt * vt
        # -tt * xt < tt * vt
        A_ub_list.append(-transform_matrix.T)
        b_ub_list.append(transform_matrix.T @ v)

        # (v + x) * t < 1
        # x * t < 1t - v * t
        # tt * xt < 1t - tt * vt
        A_ub_list.append(transform_matrix.T)
        b_ub_list.append(np.ones(3, dtype=float) - transform_matrix.T @ v)

    A_ub = np.concatenate(A_ub_list, axis=0)  # noqa
    b_ub = np.concatenate(b_ub_list)

    # Solve the linear programming problem
    result: OptimizeResult = linprog(c=np.ones(3), A_ub=A_ub, b_ub=b_ub, bounds=(None, None), method='highs')  # noqa

    print(f"Solving polyhedral containment problem using linear programming: {result.message}")

    return result.success, result.x


if __name__ == '__main__':
    # Example usage
    contained_lattice_matrix = np.array([[1, -0.5, 0], [-0.5, 1, 0], [0, 0, 1.5]])
    containing_lattice_matrix = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])

    success, translation_vector = find_translation_vector(contained_lattice_matrix, containing_lattice_matrix)
    if success:
        print(
            f"The contained parallelepiped can be fully contained within the containing parallelepiped with translation vector {translation_vector}.")
    else:
        print("The contained parallelepiped cannot be fully contained within the containing parallelepiped.")
