import pathlib as pl

import meshio
import numpy as np
import pyvista as pv


def standardise_mesio_mesh(
    mesh: pl.Path | str, cell_data_key: str = "cell_tags"
) -> meshio.Mesh:
    """Output standard meshio Mesh from msh, med, and vtu files

    This function takes in mesh files in either med, msh, or vtu format and
    converts them to a standard meshio mesh object that only has tetrahedral
    cells and subregion tags that are stored as cell data with 'cell_data_key'.
    It is necessary to provide the 'cell_data_key' identifier for med and vtu
    files in order to retrieve the right subregion tags from the file.
    """
    mesh_path = mesh if isinstance(mesh, pl.Path) else pl.Path(mesh)

    if mesh_path.suffix not in {".med", ".msh", ".vtu"}:
        raise RuntimeError(
            f"{mesh_path.suffix} meshes are not supported."
            "Only med, msh, and vtu meshes are supported."
        )

    meshio_mesh: meshio.Mesh = meshio.read(mesh_path)
    if mesh_path.suffix != ".msh" and cell_data_key not in meshio_mesh.cell_data:
        raise RuntimeError(
            f"Key {cell_data_key} not found in the cell data of the mesh."
            f"The keys in the cell data are {meshio_mesh.cell_data.keys()}."
        )

    points = meshio_mesh.points
    found_tetra_cells = False
    for cell_block in meshio_mesh.cells:
        if cell_block.type == "tetra":
            connectivity = cell_block.data
            found_tetra_cells = True
            break
    if not found_tetra_cells:
        raise RuntimeError("No tetrahedral cells found in the mesh.")

    if mesh_path.suffix == ".msh":
        cell_data = meshio_mesh.cell_data_dict["gmsh:physical"]["tetra"]
    else:
        cell_data = meshio_mesh.cell_data_dict[cell_data_key]["tetra"]

    if mesh_path.suffix == ".med":
        for key, val in meshio_mesh.cell_tags.items():
            cell_data[cell_data == key] = int(val[0])

    return meshio.Mesh(
        points=points,
        cells=[("tetra", connectivity)],
        cell_data={cell_data_key: [cell_data]},
    )


def meshio_to_numpy_file(
    mesh: meshio.Mesh, file_name: str, cell_data_key: str = "cell_tags"
):
    """Convert standard meshio mesh object to Tom's numpy file.

    The function assumes that the input meshio mesh object is standardised
    according to the 'standardise_mesio_mesh' function. The meshio mesh object
    is subsequently converted to Tom's npz numpy mesh format.
    """
    cell_tags = mesh.cell_data[cell_data_key][0]
    ijk = np.empty((cell_tags.shape[0], 5), dtype=np.int_)
    ijk[:, 0:-1] = mesh.cells[0].data
    ijk[:, -1] = cell_tags

    np.savez(file_name, knt=mesh.points, ijk=ijk)


def numpy_file_to_pyvista(
    mesh: pl.Path | str, cell_data_key: str = "cell_tags"
) -> pv.UnstructuredGrid:
    """Convert Tom's numpy mesh to PyVista object.

    The function requires a path to Tom's npz numpy mesh file and converts it
    into a pyvista.UnstructuredGrid object.
    """
    npz = np.load(mesh)
    points = npz["knt"]
    connectivity = npz["ijk"][:, 0:-1]
    cell_tags = npz["ijk"][:, -1]
    m = meshio.Mesh(
        points, [("tetra", connectivity)], cell_data={cell_data_key: [cell_tags]}
    )

    return pv.from_meshio(m)
