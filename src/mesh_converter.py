from pathlib import Path

import meshio
import numpy as np

TETRA = "tetra"


def convert_mesh(mesh_file: Path) -> None:
    mesh = meshio.read(mesh_file)

    knt = mesh.points
    tets = mesh.cells_dict.get(TETRA)
    if tets is None:
        raise RuntimeError("No tetra cells found in Neper output VTK.")

    # Try to find a per-tetra cell-data array to use as material/grain IDs.
    mat = None
    # Prefer the cell_data_dict (present in modern meshio versions)
    try:
        cd_tet = mesh.cell_data_dict.get(TETRA, {})
        for key in (
            "matids",
            "mat_id",
            "poly",
            "grain",
            "gmsh:physical",
            "material",
            "region",
            "domain",
        ):
            if key not in cd_tet:
                print(f"Looking for cell data key '{key}'... not found.")
            mat = np.asarray(cd_tet[key], dtype=np.int32).ravel()
            break
    except Exception:
        print("Error accessing cell_data_dict in meshio; falling back to older layout.")
        pass

    # Fallback: inspect mesh.cell_data (older meshio layout)
    if mat is None and hasattr(mesh, "cell_data"):
        print("Falling back to older meshio cell_data layout.")
        for key, data_list in mesh.cell_data.items():
            # Each data_list aligns with mesh.cells blocks
            for cell_block, data in zip(mesh.cells, data_list):
                if (
                    getattr(cell_block, "type", getattr(cell_block, "type", None))
                    == "tetra"
                ):
                    mat = np.asarray(data, dtype=np.int32).ravel()
                    break
            if mat is not None:
                break

    # Last resort: all ones (warn)
    if mat is None:
        print(
            "[warn] No per-tetra cell data found in Neper VTK; defaulting mat_id=1.",
        )
        mat = np.ones((tets.shape[0],), dtype=np.int32)

    print(f"Found group IDs for {np.unique(mat)} materials/grains.")
    # Convert negative groups to 1-based positive IDs based on max negaive ID
    if np.any(mat < 0):
        min_id = np.min(mat)
        offset = 1 - min_id
        print(
            f"[warn] Found negative material IDs; offsetting all IDs by {offset}.",
        )
        mat += offset

    # Build ijk (E,5): 4 indices + mat_id
    ijk = np.column_stack([tets, mat])
    out_npz = mesh_file.with_suffix(".npz")
    np.savez(out_npz, knt=knt.astype(np.float64), ijk=ijk.astype(np.int32))


if __name__ == "__main__":
    convert_mesh(Path("/home/david/repos/salome/results/box_grains/box_grains.vtk"))
