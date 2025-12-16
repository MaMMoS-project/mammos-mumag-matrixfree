"""
Benchmark 1 workflow (draft/runbook)

Step 1 — Generate mesh
	Purpose: Create a polycrystal mesh via Neper with fixed grain count and extents.
	Command: python ../../src/mesh.py --geom poly --n 8 --id 123 --extent 80,80,80
	Output : isotrop.npz

Step 2 — Build krn for isotropic material
	Purpose: Create isotropic krn from the mesh; K1 = 700 kJ/m^3, Js = 0.8 T (defaults).
	Command: python ../../src/make_krn.py --tol 0.05 --mesh --out isotrop.krn

Step 3 — Run micromagnetic loop
	Purpose: Simulate loop with hstep=0.001 T, hstart=2 T, hfinal=-2 T.
	.p2 contents:
		[mesh]
		size = 1.0e-9

		[initial state]
		mx = 0.
		My = 0.
		mz = 1.

		[field]
		hstart = 2.
		hfinal = -2.
		hstep = 0.01
		hx = 0.
		hy = 0.
		hz = 1.
		mstep = 0.4
		mfinal = -2.0

		[minimizer]
		tol_fun = 1e-10
		tol_hmag_factor = 1
	Command: python ../../src/loop.py --mesh isotrop.npz

Step 4 — Repeat and average
	Purpose: Repeat Steps 1–3 ten times; compute the average hysteresis loop across runs.
"""