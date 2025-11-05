import numpy as np

# Define constants
field_angle_deg = 0.1  # degrees
field_angle_rad = field_angle_deg * np.pi / 180.  # convert to radians
mu0 = 4e-7 * np.pi
k1 = 4.3e6
js = 1.61
ha = 2 * k1 / js
a = 7.7e-12
lex = np.sqrt(mu0 * a / (js * js))

# prolate
N_a = 0.17356 
N_b = 0.41332
hn_prol = mu0*ha - js*(N_a - N_b)

# oblate
N_c = 0.5272
N_a = 0.2364
hn_obl = mu0*ha - js*(N_c - N_a)

# Compute Stoner-Wohlfarth reduction factor
f = (np.cos(field_angle_rad))**(2/3) + (np.sin(field_angle_rad))**(2/3)
f = f**(-3/2)

# Compute unit vector of the field in xz-plane
hx = np.sin(field_angle_rad)
hz = np.cos(field_angle_rad)
norm = np.sqrt(hx**2 + hz**2)
unit_vector = (float(hx / norm), 0.0, float(hz / norm))

# Output results
print("for Nd2Fe14B")
print("Field angle (radians):", field_angle_rad)
print("Stoner-Wohlfarth reduction factor:", f)
print("Unit vector of the field (tilted in xz-plane):  ", unit_vector)
print("Critical radius for coherent rotation (sphere) =", 10.2 * lex, "m")
print("Switching field of a sphere,            mu0H   =", f * mu0 * ha, "T")
print("Switching field of a prolate ellipsoid, mu0H   =", f * hn_prol, "T")
print("Switching field of a oblate  ellipsoid, mu0H   =", f * hn_obl, "T")
