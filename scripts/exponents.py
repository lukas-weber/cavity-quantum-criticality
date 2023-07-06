import uncertainties

# Kos et al, http://doi.org/10.1007/JHEP08(2016)036
# 3D O(3)
Δφ = uncertainties.ufloat(0.51928, 62e-5)
Δs = uncertainties.ufloat(1.5957, 55e-4)
d = 2
D = d + 1

beta = Δφ / (D - Δs)
nu = 1 / (D - Δs)


if __name__ == "__main__":
    print(f"β = {beta:uS}")
    print(f"ν = {nu:uS}")
    print(f"1/ν-d = {1/nu-d:uS}")
    print(f"(1-2β)/ν = {(1-2*beta)/nu:uS}")
