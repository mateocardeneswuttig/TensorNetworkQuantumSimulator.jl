using TensorNetworkQuantumSimulator
using Random

Random.seed!(1934)

R = 2
g = named_grid((4,3))
s = siteinds("S=1/2", g)
ψ = random_tensornetworkstate(g, s; bond_dimension = 3)
ψ = normalize(ψ; alg = "bp")
z_exact = norm_sqr(ψ; alg = "exact")
ψ_ortho = update(BoundaryMPSCache(ψ, R))
z_ortho = partitionfunction(ψ_ortho)

@show abs(z_exact - z_ortho)

# --- biorthogonal (seed from the fitting cache!) ---
ψ_bio = TensorNetworkQuantumSimulator.converge_biorthogonal(ψ_ortho;
    maxiter = 200, tolerance = 1e-10, verbose = true)

z_biortho = partitionfunction(ψ_bio)
@show abs(z_exact - z_biortho)