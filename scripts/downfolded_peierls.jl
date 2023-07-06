using SpecialFunctions
using StaticArrays
using HDF5

struct Mode{F}
    omega::F
    coupling::F
    max_photons::Int
end

struct DownfoldedPeierlsGenerator{F}
    mode::Mode{F}
    lmax::Int
    lgamma::Vector{F}
    logtolerance::F
    tolerance::F
end

function disp_op(gen::DownfoldedPeierlsGenerator, n::Integer, l::Integer, g::AbstractFloat)
    if g == 0
        return complex(zero(g), zero(g))
    end
    
    sum = zero(g)
    logg = log(Complex(g))
    @inbounds lgamma_ln = 0.5 * gen.lgamma[l+1] + 0.5 * gen.lgamma[n+1]
    
    sign = SVector(1.0, 1.0im, -1.0, -1.0im)
    
    m = min(n, l)
    d = abs(n-l)
    
    for a in 0:m
        @inbounds exponent = logg * (2 * a + d) + lgamma_ln - gen.lgamma[a+1] - gen.lgamma[m-a + 1] - gen.lgamma[d+a+1]
        if abs(exponent) > gen.logtolerance
            sum += (1-2*(a&1)) * exp(exponent)
        end
    end
    
    return sign[1 + d % 4] * sum
end

function element(gen::DownfoldedPeierlsGenerator, n::Integer, m::Integer)
    sum = zero(gen.mode.coupling)
    for l in 0:gen.lmax
        disp = disp_op(gen, n, l, gen.mode.coupling)*conj(disp_op(gen, m, l, gen.mode.coupling))
        sum += real(disp) * (1.0/(1 + gen.mode.omega * (l-n)) + 1.0 / (1 + gen.mode.omega*(l-m)))
    end
    sign = (1 - 2* ((n-m)÷2&1))
    return exp(-gen.mode.coupling^2) * sign * 0.5 * sum
end

function DownfoldedPeierlsGenerator(mode::Mode; tolerance::AbstractFloat)
    lmax = round(Int, 2*mode.max_photons - log(tolerance))
    return DownfoldedPeierlsGenerator(mode, lmax, collect(map(x->loggamma(x), 1:lmax+1)), log(tolerance), tolerance)
end

function matrix(gen::DownfoldedPeierlsGenerator)    
    mat = zeros(gen.mode.max_photons, gen.mode.max_photons)
    for n in 0:gen.mode.max_photons-1
        for m in 0:gen.mode.max_photons-1
            el = element(gen, n, m)
            
            mat[m+1,n+1] = abs(el) > 10*gen.tolerance ? el : 0
        end
    end
    return mat
end

function sw_correction(;omega, Js, gs, weights, func, max_photons=10)
    res = zero(omega)
    Jsum = sum(Js .* weights)
    for (J, g, weight) in zip(Js, gs, weights)
        gen = DownfoldedPeierlsGenerator(Mode(omega, g, max_photons), tolerance=1e-9)
        for n in 0:max_photons
            res += weight*J/Jsum * func(n) * real(disp_op(gen, 0, n, g) * conj(disp_op(gen, 0, n, g))) / (1 + gen.mode.omega * n)^2
        end
    end
    
    return res
end
