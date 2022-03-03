#=
Utilities for Chebyshev polynomial series:
multiplication, dilation, and truncation.
=#
module Cheb

using Fct: chebsample, chebcoeffs, doublesample

abstract type ChebSeries{T<:Number} end
struct FullChebSeries{T<:Number} <: ChebSeries{T}
    coeff::AbstractArray{T}
end
struct EvenChebSeries{T<:Number} <: ChebSeries{T}
    coeff::AbstractArray{T}
end
struct OddChebSeries{T<:Number} <: ChebSeries{T}
    coeff::AbstractArray{T}
end

FullChebSeries(a::AbstractArray{T}) where T = FullChebSeries{T}(a)
EvenChebSeries(a::AbstractArray{T}) where T = EvenChebSeries{T}(a)
OddChebSeries(a::AbstractArray{T}) where T = OddChebSeries{T}(a)
ChebSeries(a::AbstractArray{T}) where T = FullChebSeries{T}(a)
ChebSeries{T}(a::AbstractArray{T}) where T = FullChebSeries{T}(a)

FullChebSeries(T::Type,deg::Integer) = FullChebSeries{T}(Vector{T}(undef,deg+1))
EvenChebSeries(T::Type,deg::Integer) = EvenChebSeries{T}(Vector{T}(undef,div(deg,2)+1))
OddChebSeries(T::Type,deg::Integer) = OddChebSeries{T}(Vector{T}(undef,div(deg,2)+1))
ChebSeries(T::Type,deg::Integer) = FullChebSeries(T,deg)
    
EvenChebSeries(c::FullChebSeries{T}) where T = EvenChebSeries{T}(view(c.coeff,1:2:length(c.coeff)))
OddChebSeries(c::FullChebSeries{T}) where T = OddChebSeries{T}(view(c.coeff,2:2:length(c.coeff)))

#external 0-based indexing; internal 1-based indexing
Base.getindex(c::FullChebSeries{T}, n::Integer) where T = c.coeff[1+n]
Base.getindex(c::EvenChebSeries{T}, n::Integer) where T = iseven(n) ? c.coeff[1+div(n,2)] : zero(T)
Base.getindex(c::OddChebSeries{T}, n::Integer) where T = isodd(n) ? c.coeff[1+div(n,2)] : zero(T)
Base.setindex!(c::FullChebSeries{T}, x, n::Integer) where T = (c.coeff[1+n] = T(x))
Base.setindex!(c::EvenChebSeries{T}, x, n::Integer) where T = (c.coeff[1+div(n,2)] = T(x))
Base.setindex!(c::OddChebSeries{T}, x, n::Integer) where T = (c.coeff[1+div(n,2)] = T(x))

#assumes length(c.coeff) >= 1
degree(c::FullChebSeries) = length(c.coeff) - 1
degree(c::EvenChebSeries) = 2*(length(c.coeff) - 1)
degree(c::OddChebSeries) = 2*length(c.coeff) - 1

support(c::FullChebSeries) = 0:degree(c)
support(c::EvenChebSeries) = 0:2:degree(c)
support(c::OddChebSeries) = 1:2:degree(c)

#assumes deg >= degree(c)
truncate(c::FullChebSeries, deg::Integer) = FullChebSeries(view(c.coeff,1:deg+1))
truncate(c::EvenChebSeries, deg::Integer) = EvenChebSeries(view(c.coeff,1:div(deg,2)+1))

#assumes deg >= degree(c) 
truncate(c::OddChebSeries, deg::Integer) = OddChebSeries(view(c.coeff,1:div(deg-1,2)+1))

Base.eltype(c::ChebSeries) = eltype(c.coeff)

FullChebSeries(f::Function) = autotruncate(FullChebSeries(autocheb(f)))
ChebSeries(f::Function) = FullChebSeries(f)
EvenChebSeries(f::Function) = EvenChebSeries(FullChebSeries(f))
OddChebSeries(f::Function) = OddChebSeries(FullChebSeries(f))

import Base: *
function (*)(a::ChebSeries, b::ChebSeries)::FullChebSeries
    L,M=degree(a),degree(b)
    T=promote_type(eltype(a),eltype(b))
    c=FullChebSeries(T,L+M)
    mult!(a,L,b,M,c)
end
function (*)(a::EvenChebSeries, b::EvenChebSeries)::EvenChebSeries
    L,M=degree(a),degree(b)
    T=promote_type(eltype(a),eltype(b))
    c=EvenChebSeries(T,L+M)
    mult!(a,L,b,M,c)
end
function (*)(a::OddChebSeries, b::OddChebSeries)::EvenChebSeries
    L,M=degree(a),degree(b)
    T=promote_type(eltype(a),eltype(b))
    c=EvenChebSeries(T,L+M)
    mult!(a,L,b,M,c)
end
function (*)(a::OddChebSeries, b::EvenChebSeries)::OddChebSeries
    L,M=degree(a),degree(b)
    T=promote_type(eltype(a),eltype(b))
    c=OddChebSeries(T,L+M)
    mult!(a,L,b,M,c)
end
function mult!(a::ChebSeries, L, b::ChebSeries, M, c::ChebSeries)
    T = eltype(c)
    half, init = T(0.5), zero(T)
        
    for n in support(c)
        #2T_i*T_j==T_{i+j}+T_{|i-j|}
        c_n = init
        
        itr = intersect(max(0,n-M):min(L,n), support(a))
        length(itr)>0 && (c_n += sum(a[l]*b[n-l] for l in itr))
        
        itr = intersect(min(L,M+n):-1:n, support(a))
        length(itr)>0 && (c_n += sum(a[l]*b[l-n] for l in itr))

        if n > 0
            itr = intersect(min(L,M-n):-1:0, support(a))
            length(itr)>0 && (c_n += sum(a[l]*b[l+n] for l in itr))
        end
        
        c[n] = half * c_n
    end
    c
end
(*)(a::EvenChebSeries, b::OddChebSeries)::OddChebSeries = b * a
(*)(a::FullChebSeries, b::EvenChebSeries)::FullChebSeries = b * a
(*)(a::FullChebSeries, b::OddChebSeries)::FullChebSeries = b * a


Base.conj(c::ChebSeries) = typeof(c)(conj(c.coeff))
Base.conj!(c::ChebSeries) = typeof(c)(conj!(c.coeff))

#Outputs o such that sum_n o[n]T_n(x)=sum_n c[n]T_n(kappa*x)
function dilate(c::ChebSeries,kappa::Number)::ChebSeries
    T=eltype(c)
    o=typeof(c)(zeros(eltype(c),size(c.coeff)))
    d=zeros(T,3,div(degree(c),2)+1)
    n0,n1,n2=3,2,1
    d[n0,1]=1
    0 in support(c) && (o[0]=c[0])
    n0,n1,n2 = n2,n0,n1
    d[n0,1]=kappa
    1 in support(c) && (o[1]=c[1]*kappa)
    for n in 2:degree(c)
        n0,n1,n2 = n2,n0,n1
        d[n0,1] = d[n1,1]*kappa
        m0 = div(n,2)
        for m in 1:m0-1
            d[n0,1+m] = (d[n1,1+m] + d[n1,m])*kappa - d[n2,m]
        end
        if isodd(n)
            d[n0,1+m0] = (2d[n1,1+m0] + d[n1,m0])*kappa - d[n2,m0]
        else
            d[n0,1+m0] = d[n1,m0]*kappa - d[n2,m0]
        end
        if n in support(c)
            cn = c[n]
            for m in 0:m0
                o[n-2m] += cn*d[n0,1+m]
            end
        end
    end
    o
end

#evaluate sum_n c[n+1]*T_n(x)
function chebinterp(c::ChebSeries,x::Number)
    old,new=1,x
    out=c[0]+c[1]*x
    for n in 2 : degree(c)
        old, new = new, 2x*new-old
        out+=c[n]*new
    end
    out
end

#=
Outputs coefficient vector c such that f(x) is interpolated 
and well approximated by sum_n c[n+1] T_n(x).
=#
function autocheb(f;log2N0::Integer=1,T=Complex{Float64})
    N = 2^max(log2N0,1)
    while true
        samples = chebsample(f,2N;T=T) #length 2N+1
        coeffs = chebcoeffs(samples,T)
        head = sum(abs2(coeffs[k]) for k in N+1:-1:1)
        tail = sum(abs2(coeffs[k]) for k in 2N+1:-1:N+2)
        rtol = eps()*(N<=64 ? N<=32 ? 1 : 4 : N<=128 ? 16 : N)
        if tail > head * rtol^2 && N <= 2^19
            samples = doublesample(f,samples)
            N<<=1
        else
            return view(coeffs,1:min(N+1,2N+1))
        end
    end
end

function autotruncate(c::ChebSeries)
    base = maximum((abs2(c[n]) for n in 0:degree(c)))
    base *= eps()^2
    n = degree(c)
    while abs2(c[n]) < base * n
        n -= 1
    end
    truncate(c,n)
end

__init__(args...) = println("Cheb.__init__($args)")

end #module
