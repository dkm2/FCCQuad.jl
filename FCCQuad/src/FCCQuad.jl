module FCCQuad

using LinearAlgebra, Scratch
include("Fct.jl")
include("chebweights.jl")
include("Cheb.jl")
include("chirps.jl")
include("Jets.jl")
using Jets: Jet, phase_velocity, phase_acceleration
using Fct: chebcoeffs, chebsample, doublesample, chebsample!, chebcoeffs!, fct_alloc

#=
Filon-Clenshaw-Curtis quadrature.
Assuming chebfun is the coefficient vector of
a Chebyshev expansion of f(x) of degree at least N,
returns integral of f(x)*exp(2pi*im*w*x)*dx over [-1,1]
using degree N, and discrepancy between that and using degree N/2.
WARNING: N is assumed to be a positive multiple of 4.
=#
function fccquad(chebfun::Vector,w::Real,N::Integer)
    @assert rem(N,4)==0
    i=getweights(N,w)
    a=chebfun
    half=(sum(a[n]*i[n] for n in div(N,2)+1:-2:1) +
          sum(a[n]*i[n] for n in div(N,2):-2:2)*im)
    diff=(sum(a[n]*i[n] for n in N+1:-2:div(N,2)+3) +
          sum(a[n]*i[n] for n in N:-2:div(N,2)+2)*im)
    full = diff + half
    full,diff
end

function fccquad(f::Function,w::Real,log2N::Integer)
    N=1<<log2N
    a=chebcoeffs(chebsample(f,N))
    full,diff=fccquad(a,w,N)
    full,abs(diff)
end

function fccquad(f::Function,freqs::AbstractArray,log2N::Integer,T=Complex{Float64})
    N=1<<log2N
    a=chebcoeffs(chebsample(f,N,T))
    M=length(freqs)
    result=Array{T}(undef,2,M)
    for m in 1:M
        w = freqs[m]
        result[1,m],result[2,m]=fccquad(a,w,N)
    end
    result
end

#=
Filon-Clenshaw-Curtis quadrature for f(x) and xf(x) on [-1,1].
Assuming chebfun is the coefficient vector of
a Chebyshev expansion of degree at least N of some function f(x),
returns, for each of i=0,1, x^i*f(x)*exp(2pi*im*w*x)*dx integrated over [-1,1]
using degree N, and discrepancy between that and using reduced_degree(N).
WARNING: N and baseN are assumed to be even.
=#
function fccquadxUnit!(chebfun::AbstractVector,freq::Real,N::Integer,
                       weightmethod::Symbol,weights::AbstractVector,workspace::AbstractArray,
                       baseN=reduced_degree(N))
    getweights!(weights,workspace,length(weights)-1,freq,weightmethod)
    fccx_core(chebfun,weights,N,baseN)
end    
reduced_degree(N)=div(3N,4)
function fccx_core(chebfun::AbstractVector,weights::Vector,N::Integer,baseN::Integer)
    @assert iseven(N) && iseven(baseN)
    a=chebfun
    i=weights
    base=(sum(a[n]*i[n] for n in baseN+1:-2:1) +
          sum(a[n]*i[n] for n in baseN:-2:2)*im)
    diff=(sum(a[n]*i[n] for n in N+1:-2:baseN+3) +
          sum(a[n]*i[n] for n in N:-2:baseN+2)*im)
    full = base + diff
    
    basex = (sum(a[n]*(i[n-1]+i[n+1]) for n in baseN:-2:2)*0.5 + 
             sum(a[n]*(i[n-1]+i[n+1]) for n in baseN-1:-2:3)*0.5im +
             a[1]*i[2]*im)
    diffx = (sum(a[n]*(i[n-1]+i[n+1]) for n in N:-2:baseN+2)*0.5 + 
             sum(a[n]*(i[n-1]+i[n+1]) for n in N-1:-2:baseN+1)*0.5im)
    fullx = basex + diffx
             
    full,diff,fullx,diffx
end
#=
Alternative, more flexible error estimation method
that compares the results of using two user-provided Chebyshev expansions 
instead of one Chebyshev expansion and its truncation.
WARNING: N and baseN are assumed to be even.
=#
function fccx_alt(chebfun::AbstractVector,basefun::AbstractVector,
                  weights::Vector,N::Integer,baseN::Integer)
    @assert iseven(N) && iseven(baseN)
    a,b,i=chebfun,basefun,weights
    full=(sum(a[n]*i[n] for n in N+1:-2:1) +
          sum(a[n]*i[n] for n in N:-2:2)*im)
    base=(sum(b[n]*i[n] for n in baseN+1:-2:1) +
          sum(b[n]*i[n] for n in baseN:-2:2)*im)
    diff = full - base
    
    fullx = (sum(a[n]*(i[n-1]+i[n+1]) for n in N:-2:2)*0.5 + 
             sum(a[n]*(i[n-1]+i[n+1]) for n in N-1:-2:3)*0.5im +
             a[1]*i[2]*im)
    basex = (sum(b[n]*(i[n-1]+i[n+1]) for n in baseN:-2:2)*0.5 + 
             sum(b[n]*(i[n-1]+i[n+1]) for n in baseN-1:-2:3)*0.5im +
             b[1]*i[2]*im)
    diffx = fullx - basex
             
    full,diff,fullx,diffx
end

#=
Filon-Clenshaw-Curtis quadrature for f(x) and xf(x) 
with arbitrary finite interval domain.
Assuming chebfun is the coefficient vector of
a Chebyshev expansion of degree at least N of some function g(x)=f(c+rx),
returns, for each of i=0,1, the integral over [-1,1] of
(c+rx)^i*f(c+rx)*exp(2pi*im*(w*(c+rx)+v*rx))*d(c+rx)
using degree N, and discrepancy between that and using degree baseN.
Above, w=freq, c=center, r=radius, and v=doppler.
WARNING: N and baseN are assumed to be even
=#
function fccquadxSampled(chebfun::AbstractVector,freq::Real,N::Integer,center::Real,radius::Real,
                         doppler::Real=0.0,weightmethod::Symbol=:thomas,baseN::Integer=reduced_degree(N))
    weights,workspace=weights_alloc(N,weightmethod)[1:2]
    fccquadxSampled!(chebfun,freq,N,center,radius,doppler,weightmethod,weights,workspace,baseN)
end
function fccquadxSampled!(chebfun::AbstractVector,freq::Real,N::Integer,
                          center::Real,radius::Real,doppler::Real,
                          weightmethod::Symbol,weights::AbstractVector,workspace::AbstractArray,
                          baseN::Integer=reduced_degree(N))
    w = freq*radius + doppler
    interval_transform(fccquadxUnit!(chebfun,w,N,weightmethod,weights,workspace,baseN)...,freq,center,radius)
end
function interval_transform(full,diff,fullx,diffx,freq,center,radius)
    fullx = full*center + fullx*radius
    diffx = diff*center + diffx*radius
    si,co=sincos(freq*center)
    mult = radius*complex(co,si)
    full*mult, diff*mult, fullx*mult, diffx*mult
end

#=
#Integral of x^i*f(x)*exp(2pi*im*w*x)*dx over [xmin,xmax] for i=0,1
#computed using degree-N Filon-Clensaw-Curtis quadrature.
#WARNING: N is assumed to be a power of 2; baseN is assumed to be even.
=#
function fccquadxLimits(f::Function,xmin::Real,xmax::Real,freq::Real,
                        N::Integer,weightmethod::Symbol,baseN=reduced_degree(N))
    center, radius = 0.5(xmin+xmax), 0.5(xmax-xmin)
    g(x)=f(x*radius + center)
    a=chebcoeffs(chebsample(g,N))
    fccquadxSampled(a,freq,N,center,radius,0.,weightmethod,baseN)
end

#Returns x^i*f(x)*exp(2pi*im*w*x) over -1,1 for i=0,1 using degree N
#and discrepancy between that and using reduced_degree(N).
#WARNING: log2N is assumed to be at least 2.
function fccquadxBatch(f::Function,freqs::AbstractArray,log2N::Integer,weightmethod::Symbol;
                       T::Type=Complex{Float64})
    output,workspaces = fccquad_alloc(freqs,log2N,weightmethod,T)
    fccquadxBatch!(output,workspaces,0.0,1.0,f,freqs,log2N,weightmethod)
end
function fccquad_alloc(freqs::AbstractArray,log2N::Integer,weightmethod::Symbol,T::Type=Complex{Float64})
    output=Matrix{T}(undef,4,length(freqs))
    N=1<<log2N
    samples = Vector{T}(undef,1+N)
    fct_workspaces = fct_alloc(samples,T)
    weights,weights_workspace = weights_alloc(N,weightmethod,real(T))[1:2]
    workspaces = (samples,fct_workspaces),(weights,weights_workspace)
    output,workspaces
end
function fccquadxBatch!(output::AbstractArray,workspaces,center::Real,radius::Real,
                        f::Function,freqs::AbstractArray,log2N::Integer,
                        weightmethod::Symbol,baseN=reduced_degree(1<<log2N))
    N=1<<log2N
    g(x)=f(x*radius + center)
    samples,fct_workspaces = workspaces[1]
    a=chebcoeffs!(chebsample!(g,samples,N),fct_workspaces...)
    M=length(freqs)
    weights,weights_workspace = workspaces[2]
    for m in 1:M
        output[:,m] = collect(fccquadxSampled!(a,freqs[m],N,center,radius,0.,
                                               weightmethod,weights,weights_workspace,baseN))
    end
    output
end

function adaptdegree(f::Function,freqs::AbstractArray;T::Type=Complex{Float64},maxdegree=1<<20,
                     xmin=-1.0,xmax=1.0,reltol::Real=1e-8,abstol::Real=0.0)
    center = 0.5(xmax + xmin)
    radius = 0.5(xmax - xmin)
    g(x) = f(x*radius + center)
    N = 16
    samples=chebsample(g,N)
    output=Array{T}(undef,4,length(freqs))
    while true
        chebfun=chebcoeffs(samples)
        weights,workspace=weights_alloc(N,:thomas)[1:2]
        for m in 1:length(freqs)
            output[:,m] = collect(fccquadxSampled!(chebfun,freqs[m],N,center,radius,0.0,
                                                    :thomas,weights,workspace))
        end
        base=norm(view(output,1,:))
        delta=norm(view(output,2,:))
        basex=norm(view(output,3,:))
        deltax=norm(view(output,4,:))
        #println((N,base,delta,basex,deltax))
        if ((delta <= base * reltol || delta * radius <= abstol) &&
            (deltax<= basex* reltol || deltax* radius <= abstol)) ||
            N >= maxdegree
            break
        end
        samples = doublesample(g,samples)
        N<<=1
    end
    output,N+1
end

#Returns x^i*f(x)*exp(2pi*im*w*x) over -1,1 for i=0,1 using degree N
#and discrepancy between that and using degree N/2.
#Uses tone (linear phase) removal.
#WARNING: log2N is assumed to be at least 2.
function tonequad(f::Function,freqs::AbstractArray,log2N::Integer,weightmethod::Symbol=:thomas;
                  T::Type=Complex{Float64})
    output,workspaces = fccquad_alloc(freqs,log2N,weightmethod,T)
    tonequad!(output,workspaces,0.0,1.0,f,freqs,log2N,weightmethod)
end
function tonequad!(output::AbstractArray,workspaces,center::Real,radius::Real,
                   f::Function,freqs::AbstractArray,log2N::Integer,weightmethod::Symbol)
    N=1<<log2N
    g(x)=f(x*radius + center)
    cfreq = phase_velocity(g(Jet(0,1)))
    function h(y)
        s,c=sincos(cfreq*y)
        g(y)*complex(c,-s)
    end
    samples,fct_workspaces = workspaces[1]
    a=chebcoeffs!(chebsample!(h,samples,N),fct_workspaces...)
    weights,weights_workspace = workspaces[2]
    for m in 1:length(freqs)
        output[:,m] = collect(fccquadxSampled!(a,freqs[m],N,center,radius,cfreq,
                                               weightmethod,weights,weights_workspace))
    end
    output
end

#Returns x^i*f(x)*exp(2pi*im*w*x) over -1,1 for i=0,1 using degree N
#and discrepancy between that and using degree N/2.
#Uses linear chirp removal.
#WARNING: log2N is assumed to be at least 2.
function chirpquad(f::Function,freqs::AbstractArray,log2N::Integer,weightmethod::Symbol=:thomas;
                   T::Type=Complex{Float64})
    output=Matrix{T}(undef,4,length(freqs))
    chirpquad!(output,0.0,1.0,f,freqs,log2N,weightmethod;T=T)
end
function chirpquad!(output::AbstractArray,center::Real,radius::Real,
                    f::Function,freqs::AbstractArray,log2N::Integer,weightmethod::Symbol;
                    T::Type=Complex{Float64})
    N=1<<log2N
    g(x)=f(x*radius + center)
    jet = g(Jet(0,1))
    cfreq::Real = phase_velocity(jet)
    chirp::Real = phase_acceleration(jet)
    if Chirps.rates[end] < abs(chirp)
        #give up, output garbage
        #println((center,radius,chirp,Chirps.rates[5]))
        output[:,1:length(freqs)-1] .= zero(eltype(output))
        output[1,end] = one(eltype(output))
        output[3,end] = one(eltype(output))
        output[2,end] = Inf
        output[4,end] = Inf
        return output
    end    
    function h(y)
        s,c=sincos(y*(cfreq+y*chirp))
        g(y)*complex(c,-s)
    end
    a=Cheb.ChebSeries(chebcoeffs(chebsample(h,N)))

    index=findfirst(x->abs(chirp)<=x,Chirps.rates)::Integer
    faster=Chirps.rates[index]
    b0 = Cheb.EvenChebSeries(Chirps.getchirp(index))
    b=Cheb.dilate(b0,sqrt(abs(chirp)/faster))
    if chirp < zero(chirp)
        b = conj(b)
    end
    ab,ab2 = a*b, Cheb.truncate(a,reduced_degree(N))*b
    deg = Cheb.degree(Cheb.autotruncate(ab))
    deg2 = Cheb.degree(Cheb.autotruncate(ab2))
    @assert deg >=2 && deg2 >= 2
    deg1 = max(1, deg >> 2) << 2 #truncate to multiple of 4
    deg2 = max(1, deg2 >> 2) << 2 #truncate to multiple of 4
    deg3 = max(deg,deg2)

    weights,workspace,deg4=weights_alloc(deg3,weightmethod)
    for m in 1:length(freqs)
        w = freqs[m]*radius + cfreq
        getweights!(weights,workspace,deg4,w,weightmethod)
        step1 = fccx_alt(ab.coeff,ab2.coeff,weights,deg1,deg2)
        step2 = interval_transform(step1...,freqs[m],center,radius)
        output[:,m] = collect(step2)
    end
    output
end

#Adaptively integrates x^i*f(x)*exp(2pi*im*w*x) for a batch of w, for i=0,1.
function adaptquad(f::Function,freqs::AbstractArray,log2N::Integer;
                   xmin=-1.0,xmax=1.0,reltol=1e-8,abstol=0.0,
                   interpolation=:tone,weightmethod=:thomas,T::Type=Complex{Float64})
    output=zeros(T,4,length(freqs))
    center = 0.5(xmax + xmin)
    radius = 0.5(xmax - xmin)
    if interpolation == :chirp
        subintegrals,workspaces = Array{T}(undef,4,length(freqs)),nothing
    else
        subintegrals,workspaces = fccquad_alloc(freqs,log2N,weightmethod,T)
    end
    adaptquad!(output,subintegrals,workspaces,center,radius,f,freqs,log2N,reltol,abstol,interpolation,weightmethod)
end
#adds results in place to output
function adaptquad!(output::AbstractArray,subintegrals::AbstractArray,workspaces,
                    center::Real,radius::Real,
                    f::Function,freqs::AbstractArray,log2N::Integer,
                    reltol::Real,abstol::Real,interpolation::Symbol,weightmethod::Symbol)
    if interpolation == :chirp #Filon-Clenshaw-Curtis quadrature with linear chirp removal
        chirpquad!(subintegrals,center,radius,f,freqs,log2N,weightmethod)
    elseif interpolation == :tone #Filon-Clenshaw-Curtis quadrature with tone removal
        tonequad!(subintegrals,workspaces,center,radius,f,freqs,log2N,weightmethod)
    else #Filon-Clenshaw-Curtis quadrature (:plain)
        fccquadxBatch!(subintegrals,workspaces,center,radius,f,freqs,log2N,weightmethod)
    end
    base=norm(view(subintegrals,1,:))
    delta=norm(view(subintegrals,2,:))
    basex=norm(view(subintegrals,3,:))
    deltax=norm(view(subintegrals,4,:))
    evals = isfinite(delta) ? 1+(1<<log2N) : 1
    if ((delta <= base * reltol || delta  <= abstol) &&
        (deltax<= basex* reltol || deltax <= abstol))
        output .+= subintegrals
    else
        #println((center,radius,delta/base,deltax/basex))
        r = 0.25radius
        abstol = 0.25max(abstol, min(base * reltol, basex * reltol))
        for t in -3:2:3
            evals += adaptquad!(output,subintegrals,workspaces,r*t+center,r,
                                f,freqs,log2N,reltol,abstol,interpolation,weightmethod)[2]
        end
    end
    output,evals
end

function __init__(args...)
    println("FCCQuad.__init__($args)")
    cache_dir = @get_scratch!("precomputed_chebyshev_weights")
    cache_file = joinpath(cache_dir,"chirps.h5")
    isfile(cache_file) || Chirps.store(cache_file)
    Chirps.load(cache_file)
    Fct.__init__(1)
    Cheb.__init__(1)
    Chirps.__init__(1)
    Jets.__init__(1)
end

end #module
