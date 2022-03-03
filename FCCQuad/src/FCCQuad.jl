module FCCQuad

include("Fct.jl")
include("Cheb.jl")
include("chirps.jl")
include("chebweights.jl")
include("Jets.jl")

using LinearAlgebra, Scratch

#=
Filon-Clenshaw-Curtis quadrature.
Assuming chebfun is the coefficient vector of
a Chebyshev expansion of f(x) of degree at least N,
returns integral of f(x)*exp(im*w*x)*dx over [-1,1]
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
    a=Fct.chebcoeffs(Fct.chebsample(f,N))
    full,diff=fccquad(a,w,N)
    full,abs(diff)
end

function fccquad(f::Function,freqs::AbstractArray,log2N::Integer,T=Complex{Float64})
    N=1<<log2N
    a=Fct.chebcoeffs(Fct.chebsample(f,N,T))
    M=length(freqs)
    result=Array{T}(undef,2,M)
    for m in 1:M
        w = freqs[m]
        result[1,m],result[2,m]=fccquad(a,w,N)
    end
    result
end

#=
Filon-Clenshaw-Curtis quadrature for f(x) on [-1,1].
Assuming chebfun is the coefficient vector of
a Chebyshev expansion of degree at least N of some function f(x),
returns f(x)*exp(im*w*x)*dx integrated over [-1,1]
using degree N, and discrepancy between that and using reduced_degree(N).
WARNING: N and baseN are assumed to be even.
=#
function fccquadUnit!(chebfun::AbstractVector,freq::Real,N::Integer,
                       weightmethod::Symbol,weights::AbstractVector,workspace::AbstractArray,
                       baseN=reduced_degree(N))
    getweights!(weights,workspace,length(weights)-1,freq,weightmethod)
    fcc_core(chebfun,weights,N,baseN)
end    
reduced_degree(N)=div(3N,4)
function fcc_core(chebfun::AbstractVector,weights::Vector,N::Integer,baseN::Integer)
    @assert iseven(N) && iseven(baseN)
    a=chebfun
    i=weights
    base=(sum(a[n]*i[n] for n in baseN+1:-2:1) +
          sum(a[n]*i[n] for n in baseN:-2:2)*im)
    diff=(sum(a[n]*i[n] for n in N+1:-2:baseN+3) +
          sum(a[n]*i[n] for n in N:-2:baseN+2)*im)
    full = base + diff
    full,diff
end
#=
Alternative, more flexible error estimation method
that compares the results of using two user-provided Chebyshev expansions 
instead of one Chebyshev expansion and its truncation.
WARNING: N and baseN are assumed to be even.
=#
function fcc_alt(chebfun::AbstractVector,basefun::AbstractVector,
                  weights::Vector,N::Integer,baseN::Integer)
    @assert iseven(N) && iseven(baseN)
    a,b,i=chebfun,basefun,weights
    full=(sum(a[n]*i[n] for n in N+1:-2:1) +
          sum(a[n]*i[n] for n in N:-2:2)*im)
    base=(sum(b[n]*i[n] for n in baseN+1:-2:1) +
          sum(b[n]*i[n] for n in baseN:-2:2)*im)
    diff = full - base
    full,diff
end

#=
Filon-Clenshaw-Curtis quadrature for f(x) and xf(x) 
with arbitrary finite interval domain.
Assuming chebfun is the coefficient vector of
a Chebyshev expansion of degree at least N of some function g(x)=f(c+rx),
returns the integral over [-1,1] of f(c+rx)*exp(2pi*im*(w*(c+rx)+v*rx))*d(c+rx)
using degree N, and discrepancy between that and using degree baseN.
Above, w=freq, c=center, r=radius, and v=doppler.
WARNING: N and baseN are assumed to be even
=#
function fccquadSampled(chebfun::AbstractVector,freq::Real,N::Integer,center::Real,radius::Real,
                        doppler::Real=0.0,weightmethod::Symbol=:thomas,baseN::Integer=reduced_degree(N))
    T = promote_type(eltype(freq),typeof(center),typeof(radius),typeof(doppler))
    weights,workspace=weights_alloc(N,weightmethod,T)[1:2]
    fccquadSampled!(chebfun,freq,N,center,radius,doppler,weightmethod,weights,workspace,baseN)
end
function fccquadSampled!(chebfun::AbstractVector,freq::Real,N::Integer,
                          center::Real,radius::Real,doppler::Real,
                          weightmethod::Symbol,weights::AbstractVector,workspace::AbstractArray,
                          baseN::Integer=reduced_degree(N))
    w = freq*radius + doppler
    interval_transform(fccquadUnit!(chebfun,w,N,weightmethod,weights,workspace,baseN)...,freq,center,radius)
end
function interval_transform(full,diff,freq,center,radius)
    si,co=sincos(freq*center)
    mult = radius*complex(co,si)
    full*mult, diff*mult
end

#=
#Integral of f(x)*exp(im*w*x)*dx over [xmin,xmax]
#computed using degree-N Filon-Clensaw-Curtis quadrature.
#WARNING: N is assumed to be a power of 2; baseN is assumed to be even.
=#
function fccquadLimits(f::Function,xmin::Real,xmax::Real,freq::Real,
                        N::Integer,weightmethod::Symbol,baseN=reduced_degree(N))
    center, radius = 0.5(xmin+xmax), 0.5(xmax-xmin)
    g(x)=f(x*radius + center)
    a=Fct.chebcoeffs(Fct.chebsample(g,N))
    fccquadSampled(a,freq,N,center,radius,0.,weightmethod,baseN)
end

#Returns f(x)*exp(im*w*x) over -1,1 using degree N
#and discrepancy between that and using reduced_degree(N).
#WARNING: log2N is assumed to be at least 2.
function fccquadBatch(f::Function,freqs::AbstractArray,log2N::Integer,weightmethod::Symbol;
                       T::Type=Complex{Float64})
    output,workspaces = fccquad_alloc(freqs,log2N,weightmethod,T)
    fccquadBatch!(output,workspaces,zero(real(T)),one(real(T)),f,freqs,log2N,weightmethod)
end
function fccquad_alloc(freqs::AbstractArray,log2N::Integer,weightmethod::Symbol,T::Type=Complex{Float64})
    output=Matrix{T}(undef,2,length(freqs))
    N=1<<log2N
    samples = Vector{T}(undef,1+N)
    fct_workspaces = Fct.fct_alloc(samples,T)
    weights,weights_workspace = weights_alloc(N,weightmethod,real(T))[1:2]
    workspaces = (samples,fct_workspaces),(weights,weights_workspace)
    output,workspaces
end
function fccquadBatch!(output::AbstractArray,workspaces,center::Real,radius::Real,
                        f::Function,freqs::AbstractArray,log2N::Integer,
                        weightmethod::Symbol,baseN=reduced_degree(1<<log2N))
    N=1<<log2N
    g(x)=f(x*radius + center)
    samples,fct_workspaces = workspaces[1]
    a=Fct.chebcoeffs!(Fct.chebsample!(g,samples,N),fct_workspaces...)
    M=length(freqs)
    weights,weights_workspace = workspaces[2]
    for m in 1:M
        output[:,m] = collect(fccquadSampled!(a,freqs[m],N,center,radius,0.,
                                              weightmethod,weights,weights_workspace,baseN))
    end
    output
end

function adaptdegree(f::Function,freqs::AbstractArray;T::Type=Complex{Float64},maxdegree=1<<20,
                     xmin=-1,xmax=1,reltol::Real=1e-8,abstol::Real=0.0)
    xminT,xmaxT = real(T)(xmin),real(T)(xmax)
    center = 0.5(xmaxT + xminT)
    radius = 0.5(xmaxT - xminT)
    g(x) = f(x*radius + center)
    N = 16
    samples=Fct.chebsample(g,N;T=T)
    output=Array{T}(undef,2,length(freqs))
    while true
        chebfun=Fct.chebcoeffs(samples)
        weights,workspace=weights_alloc(N,:thomas,real(T))[1:2]
        for m in 1:length(freqs)
            output[:,m] = collect(fccquadSampled!(chebfun,freqs[m],N,center,radius,0.0,
                                                  :thomas,weights,workspace))
        end
        base=norm(view(output,1,:))
        delta=norm(view(output,2,:))
        if (delta <= base * reltol || delta * radius <= abstol) || N >= maxdegree
            break
        end
        samples = Fct.doublesample(g,samples)
        N<<=1
    end
    output,N+1
end

#Returns f(x)*exp(im*w*x) over -1,1 using degree N
#and discrepancy between that and using degree N/2.
#Uses tone (linear phase) removal.
#WARNING: log2N is assumed to be at least 2.
function tonequad(f::Function,freqs::AbstractArray,log2N::Integer,weightmethod::Symbol=:thomas;
                  T::Type=Complex{Float64})
    output,workspaces = fccquad_alloc(freqs,log2N,weightmethod,T)
    tonequad!(output,workspaces,zero(real(T)),one(real(T)),f,freqs,log2N,weightmethod)
end
function tonequad!(output::AbstractArray,workspaces,center::Real,radius::Real,
                   f::Function,freqs::AbstractArray,log2N::Integer,weightmethod::Symbol,T::Type)
    N=1<<log2N
    g(x)=f(x*radius + center)
    cfreq = Jets.phase_velocity(g(Jets.Jet(zero(real(T)),one(real(T)))))
    function h(y)
        s,c=sincos(cfreq*y)
        g(y)*complex(c,-s)
    end
    samples,fct_workspaces = workspaces[1]
    a=Fct.chebcoeffs!(Fct.chebsample!(h,samples,N),fct_workspaces...)
    weights,weights_workspace = workspaces[2]
    for m in 1:length(freqs)
        output[:,m] = collect(fccquadSampled!(a,freqs[m],N,center,radius,cfreq,
                                              weightmethod,weights,weights_workspace))
    end
    output
end

#Returns f(x)*exp(im*w*x) over -1,1 using degree N
#and discrepancy between that and using degree N/2.
#Uses linear chirp removal.
#WARNING: log2N is assumed to be at least 2.
function chirpquad(f::Function,freqs::AbstractArray,log2N::Integer,weightmethod::Symbol=:thomas;
                   T::Type=Complex{Float64})
    output=Matrix{T}(undef,2,length(freqs))
    chirpquad!(output,zero(real(T)),one(real(T)),f,freqs,log2N,weightmethod;T=T)
end
function chirpquad!(output::AbstractArray,center::Real,radius::Real,
                    f::Function,freqs::AbstractArray,log2N::Integer,weightmethod::Symbol;
                    T::Type=Complex{Float64})
    N=1<<log2N
    g(x)=f(x*radius + center)
    jet = g(Jets.Jet(zero(real(T)),one(real(T))))
    cfreq::Real = Jets.phase_velocity(jet)
    chirp::Real = Jets.phase_acceleration(jet)
    if Chirps.rates[end] < abs(chirp)
        #give up, output garbage
        output[:,1:length(freqs)-1] .= zero(eltype(output))
        output[1,end] = one(eltype(output))
        output[2,end] = Inf
        return output
    end    
    function h(y)
        s,c=sincos(y*(cfreq+y*chirp))
        g(y)*complex(c,-s)
    end
    a=Cheb.ChebSeries(Fct.chebcoeffs(Fct.chebsample(h,N;T=T)))

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

    weights,workspace,deg4=weights_alloc(deg3,weightmethod,real(T))
    for m in 1:length(freqs)
        w = freqs[m]*radius + cfreq
        getweights!(weights,workspace,deg4,w,weightmethod)
        step1 = fcc_alt(ab.coeff,ab2.coeff,weights,deg1,deg2)
        step2 = interval_transform(step1...,freqs[m],center,radius)
        output[:,m] = collect(step2)
    end
    output
end

#Adaptively integrates f(x)*exp(im*w*x) for a batch of w, for i=0,1.
function adaptquad(f::Function,freqs::AbstractArray,log2N::Integer;
                   xmin=-1.0,xmax=1.0,reltol=1e-8,abstol=0.0,
                   interpolation=:tone,weightmethod=:thomas,T::Type=Complex{Float64})
    output=zeros(T,2,length(freqs))
    xminT,xmaxT = real(T)(xmin),real(T)(xmax)
    center = 0.5(xmax + xmin)
    radius = 0.5(xmax - xmin)
    if interpolation == :chirp
        subintegrals,workspaces = Array{T}(undef,2,length(freqs)),nothing
    else
        subintegrals,workspaces = fccquad_alloc(freqs,log2N,weightmethod,T)
    end
    adaptquad!(output,subintegrals,workspaces,center,radius,f,freqs,
               log2N,reltol,abstol,interpolation,weightmethod,T)
end
#adds results in place to output
function adaptquad!(output::AbstractArray,subintegrals::AbstractArray,workspaces,
                    center::Real,radius::Real,f::Function,freqs::AbstractArray,
                    log2N::Integer,reltol::Real,abstol::Real,
                    interpolation::Symbol,weightmethod::Symbol,T::Type)
    if interpolation == :chirp #Filon-Clenshaw-Curtis quadrature with linear chirp removal
        chirpquad!(subintegrals,center,radius,f,freqs,log2N,weightmethod,T)
    elseif interpolation == :tone #Filon-Clenshaw-Curtis quadrature with tone removal
        tonequad!(subintegrals,workspaces,center,radius,f,freqs,log2N,weightmethod,T)
    else #Filon-Clenshaw-Curtis quadrature (:plain)
        fccquadBatch!(subintegrals,workspaces,center,radius,f,freqs,log2N,weightmethod)
    end
    base=norm(view(subintegrals,1,:))
    delta=norm(view(subintegrals,2,:))
    evals = isfinite(delta) ? 1+(1<<log2N) : 1
    if delta <= base * reltol || delta  <= abstol
        output .+= subintegrals
    else
        r = 0.25radius
        abstol = 0.25max(abstol, base * reltol)
        for t in -3:2:3
            evals += adaptquad!(output,subintegrals,workspaces,r*t+center,r,
                                f,freqs,log2N,reltol,abstol,interpolation,weightmethod,T)[2]
        end
    end
    output,evals
end

function __init__()
    cache_dir = @get_scratch!("precomputed_chebyshev_weights")
    cache_file = joinpath(cache_dir,"chirps.h5")
    isfile(cache_file) || Chirps.store(cache_file)
    Chirps.load(cache_file)
end

end #module
