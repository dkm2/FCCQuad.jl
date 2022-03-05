#=
Variations on Filon-Clenshaw-Curtis quadrature.
=#
module FCCQuad
export fccquad

include("Fct.jl")
include("Cheb.jl")
include("chirps.jl")
include("chebweights.jl")
include("Jets.jl")

using LinearAlgebra, Scratch

#=
Filon-Clenshaw-Curtis quadrature for f(x) on [-1,1].
Assuming chebfun is the coefficient vector of
a Chebyshev expansion of degree at least N of some function f(x),
returns f(x)*exp(im*w*x)*dx integrated over [-1,1]
using degree N, and discrepancy between that and using reduced_degree(N).
WARNING: N and baseN are assumed to be even.
=#
function fccquadUnit!(chebfun::AbstractVector,freq::Real,N::Integer,
                      weightmethod::Symbol,weights::AbstractVector,weights_workspace::AbstractArray,
                      baseN=reduced_degree(N))
    getweights!(weights,weights_workspace,length(weights)-1,freq,weightmethod)
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
Filon-Clenshaw-Curtis quadrature for f(x) with arbitrary finite interval domain.
Assuming chebfun is the coefficient vector of
a Chebyshev expansion of degree at least N of some function g(x)=f(c+rx),
returns the integral over [-1,1] of f(c+rx)*exp(im*(w*(c+rx)+v*rx))*d(c+rx)
using degree N, and discrepancy between that and using degree baseN.
Above, w=freq, c=center, r=radius, and v=doppler.
WARNING: N and baseN are assumed to be even
=#
function fccquadSampled!(chebfun::AbstractVector,freq::Real,N::Integer,
                         center::Real,radius::Real,doppler::Real,
                         weightmethod::Symbol,weights::AbstractVector,weights_workspace::AbstractArray,
                         baseN::Integer=reduced_degree(N))
    w = freq*radius + doppler
    interval_transform(fccquadUnit!(chebfun,w,N,weightmethod,weights,weights_workspace,baseN)...,
                       freq,center,radius)
end
function interval_transform(full,diff,freq,center,radius)
    si,co=sincos(freq*center)
    mult = radius*complex(co,si)
    full*mult, diff*mult
end

#Integrates f(x)*exp(im*w*x)*dx over [xmin,xmax] for each w in freqs
#using degree N and discrepancy between that and using reduced_degree(N).
#WARNING: log2N is assumed to be at least 2.
function fccquadBatch(f::Function,freqs::AbstractArray,log2N::Integer;
                      weightmethod::Symbol=:thomas,T::Type=Complex{Float64},
                      xmin::Real=-1.0,xmax::Real=1.0)
    xminT,xmaxT = real(T)(xmin),real(T)(xmax)
    center = 0.5(xmaxT + xminT)
    radius = 0.5(xmaxT - xminT)
    output,workspaces = fccquad_alloc(freqs,log2N,weightmethod,T)
    fccquadBatch!(output,workspaces,center,radius,f,freqs,log2N,weightmethod),1+(1<<log2N)
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

#Degree-adaptive FCC quadrature.
function adaptdegree(f::Function,freqs::AbstractArray;T::Type=Complex{Float64},maxdegree=1<<20,
                     xmin=-1,xmax=1,reltol::Real=1e-8,abstol::Real=0.0,
                     weightmethod::Symbol=:thomas,vectornorm=LinearAlgebra.norm)
    xminT,xmaxT = real(T)(xmin),real(T)(xmax)
    center = 0.5(xmaxT + xminT)
    radius = 0.5(xmaxT - xminT)
    g(x) = f(x*radius + center)
    N = 16
    samples=Fct.chebsample(g,N;T=T)
    output=Matrix{T}(undef,2,length(freqs))
    while true
        chebfun=Fct.chebcoeffs(samples)
        weights,workspace=weights_alloc(N,weightmethod,real(T))[1:2]
        for m in 1:length(freqs)
            output[:,m] = collect(fccquadSampled!(chebfun,freqs[m],N,center,radius,0.0,
                                                  weightmethod,weights,workspace))
        end
        base=vectornorm(view(output,1,:))
        delta=vectornorm(view(output,2,:))
        if (delta <= base * reltol || delta <= abstol) || N >= maxdegree
            break
        end
        samples = Fct.doublesample(g,samples)
        N<<=1
    end
    output,N+1
end

#FCC quadrature with tone removal.
#Expects output,workspaces = fccquad_alloc(freqs,log2N,weightmethod,T).
#WARNING: log2N is assumed to be at least 2.
function tonequad!(output::AbstractArray,workspaces,center::Real,radius::Real,
                   prefactor::Function,oscillator::Function,
                   freqs::AbstractArray,log2N::Integer,weightmethod::Symbol,T::Type)
    N=1<<log2N
    g(y)=oscillator(y*radius + center)
    cfreq = Jets.phase_velocity(g(Jets.Jet(zero(real(T)),one(real(T)))))
    function h(y)
        s,c=sincos(cfreq*y)
        prefactor(y*radius + center) * g(y) * complex(c,-s)
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

#FCC quadrature with chirp removal.
#WARNING: log2N is assumed to be at least 2.
function chirpquad!(output::AbstractArray,center::Real,radius::Real,
                    prefactor::Function,oscillator::Function,
                    freqs::AbstractArray,log2N::Integer,weightmethod::Symbol,T::Type)
    N=1<<log2N
    g(x)=oscillator(x*radius + center)
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
        prefactor(y*radius + center) * g(y) * complex(c,-s)
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

#=
Integrates f(x)*exp(im*w*x)*dx over [xmin,xmax] for all w in freqs
where f(x)=prefactor(x)*oscillator(x), 
using a variant of Filon-Chenshaw-Curtis quadrature.
The method keyword argument specifies the variant.

Outputs a tuple (results, function_evaluation_count) where results
is a 2xL complex matrix, where L=length(freqs), 
with row 1 containing the estimated integrals
and row 2 the discrepancies between row 1 and estimates made
using a reduced Chebyshev interpolation degree.

WARNING: oscillator(x) is assumed to be nonzero on [xmin,xmax].
WARNING: log2degree is assumed to be at least 2.
=#
function fccquad(prefactor::Function,oscillator::Function,freqs::AbstractArray{<:Real};
                 xmin::Real=-1.0,xmax::Real=1.0,
                 reltol::Real=1e-8,abstol::Real=0.0,T::Type=Complex{Float64},
                 method::Symbol=:tone,weightmethod=:thomas,vectornorm=LinearAlgebra.norm,
                 log2degree::Integer=6,maxdegree::Integer=1<<20)
    product(x) = prefactor(x) * oscillator(x)
    if method == :degree
        return adaptdegree(product,freqs;
                           T=T,maxdegree=maxdegree,weightmethod=weightmethod,
                           xmin=xmin,xmax=xmax,reltol=reltol,abstol=abstol,
                           vectornorm=vectornorm)
    end
    log2N = log2degree
    if method == :nonadaptive
        f(x) = prefactor(x) * oscillator(x)
        return fccquadBatch(product,freqs,log2N;
                            T=T,xmin=xmin,xmax=xmax,weightmethod=weightmethod)
    end
    output=zeros(T,2,length(freqs))
    xminT,xmaxT = real(T)(xmin),real(T)(xmax)
    center = 0.5(xmax + xmin)
    radius = 0.5(xmax - xmin)
    if method == :chirp
        subintegrals,workspaces = Matrix{T}(undef,2,length(freqs)),nothing
    else # :plain, :tone
        subintegrals,workspaces = fccquad_alloc(freqs,log2N,weightmethod,T)
    end
    interval_adaptive!(output,subintegrals,workspaces,center,radius,
                       prefactor,oscillator,product,
                       freqs,log2N,reltol,abstol,
                       method,weightmethod,T,vectornorm)
end
#adds results in place to output
function interval_adaptive!(output::AbstractArray,subintegrals::AbstractArray,workspaces,
                            center::Real,radius::Real,
                            prefactor::Function,oscillator::Function,product::Function,
                            freqs::AbstractArray,log2N::Integer,reltol::Real,abstol::Real,
                            interpolation::Symbol,weightmethod::Symbol,T::Type,vectornorm)
    if interpolation == :chirp #Filon-Clenshaw-Curtis quadrature with linear chirp removal
        chirpquad!(subintegrals,center,radius,
                   prefactor,oscillator,freqs,log2N,weightmethod,T)
    elseif interpolation == :tone #Filon-Clenshaw-Curtis quadrature with tone removal
        tonequad!(subintegrals,workspaces,center,radius,
                  prefactor,oscillator,freqs,log2N,weightmethod,T)
    else # :plain Filon-Clenshaw-Curtis quadrature
        fccquadBatch!(subintegrals,workspaces,center,radius,
                      product,freqs,log2N,weightmethod)
    end
    base=vectornorm(view(subintegrals,1,:))
    delta=vectornorm(view(subintegrals,2,:))
    evals = isfinite(delta) ? 1+(1<<log2N) : 1
    if delta <= base * reltol || delta  <= abstol
        output .+= subintegrals
    else
        r = 0.25radius
        abstol = 0.25max(abstol, base * reltol)
        for t in -3:2:3
            evals += interval_adaptive!(output,subintegrals,workspaces,r*t+center,r,
                                        prefactor,oscillator,product,
                                        freqs,log2N,reltol,abstol,
                                        interpolation,weightmethod,T,vectornorm
                                        )[2]
        end
    end
    output,evals
end


#=
Integrates f(x)*cos(w*x)*dx over [xmin,xmax] for all w in freqs
where f(x)=amplitude(x)*cos(angle(x)), 
using a variant of Filon-Chenshaw-Curtis quadrature.
The method keyword argument specifies the variant.

Outputs a tuple (results, function_evaluation_count) where results
is a 2xL real matrix, where L=length(freqs), 
with row 1 containing the estimated integrals
and row 2 the discrepancies between row 1 and estimates made
using a reduced Chebyshev interpolation degree.

WARNING: amplitude(x) and angle(x) are assumed to be real-valued.
WARNING: log2degree is assumed to be at least 2.
=#
function cosfccquad(amplitude::Function,angle::Function,freqs::AbstractArray{<:Real};
                    T::Type=Float64,vectornorm=LinearAlgebra.norm,kwargs...)
    symmetricfreqs = Vector{eltype(freqs)}(undef,2length(freqs))
    for i in 1:length(freqs)
        w = freqs[i]
        symmetricfreqs[2i-1],symmetricfreqs[2i] = w,-w
    end
    workspace = Vector{T}(undef,length(freqs))
    pseudonorm(v) = cosnorm!(v, vectornorm, workspace)
    
    oscillator(x) = exp(im*angle(x))
    cis,evals = fccquad(amplitude,oscillator,symmetricfreqs;T=Complex{T},vectornorm=pseudonorm,kwargs...)
    output = Matrix{T}(undef,2,length(freqs))
    for i in 1:2
        cis2cos!(view(cis,i,:),view(output,i,:))
    end
    output,evals
end
function cosnorm!(v, vectornorm, workspace)
    cis2cos!(v, workspace)
    vectornorm(workspace)
end
function cis2cos!(exps, cosines)
    for i in 1:length(cosines)
        cosines[i] = 0.5(real(exps[2i-1])+real(exps[2i]))
    end
end

#=
Integrates f(x)*sin(w*x)*dx over [xmin,xmax] for all w in freqs
where f(x)=amplitude(x)*sin(angle(x)), 
using a variant of Filon-Chenshaw-Curtis quadrature.
The method keyword argument specifies the variant.

Outputs a tuple (results, function_evaluation_count) where results
is a 2xL real matrix, where L=length(freqs), 
with row 1 containing the estimated integrals
and row 2 the discrepancies between row 1 and estimates made
using a reduced Chebyshev interpolation degree.

WARNING: amplitude(x) and angle(x) are assumed to be real-valued.
WARNING: log2degree is assumed to be at least 2.
=#
function sinfccquad(amplitude::Function,angle::Function,freqs::AbstractArray{<:Real};
                    T::Type=Float64,vectornorm=LinearAlgebra.norm,kwargs...)
    symmetricfreqs = Vector{eltype(freqs)}(undef,2length(freqs))
    for i in 1:length(freqs)
        w = freqs[i]
        symmetricfreqs[2i-1],symmetricfreqs[2i] = w,-w
    end
    workspace = Vector{T}(undef,length(freqs))
    pseudonorm(v) = sinnorm!(v, vectornorm, workspace)
    
    oscillator(x) = exp(im*angle(x))
    cis,evals = fccquad(amplitude,oscillator,symmetricfreqs;T=Complex{T},vectornorm=pseudonorm,kwargs...)
    output = Matrix{T}(undef,2,length(freqs))
    for i in 1:2
        cis2sin!(view(cis,i,:),view(output,i,:))
    end
    output,evals
end
function sinnorm!(v, vectornorm, workspace)
    cis2sin!(v, workspace)
    vectornorm(workspace)
end
function cis2sin!(exps, sines)
    for i in 1:length(sines)
        sines[i] = 0.5(imag(exps[2i-1])-imag(exps[2i]))
    end
end

function __init__()
    cache_dir = @get_scratch!("precomputed_chebyshev_weights")
    cache_file = joinpath(cache_dir,"chirps.h5")
    isfile(cache_file) || Chirps.store(cache_file)
    Chirps.load(cache_file)
end

end #module
