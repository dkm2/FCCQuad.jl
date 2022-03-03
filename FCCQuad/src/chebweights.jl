#=
for n in 0:N, compute integral I_n over [-1,1] of
dx * T_n(x) * (isodd(n) ? sin : cos)(w*x) where T_n
is the nth Chebyshev polynomial T_n(cos(x))=cos(n*x).

We use a three-term recurrence relation
found in, for example, (Evans and Webster, 1999).
When N<=|w|, forward recursion is fast and stable.

When |w|<N, forward (and backward) recursion are unstable
but, starting from the true boundary condition for
I_{S-1} where S=ceil(|w|) and using a false boundary
condition I_{N+P}=0 for P large enough, the Thomas triadiagonal
solver algorithm is fast and accurate.

For testing purposes, I also implemented a slower solution 
involving QR decomposition with pivoting
to solve the recurrence relations with the true boundary condition
for I_0 the false boundary condition I_{N+P}=0, for P large enough.
(QR w/o pivoting and LU, even with pivoting, 
are not inaccurate for some choices of w.)
=#
function getweights(N::Integer,w::Real,method::Symbol=:thomas,T::Type=Float64)::Vector
    out,work,paddedN=weights_alloc(N,method,T)
    getweights!(out,work,paddedN,w,method)
end
padding(N) = 9+2round(Int64,0.5+sqrt(18.5N),RoundUp)
function weights_alloc(N::Integer,method::Symbol,T::Type=Float64)
    if method == :qr || method == :thomas
        N += padding(N)
    end
    out = Vector{T}(undef, N+1)
    if method == :qr
        work = Matrix{T}(undef,N+1,N+2)
    elseif method == :thomas
        work = Matrix{T}(undef,2,N+1)
    else
        work = Matrix{T}(undef,0,0)
    end
    out,work,N
end
function getweights!(out::AbstractArray,work::AbstractArray,N::Integer,w::Real,method::Symbol)
    N>=0 || return out
    w = Float64(w)
    
    if iszero(w) #Clenshaw-Curtis weights
        out[1] = 2.0 #I_0
        for n in 2:2:N
            out[n] = 0.0 #I_{n-1}
            out[n+1] = 2.0 / (1 - n^2) #I_n
        end
        if isodd(N)
            out[N+1] = 0.0 #I_N
        end
        return out
    end
    
    s,c,rw = sincos(w)...,inv(w)
    out[1] = 2s*rw #wI_0 = 2sin(w)
    N >= 1 || return out

    #slow QR decomposition with pivoting
    method == :qr && return getweightsQR!(out,work,N,w,s,c)
    
    #stable forward recursion for n < n0
    n0 = min(round(Int64,abs(w),RoundUp),N+1) 
    n0 >=2 && getweightsForward!(out,1,n0-1,rw,s,c)
    n0 <= N || return out
    if n0 <= N #use method for n0 <= n <= N
        if method == :thomas
            #fast Thomas algorithm (how stable?)
            getweightsThomas!(out,work,n0,N,w,s,c)
        elseif method == :forward
            #unstable forward recursion
            getweightsForward!(out,n0,N,rw,s,c)
        end
    end
    out
end

#=
Evans-Webster (1999) forward-recursive computation of I_n(w) for nmin <= n <= nmax. 
Assumes w!=0 and 1 <= nmin <= nmax < length(out).
Assumes out[n+1]=I_n(w) initialized for max(0,nmin-2) <= n < nmin.
WARNING: unstable if nmax > |w|.
=#
function getweightsForward!(out::AbstractArray,nmin::Integer,nmax::Integer,
                            invw::Real,sinw::Real,cosw::Real)
    @assert 1 <= nmin <= nmax < length(out)

    if nmin <= 2
        if nmin <= 1
            out[2] = (out[1] - 2.0cosw)*invw #I_0 - wI_1 = 2cos(w)
            2 <= nmax || return out
        end
        out[3] = (2.0sinw - 4.0out[2])*invw #4I_1 + wI_2 = 2sin(w)
        3 <= nmax || return out
    end
    nstart=max(nmin,3)

    s4w,c4w = -4.0sinw*invw,4.0cosw*invw

    #n odd: wI_{n-2}/(2(n-2)) + I_{n-1} - wI_n/(2n) = -2cos(w)/(n(n-2))
    #out[n+1] = I_n = (nI_{n-2} + 4cos(w)/w)/(n-2) + 2nI_{n-1}/w
    getodd(n) = (n*out[n-1]+c4w)/(n-2) + 2n*out[n]*invw
    
    #n even: wI_{n-2}/(2(n-2)) - I_{n-1} - wI_n/(2n) = 2sin(w)/(n(n-2))
    #out[n+1] = I_n = (nI_{n-2} - 4sin(w)/w)/(n-2) - 2nI_{n-1}/w
    geteven(n) = (n*out[n-1]+s4w)/(n-2) - 2n*out[n]*invw

    isodd(nstart) && (out[nstart+1] = getodd(nstart))
    even_start = ((nstart+1)>>1)<<1 #round up
    for n in even_start:2:nmax-1
        out[n+1] = geteven(n)
        out[n+2] = getodd(n+1) 
    end
    iseven(nmax) && (out[nmax+1] = geteven(nmax))
    out
end
    
#=
Evans-Webster (1999): Olver's method for I_n(w) for nmin <= n <= nmax,
specifically using Thomas' tridiagonal solver algorithm.
Assumes that 1 <= nmin <= nmax < length(out),
that work has at least 2 rows and nmax-nmin+1 columns, and
that out[nmin]=I_{nmin-1}(w) is initialized.
WARNING: unstable if nmin < |w|.
WARNING: uses Olver's method, making a tail of the output inaccurate.
=#
function getweightsThomas!(out::AbstractArray,work::AbstractArray,nmin::Integer,nmax::Integer,
                           w::Real,sinw::Real,cosw::Real)
    @assert 1 <= nmin <= nmax < length(out)
    sup = view(work,1,:) #superdiagonal after row operations
    rhs = view(work,2,:) #righthand side after row operations
    #1 == n odd:                      I_1 + wI_{1+1}/(2(1+1)) = 2sin(w)/4
    #2 <= n odd: -wI_{n-1}/(2(n-1)) + I_n + wI_{n+1}/(2(n+1)) = 2sin(w)/(1-n^2)
    #2 <= n even: wI_{n-1}/(2(n-1)) + I_n - wI_{n+1}/(2(n+1)) = 2cos(w)/(1-n^2)    
    if nmin < 2
        sup[1] = sup_n = 0.25w
        rhs[1] = rhs_n = 0.5sinw
    else
        sup_n, rhs_n = 0.0, out[nmin]
    end
    p,q = 4.0cosw,-4.0sinw
    sign = 1
    nstart = max(nmin,2)
    if isodd(nstart)
        p,q = q,p
        sign = -sign
    end
    for n in nstart:nmax
        r = inv((n+1)*(2sign*(1-n) + w*sup_n))
        i = n-nmin+1
        sup[i] = sup_n = r*(n-1)*w;
        rhs[i] = rhs_n = r*(p + (n+1)*w*rhs_n);
        sign = -sign
        p,q = q,p
    end
    solution_n = 0.0 #todo: DGZ11 estimate of I_{nmax+1}?
    for n in nmax:-1:nmin
        i = n-nmin+1
        out[n+1] = solution_n = rhs[i] - sup[i]*solution_n
    end
    out
end

#= 
Assumes N>=1, w!=0, that work has >=N+1 rows and >=N+2 columns, and that length(out)>=N+1. 
Evans-Webster (1999) computation of I_n(w) for 0 <= n <= N,
but using QR with pivoting to solve the system. Stable but slow.
Assumes w!=0 and 2 <= N < length(out).
WARNING: uses Olver's method, making a tail of the output inaccurate.
=#
function getweightsQR!(out::AbstractArray,work::AbstractArray,N::Integer,
                        w::Real,sinw::Real,cosw::Real)
    #system of equations A*out = b; out=[I_0,...,I_N]
    A = view(work,1:N+1,1:N+1)
    A .= 0.0
    b = view(work,1:N+1,N+2)
    
    #wI_0 = 2sin(w)
    #I_0 - wI_1 = 2cos(w)
    #4I_1 + wI_2 = 2sin(w)
    s2 = 2.0sinw
    c2 = 2.0cosw
    negw = -w
    A[1,1], b[1] = w, s2
    A[2,1], A[2,2], b[2] = 1.0, negw, c2
    A[3,2], A[3,3], b[3] = 4.0, w, s2

    for n in 2:2:N-2
        #n even: wI_{n-1}/(2(n-1)) + I_n - wI_{n+1}/(2(n+1)) = 2cos(w)/(1-n^2)
        #n+1 odd: -wI_n/(2n) + I_{n+1} + wI_{n+2}/(2(n+2)) = 2sin(w)/(1-(n+1)^2)
        A[n+2,n] = w/(2(n-1))
        A[n+2,n+1] = 1.0
        A[n+2,n+2] = negw/(2(n+1))
        b[n+2] = c2/(1-n^2)
        A[n+3,n+1] = negw/(2n)
        A[n+3,n+2] = 1.0
        A[n+3,n+3] = w/(2(n+2))
        b[n+3] = s2/(1-(n+1)^2)
    end
    if isodd(N)
        #N-1 even: wI_{N-2}/(2(N-2)) + I_{N-1} - wI_N/(2N) = 2cos(w)/(1-(N-1)^2)
        A[N+1,N-1] = w/(2(N-2))
        A[N+1,N] = 1.0
        A[N+1,N+1] = negw/(2N)
        b[N+1] = c2/(N*(2-N))
    end
    #println(cond(A))
    ldiv!(out,qr!(A,Val(true)),b) #QR with pivoting helps in ill-conditioned cases
    out
end

